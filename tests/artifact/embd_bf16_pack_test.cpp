// tests/artifact/embd_bf16_pack_test.cpp - the BF16 embedding GGUF, against tools/embd_bf16_pack.py's own.
//
// tools/gen_safetensors_pack_golden.py builds a committable checkpoint and records what the reference wrote over
// it: its stdout, the output's size and digest, and its first bytes.  This runs the built tool over the same
// fixture and compares all of it.
//
// The fixture is small (a 4x8 embedding), so the size check is real: the tool has to place the payload at the
// aligned offset the reference chose, and a header that is one byte off fails the digest.
//
//    embd_bf16_pack_test <golden> <strata-embd-bf16-pack>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "strata/artifact/sha256.hpp"

namespace {

int failures = 0;

std::string read_text(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return {};
    return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

std::string hex_of(const unsigned char* p, std::size_t n) {
    static const char* digits = "0123456789abcdef";
    std::string s;
    for (std::size_t i = 0; i < n; ++i) {
        s += digits[p[i] >> 4];
        s += digits[p[i] & 0xF];
    }
    return s;
}

std::string shell_quote(const std::string& s) {
    std::string o = "'";
    for (char c : s) {
        if (c == '\'') o += "'\\''";
        else o += c;
    }
    return o + "'";
}

void check(const std::string& what, const std::string& got, const std::string& want) {
    if (got == want) return;
    std::fprintf(stderr, "FAIL %s\n  got  %s\n  want %s\n", what.c_str(), got.c_str(), want.c_str());
    ++failures;
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: embd_bf16_pack_test <golden> <strata-embd-bf16-pack>\n");
        return 2;
    }
    const std::string golden_path = argv[1], exe = argv[2];
    const std::string dir = std::filesystem::path(golden_path).parent_path().string();
    const std::string all = read_text(golden_path);
    if (all.empty()) {
        std::fprintf(stderr, "cannot read %s\n", golden_path.c_str());
        return 2;
    }

    // the section between "## embd_bf16_pack.py" and the next "## "
    std::string want_stdout, want_file, want_head;
    {
        std::size_t at = all.find("\n## embd_bf16_pack.py");
        if (at == std::string::npos) {
            std::fprintf(stderr, "the golden has no embd_bf16_pack.py section\n");
            return 1;
        }
        at += 1;
        const std::size_t end = all.find("\n## ", at + 1);
        const std::string section = all.substr(at, end == std::string::npos ? std::string::npos : end - at);
        std::size_t p = 0;
        while (p < section.size()) {
            const std::size_t nl = section.find('\n', p);
            const std::string line = section.substr(p, nl == std::string::npos ? std::string::npos : nl - p);
            p = nl == std::string::npos ? section.size() : nl + 1;
            if (line.rfind("stdout ", 0) == 0) want_stdout += line.substr(7) + "\n";
            else if (line.rfind("file ", 0) == 0) want_file = line.substr(5);
            else if (line.rfind("head_hex ", 0) == 0) want_head = line.substr(9);
        }
    }
    if (want_file.empty()) {
        std::fprintf(stderr, "the embd section carries no file line\n");
        return 1;
    }

    const std::string model = dir + "/safetensors_fixture";
    const std::string out = dir + "/.embd-out.gguf";
    std::error_code ec;
    std::filesystem::remove(out, ec);
    const std::string cmd = shell_quote(exe) + " --model " + shell_quote(model) + " --out " + shell_quote(out) +
                            " 2>&1";
    std::FILE* pipe = ::popen(cmd.c_str(), "r");
    if (pipe == nullptr) {
        std::fprintf(stderr, "FAIL cannot run the tool\n");
        return 1;
    }
    std::string got;
    char buf[4096];
    std::size_t n = 0;
    while ((n = std::fread(buf, 1, sizeof buf, pipe)) > 0) got.append(buf, n);
    if (::pclose(pipe) != 0) {
        std::fprintf(stderr, "FAIL the tool exited non-zero:\n%s\n", got.c_str());
        return 1;
    }
    for (const std::string& path : {out, model}) {   // the longer-lived substitution first
        std::size_t pos = 0;
        while ((pos = got.find(path, pos)) != std::string::npos) {
            got = got.substr(0, pos) + (path == out ? "<out>" : "<model>") + got.substr(pos + path.size());
            pos += 5;
        }
    }
    check("stdout", got, want_stdout);

    const std::string blob = read_text(out);
    if (blob.empty()) {
        std::fprintf(stderr, "FAIL the tool wrote nothing\n");
        return 1;
    }
    strata::Sha256 h;
    h.update(blob.data(), blob.size());
    const std::array<unsigned char, 32> d = h.finish();
    const std::string sha = hex_of(d.data(), d.size());
    char size_line[256];
    std::snprintf(size_line, sizeof size_line, "%s", want_file.c_str());
    std::string want_size;
    {   // "embd.gguf bytes N sha256 H"
        const std::string s(want_file);
        const std::size_t b = s.find("bytes ");
        const std::size_t q = s.find(" sha256 ");
        want_size = s.substr(b + 6, q - (b + 6));
    }
    check("size", std::to_string(blob.size()), want_size);
    {
        const std::string s(want_file);
        const std::size_t q = s.find("sha256 ");
        check("sha256", sha, s.substr(q + 7));
    }
    if (!want_head.empty()) {
        const std::size_t m = std::min<std::size_t>(80, blob.size());
        check("the first bytes", hex_of((const unsigned char*) blob.data(), m), want_head.substr(0, m * 2));
    }
    std::filesystem::remove(out, ec);

    if (failures) {
        std::fprintf(stderr, "embd_bf16_pack_test: FAILED (%d)\n", failures);
        return 1;
    }
    std::printf("embd_bf16_pack_test: the GGUF matches tools/embd_bf16_pack.py's own, byte for byte\n");
    return 0;
}
