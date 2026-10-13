// tests/artifact/mtp_rt_test.cpp - the MTP runtime files, against tools/mtp_rt.py's own.
//
// tools/gen_mtp_rt_golden.py builds a fixture carrying the MTP tensors at PRODUCTION geometry (H=2560, FF=640,
// only NE scaled, to 1) and records what the reference wrote: its stdout, `dense.txt` in full, and the digests of
// `experts.bin` and `dense.bin`.  This runs the built tool over the same fixture and compares all of it.
//
// It drives the binary rather than a library, because the printed index IS the acceptance.
//
//    mtp_rt_test <tests/mtp/mtp_rt_golden.txt> <path to strata-mtp-rt>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

std::string hex_of(const uint8_t* p, std::size_t n) {
    static const char* digits = "0123456789abcdef";
    std::string s;
    for (std::size_t i = 0; i < n; ++i) {
        s += digits[p[i] >> 4];
        s += digits[p[i] & 0xF];
    }
    return s;
}

std::string digest_of(const std::string& path) {
    const std::string blob = read_text(path);
    if (blob.empty()) return "(missing)";
    strata::Sha256 h;
    h.update(blob.data(), blob.size());
    const std::array<uint8_t, 32> d = h.finish();
    return hex_of(d.data(), d.size());
}

void check(const std::string& what, const std::string& got, const std::string& want) {
    if (got == want) return;
    std::fprintf(stderr, "FAIL %s\n  got  %s\n  want %s\n", what.c_str(), got.c_str(), want.c_str());
    ++failures;
}

std::string shell_quote(const std::string& s) {
    std::string o = "'";
    for (char c : s) {
        if (c == '\'') o += "'\\''";
        else o += c;
    }
    return o + "'";
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: mtp_rt_test <golden> <strata-mtp-rt>\n");
        return 2;
    }
    const std::string golden_path = argv[1], exe = argv[2];
    const std::string dir = std::filesystem::path(golden_path).parent_path().string();
    const std::string all = read_text(golden_path);
    if (all.empty()) {
        std::fprintf(stderr, "cannot read %s\n", golden_path.c_str());
        return 2;
    }

    std::string fixture, want_stdout, want_dense_txt;
    std::vector<std::pair<std::string, std::string>> want_files;   // name -> sha256
    std::size_t at = 0;
    while (at < all.size()) {
        const std::size_t nl = all.find('\n', at);
        const std::string line = all.substr(at, nl == std::string::npos ? std::string::npos : nl - at);
        at = nl == std::string::npos ? all.size() : nl + 1;
        if (line.rfind("fixture ", 0) == 0) {
            const std::size_t sp = line.find(' ', 8);
            fixture = line.substr(8, sp == std::string::npos ? std::string::npos : sp - 8);
        } else if (line.rfind("file ", 0) == 0) {
            char name[64] = {0}, sha[80] = {0};
            if (std::sscanf(line.c_str(), "file %63s bytes %*u sha256 %79s", name, sha) == 2)
                want_files.push_back({name, sha});
        } else if (line.rfind("stdout ", 0) == 0) {
            want_stdout += line.substr(7) + "\n";
        } else if (line.rfind("# dense.txt-begin", 0) == 0) {
            want_dense_txt = all.substr(at);   // everything after the marker is the file
            at = all.size();
        }
    }
    if (fixture.empty() || want_files.size() != 2) {
        std::fprintf(stderr, "the golden is not the shape this test expects\n");
        return 1;
    }

    const std::string out = dir + "/.mtp-rt-out";
    std::error_code ec;
    std::filesystem::remove_all(out, ec);
    const std::string cmd = shell_quote(exe) + " --gguf " + shell_quote(dir + "/" + fixture) + " --out " +
                            shell_quote(out) + " 2>/dev/null";
    std::FILE* p = ::popen(cmd.c_str(), "r");
    if (p == nullptr) {
        std::fprintf(stderr, "FAIL cannot run the tool\n");
        return 1;
    }
    std::string got;
    char buf[4096];
    std::size_t n = 0;
    while ((n = std::fread(buf, 1, sizeof buf, p)) > 0) got.append(buf, n);
    if (::pclose(p) != 0) {
        std::fprintf(stderr, "FAIL the tool exited non-zero\n");
        return 1;
    }
    // the tool prints the output directory; the golden records it as <out>
    std::size_t pos = 0;
    while ((pos = got.find(out, pos)) != std::string::npos) {
        got = got.substr(0, pos) + "<out>" + got.substr(pos + out.size());
        pos += 5;
    }
    check("stdout", got, want_stdout);

    for (const auto& [name, sha] : want_files) check(name + " sha256", digest_of(out + "/" + name), sha);
    check("dense.txt", read_text(out + "/dense.txt"), want_dense_txt);
    std::filesystem::remove_all(out, ec);

    if (failures) {
        std::fprintf(stderr, "mtp_rt_test: FAILED (%d)\n", failures);
        return 1;
    }
    std::printf("mtp_rt_test: experts.bin, dense.bin and dense.txt match tools/mtp_rt.py's own, and so does the "
                "printed index\n");
    return 0;
}
