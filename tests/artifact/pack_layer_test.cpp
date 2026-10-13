// tests/artifact/pack_layer_test.cpp - the pack-layout arena TOOL, against tools/pack_layer.py's own output.
//
// tools/gen_pack_layer_golden.py writes a committable fixture GGUF and runs the reference's own `build()` over
// it, recording the per-layer stdout, the arena's size and digest, and the sidecar manifest verbatim.  This runs
// the built tool over the same fixture and compares all three.
//
// This is a different acceptance from expert_arena_test: that one holds the layout FUNCTION, this one holds the
// tool around it - the GGUF read, the per-layer loop, the running offset and the manifest.
//
//    pack_layer_test <golden> <strata-pack-layer>
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
        std::fprintf(stderr, "usage: pack_layer_test <golden> <strata-pack-layer>\n");
        return 2;
    }
    const std::string golden_path = argv[1], exe = argv[2];
    const std::string dir = std::filesystem::path(golden_path).parent_path().string();
    const std::string all = read_text(golden_path);
    if (all.empty()) {
        std::fprintf(stderr, "cannot read %s\n", golden_path.c_str());
        return 2;
    }

    std::string fixture, want_stdout, want_bin, want_manifest;
    std::string layers;
    {
        std::size_t p = 0;
        bool in_manifest = false;
        while (p < all.size()) {
            const std::size_t nl = all.find('\n', p);
            const std::string line = all.substr(p, nl == std::string::npos ? std::string::npos : nl - p);
            p = nl == std::string::npos ? all.size() : nl + 1;
            if (in_manifest) {
                if (line == "manifest_end") {
                    in_manifest = false;
                    if (!want_manifest.empty()) want_manifest.pop_back();   // the newline before the marker
                } else {
                    want_manifest += line + "\n";
                }
                continue;
            }
            if (line.rfind("fixture ", 0) == 0) fixture = line.substr(8);
            else if (line.rfind("layers ", 0) == 0) layers = line.substr(7);
            else if (line.rfind("stdout ", 0) == 0) want_stdout += line.substr(7) + "\n";
            else if (line.rfind("bin ", 0) == 0) want_bin = line.substr(4);
            else if (line == "manifest_begin") in_manifest = true;
        }
    }
    if (fixture.empty() || want_bin.empty() || want_manifest.empty()) {
        std::fprintf(stderr, "the golden is missing a fixture/bin/manifest section\n");
        return 1;
    }

    const std::string model = dir + "/" + fixture;
    const std::string out = dir + "/.pack-layer-out.bin";
    const std::string man = dir + "/.pack-layer-out.json";
    std::error_code ec;
    std::filesystem::remove(out, ec);
    std::filesystem::remove(man, ec);

    const std::string cmd = shell_quote(exe) + " --gguf " + shell_quote(model) + " --out " + shell_quote(out) +
                            " --layers " + layers + " 2>&1";
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
    check("stdout", got, want_stdout);

    // ---- the arena, byte for byte
    const std::string blob = read_text(out);
    if (blob.empty()) {
        std::fprintf(stderr, "FAIL the tool wrote no arena\n");
        return 1;
    }
    strata::Sha256 h;
    h.update(blob.data(), blob.size());
    const std::array<unsigned char, 32> d = h.finish();
    const std::string sha = hex_of(d.data(), d.size());
    {
        const std::size_t b = want_bin.find("bytes ");
        const std::size_t q = want_bin.find("sha256 ");
        check("arena size", std::to_string(blob.size()), want_bin.substr(b + 6, q - 1 - (b + 6)));
        check("arena sha256", sha, want_bin.substr(q + 7));
    }

    // ---- the sidecar manifest, byte for byte
    check("manifest", read_text(man), want_manifest);

    std::filesystem::remove(out, ec);
    std::filesystem::remove(man, ec);

    if (failures) {
        std::fprintf(stderr, "pack_layer_test: FAILED (%d)\n", failures);
        return 1;
    }
    std::printf("pack_layer_test: the arena and its manifest match tools/pack_layer.py's own, byte for byte\n");
    return 0;
}
