// tests/artifact/canonical_codes_test.cpp - the canonical pack's code packing against the reference's answers.
//
// tools/gen_strata_pack_golden.py records what tools/strata_pack.py's own pack_codes / unpack_codes /
// choose_scale_dtype return for a set of code patterns and scales, and this compares the C++ against them.  The
// bit order is the thing worth testing: a packing that is wrong still round-trips against ITSELF, so comparing
// against the reference's bytes is the only check that sees it.
//
//    canonical_codes_test <tests/artifact/strata_pack_golden.txt>
#include "strata/artifact/canonical_codes.hpp"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

int failures = 0;

std::vector<uint8_t> unhex(const std::string& h) {
    static const char* digits = "0123456789abcdef";
    std::vector<uint8_t> out;
    if (h.size() % 2) {
        std::fprintf(stderr, "bad hex (odd length): %s\n", h.c_str());
        ++failures;
        return out;
    }
    for (std::size_t i = 0; i + 1 < h.size(); i += 2) {
        const std::size_t hi = std::string(digits).find(h[i]);
        const std::size_t lo = std::string(digits).find(h[i + 1]);
        if (hi == std::string::npos || lo == std::string::npos) {
            std::fprintf(stderr, "bad hex: %s\n", h.c_str());
            ++failures;
            return out;
        }
        out.push_back((uint8_t)((hi << 4) | lo));
    }
    return out;
}

std::string hex_of(const std::vector<uint8_t>& b) {
    static const char* digits = "0123456789abcdef";
    std::string s;
    s.reserve(b.size() * 2);
    for (uint8_t v : b) {
        s += digits[v >> 4];
        s += digits[v & 0xF];
    }
    return s;
}

void check(const std::string& what, const std::string& got, const std::string& want) {
    if (got != want) {
        std::fprintf(stderr, "FAIL %s\n  got  %s\n  want %s\n", what.c_str(), got.c_str(), want.c_str());
        ++failures;
    }
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: canonical_codes_test <tests/artifact/strata_pack_golden.txt>\n");
        return 2;
    }
    std::ifstream in(argv[1]);
    if (!in) {
        std::fprintf(stderr, "cannot read %s\n", argv[1]);
        return 2;
    }
    std::string line;
    int n_packs = 0, n_scales = 0;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        std::string what;
        ss >> what;
        if (what == "pack") {
            int bits = 0;
            std::size_t n = 0;
            std::string codes_hex, want_hex;
            if (!(ss >> bits >> n >> codes_hex >> want_hex)) {
                std::fprintf(stderr, "malformed pack line: %s\n", line.c_str());
                ++failures;
                continue;
            }
            const std::vector<uint8_t> codes = unhex(codes_hex);
            if (codes.size() != n) {
                std::fprintf(stderr, "pack %d/%zu: the golden's code count does not match\n", bits, n);
                ++failures;
                continue;
            }
            std::vector<uint8_t> packed;
            if (!strata::pack_codes(codes.data(), n, bits, packed)) {
                std::fprintf(stderr, "pack %d/%zu: refused a count the reference accepts\n", bits, n);
                ++failures;
                continue;
            }
            check("pack " + std::to_string(bits) + "/" + std::to_string(n) + " bytes", hex_of(packed), want_hex);
            std::vector<uint8_t> back;
            strata::unpack_codes(packed.data(), packed.size(), bits, n, back);
            check("unpack " + std::to_string(bits) + "/" + std::to_string(n), hex_of(back), codes_hex);
            ++n_packs;
        } else if (what == "scale") {
            int is16 = -1;
            std::string scales_hex, want_hex;
            if (!(ss >> is16 >> scales_hex >> want_hex)) {
                std::fprintf(stderr, "malformed scale line: %s\n", line.c_str());
                ++failures;
                continue;
            }
            const std::vector<uint8_t> raw = unhex(scales_hex);
            std::vector<float> scales(raw.size() / 4);
            std::memcpy(scales.data(), raw.data(), raw.size());
            std::vector<uint8_t> packed;
            const bool got16 = strata::choose_scale_dtype(scales.data(), scales.size(), packed);
            check("scale dtype", got16 ? "1" : "0", std::to_string(is16));
            check("scale bytes", hex_of(packed), want_hex);
            ++n_scales;
        } else {
            std::fprintf(stderr, "unknown golden line: %s\n", line.c_str());
            ++failures;
        }
    }
    if (failures) {
        std::fprintf(stderr, "canonical_codes_test: FAILED (%d)\n", failures);
        return 1;
    }
    std::printf("canonical_codes_test: %d code packings and %d scale choices match the reference\n", n_packs,
                n_scales);
    return 0;
}
