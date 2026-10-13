// tests/artifact/canonical_mapping_test.cpp - the canonical source-type mappings against the reference's answers.
//
// tools/gen_canonical_mapping_golden.py records what tools/canonical_xcheck.py's MAPPINGS encode each source type
// to (the code and scale planes) and what its decode makes of those planes.  This compares the C++ against both,
// for whole tensors of synthetic blocks - because the failures that matter here are the ones that still produce
// plausible weights: a signed Q8_0 code plane, or Q4_0's nibbles read as interleaved.
//
// It also checks that a type with no mapping is REFUSED rather than approximated, which is the reference's own
// contract ("NO MAPPING for %s - refusing to build").
//
//    canonical_mapping_test <tests/artifact/canonical_mapping_golden.txt>
#include "strata/artifact/canonical_mapping.hpp"

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

std::string hex_of(const uint8_t* p, std::size_t n) {
    static const char* digits = "0123456789abcdef";
    std::string s;
    s.reserve(n * 2);
    for (std::size_t i = 0; i < n; ++i) {
        s += digits[p[i] >> 4];
        s += digits[p[i] & 0xF];
    }
    return s;
}

std::string hex_of_f32(const std::vector<float>& v) {
    std::vector<uint8_t> raw(v.size() * 4);
    std::memcpy(raw.data(), v.data(), raw.size());
    return hex_of(raw.data(), raw.size());
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
        std::fprintf(stderr, "usage: canonical_mapping_test <tests/artifact/canonical_mapping_golden.txt>\n");
        return 2;
    }
    std::ifstream in(argv[1]);
    if (!in) {
        std::fprintf(stderr, "cannot read %s\n", argv[1]);
        return 2;
    }
    std::string line;
    int cases = 0;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        std::string what, tname, raw_hex, codes_hex, scales_hex, offsets_hex, values_hex;
        if (!(ss >> what >> tname >> raw_hex >> codes_hex >> scales_hex >> offsets_hex >> values_hex) ||
            what != "case") {
            std::fprintf(stderr, "malformed golden line: %s\n", line.c_str());
            ++failures;
            continue;
        }
        const std::vector<uint8_t> raw = unhex(raw_hex);
        const std::string label = tname + " (" + std::to_string(raw.size()) + " bytes)";

        strata::CanonicalParts parts;
        std::string err;
        if (!strata::canonical_encode(tname, raw.data(), raw.size(), parts, err)) {
            std::fprintf(stderr, "FAIL %s: encode refused: %s\n", label.c_str(), err.c_str());
            ++failures;
            continue;
        }
        // The S forms carry code and scale planes; the P forms a single value plane, so the golden has a `-`.
        if (codes_hex != "-")
            check(label + " codes", hex_of(parts.codes.data(), parts.codes.size()), codes_hex);
        if (scales_hex != "-")
            check(label + " scales", hex_of_f32(parts.scales), scales_hex);
        if (offsets_hex != "-")
            check(label + " offsets", hex_of_f32(parts.offsets), offsets_hex);

        std::vector<float> values;
        if (!strata::canonical_decode(tname, parts, values, err)) {
            std::fprintf(stderr, "FAIL %s: decode refused: %s\n", label.c_str(), err.c_str());
            ++failures;
            continue;
        }
        check(label + " values", hex_of_f32(values), values_hex);
        ++cases;
    }

    // A type with no mapping must be refused, not approximated - the reference's own contract.  All thirteen
    // source types are mapped now, so this is a real ggml type the reference has no mapping for either.
    {
        strata::CanonicalParts parts;
        std::string err;
        const uint8_t block[24] = {0};
        if (strata::canonical_encode("Q4_1", block, sizeof block, parts, err)) {
            std::fprintf(stderr, "FAIL an unmapped type (Q4_1) was accepted instead of refused\n");
            ++failures;
        } else if (err.find("refusing to build") == std::string::npos) {
            std::fprintf(stderr, "FAIL the refusal does not say what it is: %s\n", err.c_str());
            ++failures;
        }
        // ...and a byte count that is not whole blocks is refused too.
        if (strata::canonical_encode("Q8_0", block, 33, parts, err)) {
            std::fprintf(stderr, "FAIL a partial block was accepted\n");
            ++failures;
        }
    }

    if (failures) {
        std::fprintf(stderr, "canonical_mapping_test: FAILED (%d)\n", failures);
        return 1;
    }
    std::printf("canonical_mapping_test: %d source types match the reference's planes and their decode, and an "
                "unmapped type is refused\n", cases);
    return 0;
}
