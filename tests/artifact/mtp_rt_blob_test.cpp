// tests/artifact/mtp_rt_blob_test.cpp - the expert blob at PRODUCTION geometry, against tools/mtp_rt.py's own.
//
// `mtp_rt.blob_of` (the MTP draft layer's runtime files, plan v0.3 P6) is a second, independent implementation of
// the relayout `expert_arena::pack_expert_blob` does.  The arena's own golden runs at a scaled geometry
// (H=256/FF=64/NE=2) because a fixture at 2,560x640 is 1.4 MB per expert; this one runs at the geometry the MTP
// layer actually uses, so the port is checked against the reference where it matters.
//
// Both sides generate the same synthetic Q2_0 blocks from a formula, so the golden carries the reference's digest
// rather than the inputs.
//
//    mtp_rt_blob_test <tests/artifact/mtp_rt_blob_golden.txt>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "strata/artifact/expert_arena.hpp"
#include "strata/artifact/sha256.hpp"

namespace {

int failures = 0;

std::string hex_of(const unsigned char* p, std::size_t n) {
    static const char* digits = "0123456789abcdef";
    std::string s;
    s.reserve(n * 2);
    for (std::size_t i = 0; i < n; ++i) {
        s += digits[p[i] >> 4];
        s += digits[p[i] & 0xF];
    }
    return s;
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: mtp_rt_blob_test <tests/artifact/mtp_rt_blob_golden.txt>\n");
        return 2;
    }
    std::ifstream in(argv[1]);
    if (!in) {
        std::fprintf(stderr, "cannot read %s\n", argv[1]);
        return 2;
    }
    strata::ExpertGeometry g;
    std::size_t gu_bytes = 0, dn_bytes = 0, want_blob = 0;
    std::string want_sha;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        char key[32] = {0};
        char value[128] = {0};
        if (std::sscanf(line.c_str(), "%31s %127s", key, value) != 2) continue;
        const std::string k = key, v = value;
        if (k == "H") g.H = std::strtoull(value, nullptr, 10);
        else if (k == "FF") g.FF = std::strtoull(value, nullptr, 10);
        else if (k == "NE") g.NE = std::strtoull(value, nullptr, 10);
        else if (k == "gu_bytes") gu_bytes = std::strtoull(value, nullptr, 10);
        else if (k == "dn_bytes") dn_bytes = std::strtoull(value, nullptr, 10);
        else if (k == "blob_bytes") want_blob = std::strtoull(value, nullptr, 10);
        else if (k == "sha256") want_sha = v;
    }
    if (want_sha.empty() || gu_bytes == 0 || dn_bytes == 0) {
        std::fprintf(stderr, "the golden is not the shape this test expects\n");
        return 1;
    }
    if (g.role_bytes() * 2 != gu_bytes || g.role_bytes() != dn_bytes) {
        std::fprintf(stderr, "FAIL the golden's byte counts do not match the geometry: role %llu, gu %zu, dn %zu\n",
                     (unsigned long long) g.role_bytes(), gu_bytes, dn_bytes);
        return 1;
    }

    // the inputs, by the golden's formula: identical on both sides
    std::vector<uint8_t> gu(gu_bytes), dn(dn_bytes);
    for (std::size_t i = 0; i < gu.size(); ++i) gu[i] = (uint8_t) ((i * 37 + 11) & 0xFF);
    for (std::size_t i = 0; i < dn.size(); ++i) dn[i] = (uint8_t) ((i * 37 + 11) & 0xFF);

    // gate is the first FF rows of gu and up the next FF - the reference's `gu` holds both, stacked
    std::vector<uint8_t> blob((std::size_t) g.blob_bytes());
    strata::pack_expert_blob(gu.data(), gu.data() + g.role_bytes(), dn.data(), g, blob.data());

    if (blob.size() != want_blob) {
        std::fprintf(stderr, "FAIL blob is %zu bytes, the reference's is %zu\n", blob.size(), want_blob);
        ++failures;
    }
    strata::Sha256 h;
    h.update(blob.data(), blob.size());
    const std::array<uint8_t, 32> d = h.finish();
    const std::string got = hex_of(d.data(), d.size());
    if (got != want_sha) {
        std::fprintf(stderr, "FAIL the blob differs from tools/mtp_rt.py's own:\n  got  %s\n  want %s\n",
                     got.c_str(), want_sha.c_str());
        ++failures;
    }

    if (failures) {
        std::fprintf(stderr, "mtp_rt_blob_test: FAILED (%d)\n", failures);
        return 1;
    }
    std::printf("mtp_rt_blob_test: the blob matches tools/mtp_rt.py's own at H=%llu FF=%llu NE=%llu "
                "(%zu bytes)\n", (unsigned long long) g.H, (unsigned long long) g.FF, (unsigned long long) g.NE,
                blob.size());
    return 0;
}
