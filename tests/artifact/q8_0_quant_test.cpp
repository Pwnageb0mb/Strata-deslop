// tests/artifact/q8_0_quant_test.cpp - the host f32 -> Q8_0 quantizer, against tools/mtp_rt.py's own q8_0.
//
// tools/gen_mtp_rt_golden.py runs the reference's `q8_0` over a deterministic array - sixteenths, exact in
// float32, so both sides build the same input without depending on a libm - and records the byte count, the
// digest and the first block.  The rounding is exercised on exact halves, which is where half-away-from-zero and
// half-even differ: a quantizer that used the fp16 scale's rounding would pass a smooth array and fail this one.
//
//    q8_0_quant_test <tests/mtp/mtp_rt_golden.txt>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "strata/artifact/q8_0_quant.hpp"
#include "strata/artifact/sha256.hpp"

namespace {

int failures = 0;

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

} // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: q8_0_quant_test <tests/mtp/mtp_rt_golden.txt>\n");
        return 2;
    }
    std::ifstream in(argv[1]);
    if (!in) {
        std::fprintf(stderr, "cannot read %s\n", argv[1]);
        return 2;
    }
    long long want_values = 0, want_blocks = 0, want_bytes = 0;
    std::string want_sha, want_first;
    std::string line;
    while (std::getline(in, line)) {
        if (line.rfind("q8_0 ", 0) == 0) {
            char key[32] = {0};
            long long a = 0, b = 0, c = 0;
            if (std::sscanf(line.c_str(), "%31s values %lld blocks %lld bytes %lld", key, &a, &b, &c) == 4) {
                want_values = a;
                want_blocks = b;
                want_bytes = c;
            }
        } else if (line.rfind("q8_0_sha256 ", 0) == 0) {
            want_sha = line.substr(12);
        } else if (line.rfind("q8_0_first ", 0) == 0) {
            want_first = line.substr(11);
        }
    }
    if (want_sha.empty() || want_values == 0) {
        std::fprintf(stderr, "the golden carries no q8_0 section\n");
        return 1;
    }

    // the same input as the reference: (k * 37 % 101 - 50) / 16, exact in float32
    std::vector<float> x((std::size_t) want_values);
    for (std::size_t i = 0; i < x.size(); ++i)
        x[i] = (float) ((long long) ((i * 37) % 101) - 50) / 16.0f;

    std::vector<uint8_t> out(x.size() / strata::kQ8_0BlockValues * strata::kQ8_0BlockBytes);
    strata::q8_0_from_f32(x.data(), x.size(), out.data());

    if ((long long) out.size() != want_bytes) {
        std::fprintf(stderr, "FAIL the quantized size is %zu, the reference's is %lld\n", out.size(), want_bytes);
        ++failures;
    }
    strata::Sha256 h;
    h.update(out.data(), out.size());
    const std::array<uint8_t, 32> d = h.finish();
    const std::string got = hex_of(d.data(), d.size());
    if (got != want_sha) {
        std::fprintf(stderr, "FAIL the quantized bytes differ from tools/mtp_rt.py's own:\n  got  %s\n  want %s\n",
                     got.c_str(), want_sha.c_str());
        ++failures;
    }
    if (hex_of(out.data(), strata::kQ8_0BlockBytes) != want_first) {
        std::fprintf(stderr, "FAIL the first block differs:\n  got  %s\n  want %s\n",
                     hex_of(out.data(), strata::kQ8_0BlockBytes).c_str(), want_first.c_str());
        ++failures;
    }
    if ((long long) (x.size() / strata::kQ8_0BlockValues) != want_blocks) {
        std::fprintf(stderr, "FAIL block count\n");
        ++failures;
    }

    if (failures) {
        std::fprintf(stderr, "q8_0_quant_test: FAILED (%d)\n", failures);
        return 1;
    }
    std::printf("q8_0_quant_test: the quantizer matches tools/mtp_rt.py's own over %lld values (%lld blocks)\n",
                want_values, want_blocks);
    return 0;
}
