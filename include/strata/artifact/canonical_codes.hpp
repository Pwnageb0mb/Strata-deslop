// include/strata/artifact/canonical_codes.hpp - the canonical pack's code packing and scale precision.
//
// tools/strata_pack.py is the canonical Q2_0 pack builder, and the tree names the C++ `strata-pack` as its port
// ("if the two disagree the C++ is wrong").  These are the two pure functions at its base, which every S2/S4/S8
// tensor goes through:
//
//   * pack_codes - one code per element into bytes, LSB-first, `8/bits` per byte.  This is Q2_0's own convention
//     (4 codes per byte, byte j/4, bits (j%4)*2) kept for every form, so the kernel has one unpacking loop rather
//     than one per source type.  Getting the bit order wrong produces plausible weights, which is why
//     tests/artifact/canonical_codes_test.cpp compares it against the reference's own answers.
//
//   * choose_scale_dtype - a canonical scale is an FP32 product, fl(d*sc), which is generally NOT representable
//     in FP16, so the packer stores FP16 only when the round-trip is exact and FP32 otherwise.  It never has to
//     decide correctly: `strata_pack verify` re-checks bit-exactness, so a wrong choice is a failed build rather
//     than a silently lossy pack.
//
// Header-only, like dequant.hpp beside it: pure functions with no state, and the rounding is the engine's own
// (strata::kernels), not a second copy of it.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "strata/kernels/f16_bits.hpp"

namespace strata {

/// Codes per byte for a code width: 2 -> 4, 4 -> 2, 8 -> 1.
inline int codes_per_byte(int bits) { return 8 / bits; }

/// Pack one code per element into bytes, LSB-first.  False when `bits` is not a code width the pack uses, or when
/// the count is not a whole number of bytes - the reference raises on that too.
inline bool pack_codes(const uint8_t* codes, std::size_t n, int bits, std::vector<uint8_t>& out) {
    out.clear();
    if (bits != 2 && bits != 4 && bits != 8) return false;
    if (bits == 8) {
        out.assign(codes, codes + n);
        return true;
    }
    const std::size_t per = (std::size_t)codes_per_byte(bits);
    if (n % per) return false;
    const uint8_t mask = (uint8_t)((1u << bits) - 1);
    out.resize(n / per);
    for (std::size_t b = 0; b < out.size(); ++b) {
        uint8_t v = 0;
        for (std::size_t k = 0; k < per; ++k) v = (uint8_t)(v | (uint8_t)((codes[b * per + k] & mask) << (k * bits)));
        out[b] = v;
    }
    return true;
}

/// The inverse, for the tests and for `verify`: `n` codes out of `buf_bytes` bytes.
inline void unpack_codes(const uint8_t* buf, std::size_t buf_bytes, int bits, std::size_t n,
                         std::vector<uint8_t>& out) {
    out.clear();
    if (bits == 8) {
        const std::size_t take = n < buf_bytes ? n : buf_bytes;
        out.assign(buf, buf + take);
        return;
    }
    const std::size_t per = (std::size_t)codes_per_byte(bits);
    const uint8_t mask = (uint8_t)((1u << bits) - 1);
    out.reserve(n);
    for (std::size_t b = 0; b < buf_bytes && out.size() < n; ++b)
        for (std::size_t k = 0; k < per && out.size() < n; ++k)
            out.push_back((uint8_t)((buf[b] >> (k * bits)) & mask));
}

/// FP16 when every scale's round-trip is exact, else FP32; returns true when `out` holds FP16.  NaN compares
/// unequal, so a NaN scale takes the FP32 path, which is what numpy's array_equal does in the reference.
inline bool choose_scale_dtype(const float* scales, std::size_t n, std::vector<uint8_t>& out) {
    bool exact = true;
    for (std::size_t i = 0; i < n; ++i) {
        if (kernels::f32_from_f16(kernels::f16_from_f32(scales[i])) != scales[i]) {
            exact = false;
            break;
        }
    }
    out.resize(n * (exact ? 2u : 4u));
    for (std::size_t i = 0; i < n; ++i) {
        if (exact) {
            const uint16_t h = kernels::f16_from_f32(scales[i]);
            out[2 * i] = (uint8_t)(h & 0xFF);
            out[2 * i + 1] = (uint8_t)(h >> 8);
        } else {
            uint32_t u = 0;
            std::memcpy(&u, &scales[i], 4);
            for (int b = 0; b < 4; ++b) out[4 * i + (std::size_t)b] = (uint8_t)((u >> (8 * b)) & 0xFF);
        }
    }
    return exact;
}

} // namespace strata
