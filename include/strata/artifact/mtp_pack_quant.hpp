// include/strata/artifact/mtp_pack_quant.hpp - the host-side quantizers tools/mtp_pack.py uses.
//
// tools/mtp_pack.py packs the MTP draft head's routed experts into the main model's Q2_0, with ggml's Q4_0 and
// Q8_0 as the acceptance comparison arms.  These are transcriptions of ITS three quantizers - not of ggml's, and
// not of anything else in this directory:
//
//   * `q2_0_from_f32` searches 17 candidate block scales (amax/2 .. amax) for the one that MINIMIZES squared
//     error over the ternary grid {-1, 0, +1, +2}, then quantizes against the STORED fp16 scale rather than the
//     f32 one it chose.  The main model's own Q2_0 tools are a different statement: `gguf_writer.py` and
//     `canonical_mapping.hpp` set `d = amax` and never use the +2 level, and neither of them is a f32 -> Q2_0
//     host quantizer at all.
//   * `q4_0_from_f32` is ggml's reference rounding (d = signed max / -8).
//   * `q8_0_rint_from_f32` is NOT `q8_0_quant.hpp`'s `q8_0_from_f32`.  That one rounds half AWAY from zero
//     (ggml's `roundf`, transcribed for tools/mtp_rt.py); this reference rounds with numpy's `rint`, which is
//     half to EVEN.  The two agree everywhere except on an exact tie, and an exact tie is a different byte -
//     which is why this is a second function rather than a call into the first.
//
// The squared-error sums and the report's means go through numpy's PAIRWISE summation, transcribed below: a
// plain left-to-right float32 sum is a different number, and a different number picks a different scale on a
// near-tie.  It is 8 accumulators for n <= 128 and recursive halving above that (numpy's PW_BLOCKSIZE).
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

#include "strata/kernels/f16_bits.hpp"

namespace strata {

inline constexpr std::size_t kQ2_0BlockValues = 64;
inline constexpr std::size_t kQ2_0BlockBytes = 18;   ///< a 2-byte fp16 scale, then 16 packed 2-bit codes
inline constexpr std::size_t kQ4_0BlockValues = 32;
inline constexpr std::size_t kQ4_0BlockBytes = 18;   ///< a 2-byte fp16 scale, then 16 packed nibble bytes

/// numpy's `a.sum()` for a contiguous row.  Transcribed from `pairwise_sum_@TYPE@`: below 8 elements a plain
/// loop, up to PW_BLOCKSIZE (128) eight accumulators combined in a fixed tree, and above that the array is
/// halved (rounded down to a multiple of 8) and the halves summed recursively.
template <class T>
inline T np_pairwise_sum(const T* a, std::size_t n) {
    if (n < 8) {
        T res = T(0);
        for (std::size_t i = 0; i < n; ++i) res += a[i];
        return res;
    }
    if (n <= 128) {
        T r[8];
        for (int k = 0; k < 8; ++k) r[k] = a[k];
        std::size_t i = 8;
        const std::size_t lim = n - (n % 8);
        for (; i < lim; i += 8)
            for (int k = 0; k < 8; ++k) r[k] += a[i + (std::size_t) k];
        T res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for (; i < n; ++i) res += a[i];
        return res;
    }
    std::size_t n2 = n / 2;
    n2 -= n2 % 8;
    return np_pairwise_sum<T>(a, n2) + np_pairwise_sum<T>(a + n2, n - n2);
}

/// numpy's `a.mean()`: the pairwise sum divided by the count, in the array's own dtype.
inline float np_mean_f32(const float* a, std::size_t n) { return np_pairwise_sum<float>(a, n) / (float) n; }
inline double np_mean_f64(const double* a, std::size_t n) { return np_pairwise_sum<double>(a, n) / (double) n; }

/// numpy's `rint`: round to nearest, ties to EVEN.  `nearbyint` under the default rounding mode is exactly that.
inline float np_rint(float v) { return std::nearbyint(v); }

inline void write_f16(uint8_t* out, float v) {
    const std::uint16_t h = kernels::f16_from_f32(v);
    out[0] = (uint8_t) (h & 0xFFu);
    out[1] = (uint8_t) (h >> 8);
}

/// `n` must be a whole number of 64-value blocks; `out` gets `n / 64 * 18` bytes.
inline void q2_0_from_f32(const float* x, std::size_t n, uint8_t* out) {
    for (std::size_t b = 0; b < n / kQ2_0BlockValues; ++b) {
        const float* v = x + b * kQ2_0BlockValues;
        float amax = 0.0f;
        for (std::size_t i = 0; i < kQ2_0BlockValues; ++i) amax = std::max(amax, std::fabs(v[i]));

        float best_err = std::numeric_limits<float>::infinity();
        float best_d = 0.0f;
        float q[kQ2_0BlockValues], sq[kQ2_0BlockValues];
        // np.linspace(0.5, 1.0, 17, dtype=float32): 17 exactly-representable steps of 1/32.
        for (int s = 0; s < 17; ++s) {
            const float d = amax * (0.5f + (float) s * 0.03125f);
            const float inv = d > 0.0f ? 1.0f / d : 0.0f;
            for (std::size_t i = 0; i < kQ2_0BlockValues; ++i) {
                q[i] = std::min(2.0f, std::max(-1.0f, np_rint(v[i] * inv)));
                const float e = q[i] * d - v[i];
                sq[i] = e * e;
            }
            const float err = np_pairwise_sum<float>(sq, kQ2_0BlockValues);
            if (err < best_err) {
                best_err = err;
                best_d = d;
            }
        }
        // quantize against the scale that will be STORED, not the f32 one that was chosen
        const float d = kernels::f32_from_f16(kernels::f16_from_f32(best_d));
        const float inv = d > 0.0f ? 1.0f / d : 0.0f;
        uint8_t codes[kQ2_0BlockValues];
        for (std::size_t i = 0; i < kQ2_0BlockValues; ++i)
            codes[i] = (uint8_t) (std::min(2.0f, std::max(-1.0f, np_rint(v[i] * inv))) + 1.0f);

        uint8_t* o = out + b * kQ2_0BlockBytes;
        write_f16(o, best_d);
        for (std::size_t j = 0; j < 16; ++j)
            o[2 + j] = (uint8_t) (codes[j * 4] | (codes[j * 4 + 1] << 2) | (codes[j * 4 + 2] << 4) |
                                  (codes[j * 4 + 3] << 6));
    }
}

/// `n` must be a whole number of 32-value blocks; `out` gets `n / 32 * 18` bytes.
inline void q4_0_from_f32(const float* x, std::size_t n, uint8_t* out) {
    for (std::size_t b = 0; b < n / kQ4_0BlockValues; ++b) {
        const float* v = x + b * kQ4_0BlockValues;
        // the SIGNED value at the first maximum of |v|, which is what numpy's argmax over the abs row gives
        std::size_t idx = 0;
        float best = std::fabs(v[0]);
        for (std::size_t i = 1; i < kQ4_0BlockValues; ++i)
            if (std::fabs(v[i]) > best) {
                best = std::fabs(v[i]);
                idx = i;
            }
        const float d = v[idx] / -8.0f;
        const float inv = d != 0.0f ? 1.0f / d : 0.0f;
        uint8_t q[kQ4_0BlockValues];
        for (std::size_t i = 0; i < kQ4_0BlockValues; ++i)
            q[i] = (uint8_t) (int) std::min(15.0f, std::trunc(v[i] * inv + 8.5f));

        uint8_t* o = out + b * kQ4_0BlockBytes;
        write_f16(o, d);
        for (std::size_t j = 0; j < 16; ++j) o[2 + j] = (uint8_t) (q[j] | (q[j + 16] << 4));
    }
}

/// `n` must be a whole number of 32-value blocks; `out` gets `n / 32 * 34` bytes.
inline void q8_0_rint_from_f32(const float* x, std::size_t n, uint8_t* out) {
    for (std::size_t b = 0; b < n / 32; ++b) {
        const float* v = x + b * 32;
        float amax = 0.0f;
        for (std::size_t i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(v[i]));
        const float d = amax / 127.0f;
        const float inv = d > 0.0f ? 1.0f / d : 0.0f;
        uint8_t* o = out + b * 34;
        write_f16(o, d);
        for (std::size_t i = 0; i < 32; ++i) o[2 + i] = (uint8_t) (int8_t) np_rint(v[i] * inv);
    }
}

} // namespace strata
