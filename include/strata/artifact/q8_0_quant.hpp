// include/strata/artifact/q8_0_quant.hpp - the host-side f32 -> ggml Q8_0 quantizer.
//
// The MTP runtime files (tools/mtp_rt.py) write the draft layer's large projections as Q8_0 so the engine's
// multi-column MMVQ can run them, which means quantizing HOST weights - a different direction and a different
// caller from everything else in this directory:
//
//   * `canonical_mapping.hpp`'s `encode_q8_0` goes ggml bytes -> the packer's canonical planes.  That is a
//     REARRANGEMENT of bytes that are already quantized; it does not decide a scale.
//   * the engine's `kernels::quantize_q8_0` is a DEVICE activation quantizer (it takes device pointers and runs
//     on the MMVQ path), for activations rather than weights.
//
// So this is its own thing, and it is a transcription of the reference's arithmetic rather than an improvement on
// it: `d = amax / 127`, `q = sign(v) * floor(|v| + 0.5)`.  Note the rounding is half-AWAY-FROM-ZERO, not the
// round-half-even of the fp16 scale below it.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "strata/kernels/f16_bits.hpp"

namespace strata {

inline constexpr std::size_t kQ8_0BlockValues = 32;
inline constexpr std::size_t kQ8_0BlockBytes = 34;   ///< a 2-byte fp16 scale, then 32 int8 codes

/// `n` must be a whole number of 32-value blocks; `out` gets `n / 32 * 34` bytes.
inline void q8_0_from_f32(const float* x, std::size_t n, uint8_t* out) {
    for (std::size_t b = 0; b < n / kQ8_0BlockValues; ++b) {
        const float* v = x + b * kQ8_0BlockValues;
        float amax = 0.0f;
        for (std::size_t i = 0; i < kQ8_0BlockValues; ++i) amax = std::max(amax, std::fabs(v[i]));
        const float d = amax / 127.0f;
        const float inv = d > 0.0f ? 1.0f / d : 0.0f;   // an all-zero block stays zero rather than dividing by it
        uint8_t* o = out + b * kQ8_0BlockBytes;
        const std::uint16_t half = kernels::f16_from_f32(d);
        std::memcpy(o, &half, 2);
        for (std::size_t i = 0; i < kQ8_0BlockValues; ++i) {
            const float s = v[i] * inv;
            const float q = (s < 0.0f ? -1.0f : 1.0f) * std::floor(std::fabs(s) + 0.5f);
            // clamped, not wrapped: numpy's astype(int8) wraps on overflow, which is not a behaviour to reproduce.
            // With d = amax/127 the largest |q| is 127, so the clamp is unreachable for finite input.
            const float c = std::max(-128.0f, std::min(127.0f, q));
            o[2 + i] = (uint8_t) (int8_t) c;
        }
    }
}

} // namespace strata
