// include/strata/artifact/dequant_dispatch.hpp - a ggml type id to the dequantizer that owns it.
//
// `dequant.hpp` holds the scalar dequantizers, transcribed from ggml expression by expression, and deliberately
// knows nothing about ggml's type IDs or block geometry: it is one function per encoding.  This is the piece that
// maps an id (0 = F32, 8 = Q8_0, 42 = Q2_0, ...) to the function and walks the blocks, so a caller that has a
// tensor and its type id - the packer converting a tensor, and `verify` dequantizing the source to compare against
// the pack - does not each grow their own switch.
//
// `block_geometry` (gguf_reader.hpp) is the single source of block sizes, so a type whose geometry and whose
// dequantizer disagree is impossible here: an id with no geometry is refused, and one with geometry but no
// dequantizer is refused too.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "strata/artifact/dequant.hpp"
#include "strata/artifact/gguf_reader.hpp"

namespace strata {

/// The tensor's values as float32, through the engine's own dequantizers.  `bytes` must be whole blocks of the
/// type's geometry.  False when the type has no geometry, the byte count is not whole blocks, or this port has no
/// dequantizer for the encoding.
inline bool dequantize_any(uint32_t type, const uint8_t* p, uint64_t bytes, std::vector<float>& out) {
    int be = 0, bb = 0;
    if (!block_geometry(type, be, bb)) return false;
    if (bytes == 0 || bytes % (uint64_t)bb) return false;
    const uint64_t elements = bytes / (uint64_t)bb * (uint64_t)be;
    out.assign((std::size_t)elements, 0.0f);
    if (type == 0) { // F32
        std::memcpy(out.data(), p, (std::size_t)bytes);
        return true;
    }
    if (type == 1) { // F16
        dequantize_f16(p, out.data(), (int)elements);
        return true;
    }
    if (type == 30) { // BF16
        dequantize_bf16(p, out.data(), (int)elements);
        return true;
    }
    void (*row)(const uint8_t*, float*) = nullptr;
    switch (type) {
    case 2:
        row = dequantize_q4_0;
        break;
    case 6:
        row = dequantize_q5_0;
        break;
    case 7:
        row = dequantize_q5_1;
        break;
    case 8:
        row = dequantize_q8_0;
        break;
    case 11:
        row = dequantize_q3_K;
        break;
    case 12:
        row = dequantize_q4_K;
        break;
    case 13:
        row = dequantize_q5_K;
        break;
    case 14:
        row = dequantize_q6_K;
        break;
    case 20:
        row = dequantize_iq4_nl;
        break;
    case 23:
        row = dequantize_iq4_xs;
        break;
    case 42:
        row = dequantize_q2_0;
        break;
    default:
        return false;
    }
    for (uint64_t b = 0; b < bytes / (uint64_t)bb; ++b)
        row(p + b * (uint64_t)bb, out.data() + b * (uint64_t)be);
    return true;
}

} // namespace strata
