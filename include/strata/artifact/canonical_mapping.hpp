// include/strata/artifact/canonical_mapping.hpp - a source type's mapping into the canonical pack.
//
// tools/canonical_xcheck.py's MAPPINGS is the specification of record for this: each source type's encode into the
// canonical planes (one UNSIGNED code per element, one scale per block, and for some forms a separate offset per
// group) and the decode transcribed expression-by-expression from `ggml-quants.c`.  The decode is not a
// convenience: `strata_pack verify` requires decode(pack) == dequant_ggml(source) bit for bit, so it is half the
// contract.
//
// The two traps these mappings exist to pin, both of which still produce plausible weights when got wrong:
//
//   * Q8_0's code plane is UNSIGNED.  The source stores `int8_t qs[32]`; the canonical code is that value + 128,
//     so the plane is 0..255 like every other form and the decode subtracts the bias.  Reading it as the natural
//     signed value makes the two Q8_0 tensors decode 128 codes from where they belong - which is what happened
//     once, and what the bias exists to prevent.
//
//   * Q4_0's LOW nibbles fill the FIRST HALF of a block and its HIGH nibbles the SECOND.  They are not
//     interleaved, and the -8 is applied to the integer BEFORE the multiply, so the codebook is -8..7, not 0..15.
//
// ALL THIRTEEN of the reference's source types are here: Q2_0, Q3_K, Q4_0, Q4_K, Q5_0, Q5_K, Q6_K, Q8_0, IQ4_NL,
// IQ4_XS and the three float types.  A type with no mapping is still REFUSED rather than approximated - the
// reference's own behaviour ("NO MAPPING for %s - refusing to build") - which is now what happens to a ggml type
// neither implementation maps (Q4_1, MXFP4, TQ2_0).
//
// Header-only, like canonical_codes.hpp beside it: pure functions over blocks, no state.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/f16_bits.hpp"

namespace strata {

/// The canonical planes of one tensor.  Which of them a mapping fills is decided by its `form`: the S forms
/// carry `codes` + `scales` (+ `offsets` where the mapping has them), the P forms a single value plane in
/// `values`.  For P16 the stored plane is fp16 and `values` holds the widened floats - fp16 -> fp32 is exact, so
/// the plane is recoverable by narrowing, and the packing to fp16 belongs to the plane writer when it lands.
struct CanonicalParts {
    std::vector<uint8_t> codes;   ///< the S forms: blocks * block_elems, one UNSIGNED code per element
    std::vector<float> values;    ///< the P forms: the value plane, one float per element
    std::vector<float> scales;    ///< the S forms: one per block
    std::vector<float> offsets;   ///< only for the mappings with has_offset
    std::size_t blocks = 0;
};

/// One source type's mapping: what a block holds, which form its codes take, the integer bias its decode applies,
/// whether it carries a separate offset plane, and the two functions.  (tools/canonical_xcheck.py's `Mapping`.)
struct CanonicalMapping {
    const char* source_type;
    uint32_t block_bytes;
    uint32_t block_elems;
    uint32_t group_elems;
    const char* form;        ///< "S2", "S4", "S8", "P16" or "P32"
    const char* codebook;    ///< nullptr, or "IQ4NL" for the non-linear 4-bit codebooks
    int code_bias;
    bool has_offset;
    bool (*encode)(const uint8_t* raw, std::size_t blocks, CanonicalParts& out);
    bool (*decode)(const CanonicalParts& parts, std::vector<float>& values);
};

/// `kvalues_iq4nl` (`ggml-common.h`'s "Non-linear quants"): a 16-entry table, and the name is load-bearing - this
/// codebook is NOT affine, so unlike Q2_0's -1, Q3_K's -4 or Q4_0's -8 it cannot be expressed as a bias on the
/// code.  The reference checked it against two independent files (ggml-cpu/llamafile/sgemm.cpp and
/// ggml-opencl/kernels/mul_mv_iq4_nl_f32.cl) because a codebook is exactly the kind of constant that has silent
/// variants.
inline const float* iq4nl_table() {
    static const float t[16] = {-127.f, -104.f, -83.f, -65.f, -49.f, -35.f, -22.f, -10.f,
                                1.f,   13.f,   25.f,  38.f,  53.f,  69.f,  89.f,  113.f};
    return t;
}

namespace detail {

inline float f16_at(const uint8_t* p) {
    return kernels::f32_from_f16((uint16_t)((uint16_t)p[0] | ((uint16_t)p[1] << 8)));
}

inline bool encode_q8_0(const uint8_t* raw, std::size_t blocks, CanonicalParts& out) {
    out.codes.resize(blocks * 32);
    out.scales.resize(blocks);
    for (std::size_t b = 0; b < blocks; ++b) {
        const uint8_t* p = raw + b * 34;
        out.scales[b] = f16_at(p);
        for (std::size_t j = 0; j < 32; ++j)
            out.codes[b * 32 + j] = (uint8_t)((int)(int8_t)p[2 + j] + 128);   // the int8, stored unsigned
    }
    return true;
}

inline bool decode_q8_0(const CanonicalParts& parts, std::vector<float>& values) {
    if (parts.codes.size() != parts.blocks * 32 || parts.scales.size() != parts.blocks) return false;
    values.resize(parts.blocks * 32);
    for (std::size_t b = 0; b < parts.blocks; ++b)
        for (std::size_t j = 0; j < 32; ++j)
            values[b * 32 + j] = (float)((int)parts.codes[b * 32 + j] - 128) * parts.scales[b];
    return true;
}

inline bool encode_q4_0(const uint8_t* raw, std::size_t blocks, CanonicalParts& out) {
    out.codes.resize(blocks * 32);
    out.scales.resize(blocks);
    for (std::size_t b = 0; b < blocks; ++b) {
        const uint8_t* p = raw + b * 18;
        out.scales[b] = f16_at(p);
        for (std::size_t j = 0; j < 16; ++j) {
            const uint8_t q = p[2 + j];
            out.codes[b * 32 + j] = (uint8_t)(q & 0x0F);        // low nibbles -> the FIRST half
            out.codes[b * 32 + 16 + j] = (uint8_t)(q >> 4);     // high nibbles -> the SECOND half
        }
    }
    return true;
}

inline bool decode_q4_0(const CanonicalParts& parts, std::vector<float>& values) {
    if (parts.codes.size() != parts.blocks * 32 || parts.scales.size() != parts.blocks) return false;
    values.resize(parts.blocks * 32);
    for (std::size_t b = 0; b < parts.blocks; ++b)
        for (std::size_t j = 0; j < 32; ++j)
            values[b * 32 + j] = (float)((int)parts.codes[b * 32 + j] - 8) * parts.scales[b];
    return true;
}

// ---- Q2_0 -> S2.  The canonical form of the model this pack exists for, and the one that has already been got
// wrong once: the codes must be ONE 2-bit code per element, not the 16 packed bytes the source holds.  A packed
// representation passes every check its own decoder gives it and is still wrong, because `codes.size` is what the
// pack reads as a tensor's element count - for Q2_0 that came out at 1/4 of the truth.
inline bool encode_q2_0(const uint8_t* raw, std::size_t blocks, CanonicalParts& out) {
    out.codes.resize(blocks * 64);
    out.scales.resize(blocks);
    for (std::size_t b = 0; b < blocks; ++b) {
        const uint8_t* p = raw + b * 18;
        out.scales[b] = f16_at(p);
        // Source byte j holds elements 4j..4j+3 at bits 0, 2, 4 and 6.
        for (std::size_t j = 0; j < 16; ++j) {
            const uint8_t byte = p[2 + j];
            for (std::size_t t = 0; t < 4; ++t)
                out.codes[b * 64 + 4 * j + t] = (uint8_t)((byte >> (2 * t)) & 0x03);
        }
    }
    return true;
}

inline bool decode_q2_0(const CanonicalParts& parts, std::vector<float>& values) {
    if (parts.codes.size() != parts.blocks * 64 || parts.scales.size() != parts.blocks) return false;
    values.resize(parts.blocks * 64);
    for (std::size_t b = 0; b < parts.blocks; ++b)
        for (std::size_t j = 0; j < 64; ++j)
            values[b * 64 + j] = (float)((int)parts.codes[b * 64 + j] - 1) * parts.scales[b];
    return true;
}

// ---- IQ4_NL -> S4 with the IQ4 codebook.  The same nibble split as Q4_0, but the code is an INDEX into the
// non-affine table, so the whole -8 business has no analogue here.
inline bool encode_iq4_nl(const uint8_t* raw, std::size_t blocks, CanonicalParts& out) {
    out.codes.resize(blocks * 32);
    out.scales.resize(blocks);
    for (std::size_t b = 0; b < blocks; ++b) {
        const uint8_t* p = raw + b * 18;
        out.scales[b] = f16_at(p);
        for (std::size_t j = 0; j < 16; ++j) {
            const uint8_t q = p[2 + j];
            out.codes[b * 32 + j] = (uint8_t)(q & 0x0F);
            out.codes[b * 32 + 16 + j] = (uint8_t)(q >> 4);
        }
    }
    return true;
}

inline bool decode_iq4_nl(const CanonicalParts& parts, std::vector<float>& values) {
    if (parts.codes.size() != parts.blocks * 32 || parts.scales.size() != parts.blocks) return false;
    const float* table = iq4nl_table();
    values.resize(parts.blocks * 32);
    for (std::size_t b = 0; b < parts.blocks; ++b)
        for (std::size_t j = 0; j < 32; ++j)
            values[b * 32 + j] = table[parts.codes[b * 32 + j]] * parts.scales[b];
    return true;
}

// ---- the float types (P-forms).  block_bytes == block_elems == 1, so `blocks` is the element count.
inline bool encode_p32_f32(const uint8_t* raw, std::size_t n, CanonicalParts& out) {
    out.values.resize(n);
    std::memcpy(out.values.data(), raw, n * 4);   // the identity; every other mapping is measured against its
    return true;                                  // triviality
}

inline bool decode_p32_f32(const CanonicalParts& parts, std::vector<float>& values) {
    values = parts.values;
    return true;
}

inline bool encode_p32_bf16(const uint8_t* raw, std::size_t n, CanonicalParts& out) {
    // A bf16 is the top 16 bits of an f32, so widening is a shift and is exact.  P32 and not P16: bf16 has the f32
    // exponent range and fp16 does not, so an fp16 container would send large weights to infinity.
    out.values.resize(n);
    for (std::size_t i = 0; i < n; ++i)
        out.values[i] = kernels::f32_from_bf16((uint16_t)((uint16_t)raw[2 * i] | ((uint16_t)raw[2 * i + 1] << 8)));
    return true;
}

inline bool decode_p32_bf16(const CanonicalParts& parts, std::vector<float>& values) {
    values = parts.values;
    return true;
}

inline bool encode_p16_f16(const uint8_t* raw, std::size_t n, CanonicalParts& out) {
    // The pack stores the fp16 as it is and widens only to decode; fp16 -> fp32 is always exact.
    out.values.resize(n);
    for (std::size_t i = 0; i < n; ++i)
        out.values[i] = kernels::f32_from_f16((uint16_t)((uint16_t)raw[2 * i] | ((uint16_t)raw[2 * i + 1] << 8)));
    return true;
}

inline bool decode_p16_f16(const CanonicalParts& parts, std::vector<float>& values) {
    values = parts.values;
    return true;
}

// ---- the K-quants that share get_scale_min_k4: Q4_K and Q5_K.  8 groups of 32 per 256-element block, each with
// scale = d*sc and offset = -(dmin*m) - the first mappings with an offset plane.

/// `get_scale_min_k4` (ggml-quants.c L880): the 6-bit scale and min of group `j` (0..7) out of a block's 12
/// packed scale bytes.  The low four groups keep their 6 bits in a byte and the high four borrow the top two bits
/// of the LOW groups' bytes - so reading only `scb[j]` is right for the first half of the groups and silently
/// wrong for the second.  That is why this is transcribed rather than inferred from the layout.
inline void get_scale_min_k4(const uint8_t* scb, int j, int& sc, int& m) {
    if (j < 4) {
        sc = scb[j] & 63;
        m = scb[j + 4] & 63;
        return;
    }
    sc = (scb[j + 4] & 0x0F) | ((scb[j - 4] >> 6) << 4);
    m = (scb[j + 4] >> 4) | ((scb[j] >> 6) << 4);
}

/// The scales and offsets of one K-quant block: 8 groups, `d*sc` and `-(dmin*m)`, in the same two factors ggml
/// multiplies (the negation is exact, so adding the offset is bit-identical to subtracting the min).
inline void k_scale_offsets(const uint8_t* scb, float d, float dmin, float* scales, float* offsets) {
    for (int g = 0; g < 8; ++g) {
        int sc = 0, m = 0;
        get_scale_min_k4(scb, g, sc, m);
        scales[g] = d * (float)sc;
        offsets[g] = -(dmin * (float)m);
    }
}

inline bool encode_q4_k(const uint8_t* raw, std::size_t blocks, CanonicalParts& out) {
    out.codes.resize(blocks * 256);
    out.scales.resize(blocks * 8);
    out.offsets.resize(blocks * 8);
    for (std::size_t b = 0; b < blocks; ++b) {
        const uint8_t* p = raw + b * 144;
        k_scale_offsets(p + 4, f16_at(p), f16_at(p + 2), &out.scales[b * 8], &out.offsets[b * 8]);
        // TWO groups per 64-element step, taking the low and high nibbles of the SAME 32 bytes - so the group
        // size is 32, not 64, and there are 8 groups per block.
        const uint8_t* qs = p + 16;
        for (std::size_t j = 0; j < 4; ++j)
            for (std::size_t k = 0; k < 32; ++k) {
                const uint8_t byte = qs[j * 32 + k];
                out.codes[b * 256 + j * 64 + k] = (uint8_t)(byte & 0x0F);        // group 2j:   LOW nibbles
                out.codes[b * 256 + j * 64 + 32 + k] = (uint8_t)(byte >> 4);     // group 2j+1: HIGH nibbles
            }
    }
    return true;
}

inline bool encode_q5_k(const uint8_t* raw, std::size_t blocks, CanonicalParts& out) {
    out.codes.resize(blocks * 256);
    out.scales.resize(blocks * 8);
    out.offsets.resize(blocks * 8);
    for (std::size_t b = 0; b < blocks; ++b) {
        const uint8_t* p = raw + b * 176;
        k_scale_offsets(p + 4, f16_at(p), f16_at(p + 2), &out.scales[b * 8], &out.offsets[b * 8]);
        // Q4_K's shape with a 5th bit.  `qh` does NOT advance: the same 32 bytes are re-read by all four
        // 64-element steps, and what selects a distinct bit each time is the mask walking 1,4,16,64 (and
        // 2,8,32,128 for the high nibbles).  That is the Q3_K hmask trap again, which is why the masks are
        // written out rather than the high bit being folded into the nibble once.
        const uint8_t* qh = p + 16;
        const uint8_t* qs = p + 48;
        for (std::size_t j = 0; j < 4; ++j) {
            const unsigned u1 = 1u << (2 * j), u2 = 2u << (2 * j);
            for (std::size_t k = 0; k < 32; ++k) {
                const uint8_t byte = qs[j * 32 + k];
                out.codes[b * 256 + j * 64 + k] = (uint8_t)((byte & 0x0F) + ((qh[k] & u1) ? 16 : 0));
                out.codes[b * 256 + j * 64 + 32 + k] = (uint8_t)((byte >> 4) + ((qh[k] & u2) ? 16 : 0));
            }
        }
    }
    return true;
}

/// Q4_K's and Q5_K's decode are the same expression over the same geometry: `code * scale[group] + offset[group]`,
/// with one group per 32 elements.  One function, because two copies of it could drift.
inline bool decode_k256(const CanonicalParts& parts, std::vector<float>& values) {
    if (parts.codes.size() != parts.blocks * 256 || parts.scales.size() != parts.blocks * 8 ||
        parts.offsets.size() != parts.blocks * 8)
        return false;
    values.resize(parts.blocks * 256);
    for (std::size_t b = 0; b < parts.blocks; ++b)
        for (std::size_t g = 0; g < 8; ++g)
            for (std::size_t k = 0; k < 32; ++k) {
                const std::size_t e = g * 32 + k;
                values[b * 256 + e] = (float)parts.codes[b * 256 + e] * parts.scales[b * 8 + g] +
                                      parts.offsets[b * 8 + g];
            }
    return true;
}

// ---- Q3_K: 16 groups of 16, no offset plane, codebook 0..7 of the 0..15 a 4-bit plane holds.

/// The 12 scale bytes of a Q3_K block -> 16 six-bit scales, in group order (`dequantize_row_q3_K` L1323-1328).
/// The first EIGHT groups take the low nibbles of `sb[0..7]`, the last eight the HIGH nibbles of those SAME
/// bytes, and `sb[8..11]` supply two high bits each in that same order.  A reader who assumes "12 bytes = 12
/// groups, one nibble each" gets half the groups right.
inline void unpack_q3_k_scales(const uint8_t* sb, int* out) {
    for (int j = 0; j < 4; ++j) {
        const int t = sb[8 + j], lo = sb[j], hi = sb[4 + j];
        out[j] = (lo & 0x0F) | ((t & 0x03) << 4);
        out[4 + j] = (hi & 0x0F) | (((t >> 2) & 0x03) << 4);
        out[8 + j] = ((lo >> 4) & 0x0F) | (((t >> 4) & 0x03) << 4);
        out[12 + j] = ((hi >> 4) & 0x0F) | (((t >> 6) & 0x03) << 4);
    }
}

inline bool encode_q3_k(const uint8_t* raw, std::size_t blocks, CanonicalParts& out) {
    out.codes.resize(blocks * 256);
    out.scales.resize(blocks * 16);
    for (std::size_t b = 0; b < blocks; ++b) {
        const uint8_t* p = raw + b * 110;
        const uint8_t* hmask = p;
        const uint8_t* qs = p + 32;
        const float d = f16_at(p + 108);
        int sc6[16];
        unpack_q3_k_scales(p + 96, sc6);
        for (int g = 0; g < 16; ++g) out.scales[b * 16 + g] = d * (float)(sc6[g] - 32);  // integer bias, one multiply
        for (int half = 0; half < 2; ++half)
            for (int j = 0; j < 4; ++j)
                for (int sub = 0; sub < 2; ++sub) {
                    const int g = 8 * half + 2 * j + sub;
                    const std::size_t base = (std::size_t)(half * 32 + sub * 16);
                    for (int l = 0; l < 16; ++l) {
                        const uint8_t c2 = (uint8_t)((qs[base + (std::size_t)l] >> (2 * j)) & 0x03);
                        // `m` does NOT reset between halves: it is 1 << (4*half + j), not 1 << j.  Getting that
                        // wrong flips the sign of roughly half the elements of the second half.
                        const uint8_t m = (uint8_t)((hmask[(std::size_t)(sub * 16 + l)] >> (4 * half + j)) & 0x01);
                        out.codes[b * 256 + (std::size_t)g * 16 + (std::size_t)l] = (uint8_t)(c2 + 4 * m);
                    }
                }
    }
    return true;
}

/// `code - bias` then one multiply, per group of 16 - the shape every 256-element, 16-group form shares.
inline bool decode_16_groups(const CanonicalParts& parts, std::vector<float>& values, int bias) {
    if (parts.codes.size() != parts.blocks * 256 || parts.scales.size() != parts.blocks * 16) return false;
    values.resize(parts.blocks * 256);
    for (std::size_t b = 0; b < parts.blocks; ++b)
        for (std::size_t g = 0; g < 16; ++g)
            for (std::size_t l = 0; l < 16; ++l) {
                const std::size_t e = g * 16 + l;
                values[b * 256 + e] = (float)((int)parts.codes[b * 256 + e] - bias) * parts.scales[b * 16 + g];
            }
    return true;
}

inline bool decode_q3_k(const CanonicalParts& parts, std::vector<float>& values) {
    return decode_16_groups(parts, values, 4);
}

// ---- Q5_0: one scale per 32 elements, codebook bias -16, 5-bit codes.
inline bool encode_q5_0(const uint8_t* raw, std::size_t blocks, CanonicalParts& out) {
    out.codes.resize(blocks * 32);
    out.scales.resize(blocks);
    for (std::size_t b = 0; b < blocks; ++b) {
        const uint8_t* p = raw + b * 22;
        out.scales[b] = f16_at(p);
        // The 5th bit of element j is bit j of a 32-bit field, and of element j+16 is bit j+16 - so `qh` is one
        // little-endian read, not four bytes.
        const uint32_t qh = (uint32_t)p[2] | ((uint32_t)p[3] << 8) | ((uint32_t)p[4] << 16) | ((uint32_t)p[5] << 24);
        const uint8_t* qs = p + 6;
        for (int j = 0; j < 16; ++j) {
            const uint8_t xh0 = (uint8_t)(((qh >> j) & 1u) << 4);
            const uint8_t xh1 = (uint8_t)(((qh >> (j + 16)) & 1u) << 4);
            out.codes[b * 32 + (std::size_t)j] = (uint8_t)((qs[j] & 0x0F) | xh0);          // low nibbles, first half
            out.codes[b * 32 + 16 + (std::size_t)j] = (uint8_t)((qs[j] >> 4) | xh1);       // high nibbles, second
        }
    }
    return true;
}

inline bool decode_q5_0(const CanonicalParts& parts, std::vector<float>& values) {
    if (parts.codes.size() != parts.blocks * 32 || parts.scales.size() != parts.blocks) return false;
    values.resize(parts.blocks * 32);
    for (std::size_t b = 0; b < parts.blocks; ++b)
        for (std::size_t j = 0; j < 32; ++j)
            values[b * 32 + j] = (float)((int)parts.codes[b * 32 + j] - 16) * parts.scales[b];
    return true;
}

// ---- Q6_K: 16 groups of 16, `scale = d*sc` with a SIGNED int8 sc, codebook bias -32.
inline bool encode_q6_k(const uint8_t* raw, std::size_t blocks, CanonicalParts& out) {
    out.codes.resize(blocks * 256);
    out.scales.resize(blocks * 16);
    for (std::size_t b = 0; b < blocks; ++b) {
        const uint8_t* p = raw + b * 210;
        const uint8_t* ql = p;
        const uint8_t* qh = p + 128;
        const int8_t* sc = (const int8_t*)(const void*)(p + 192);
        const float d = f16_at(p + 208);
        // `d * sc[is] * q` associates LEFT, so the group scale is fl(d * float(sc)) and the value is one further
        // multiply - not fl(d * fl(sc*q)).  And sc is int8_t, so a negative scale is a sign flip and must be kept.
        for (int g = 0; g < 16; ++g) out.scales[b * 16 + (std::size_t)g] = d * (float)sc[g];
        for (int half = 0; half < 2; ++half) {
            const uint8_t* qlb = ql + half * 64;
            const uint8_t* qhb = qh + half * 32;
            // Four quads from ONE qh byte, two bits each, landing at STRIDES of 32 rather than contiguously.
            for (int q = 0; q < 4; ++q) {
                const int sh = 2 * q;
                const bool high_nib = q >= 2;
                const bool second = (q % 2) == 1;
                for (int l = 0; l < 32; ++l) {
                    const uint8_t byte = qlb[second ? (32 + l) : l];
                    const uint8_t nib = high_nib ? (uint8_t)(byte >> 4) : (uint8_t)(byte & 0x0F);
                    const uint8_t seg = (uint8_t)(nib | (uint8_t)(((qhb[l] >> sh) & 0x03) << 4));
                    out.codes[b * 256 + (std::size_t)(half * 128 + q * 32 + l)] = seg;
                }
            }
        }
    }
    return true;
}

inline bool decode_q6_k(const CanonicalParts& parts, std::vector<float>& values) {
    return decode_16_groups(parts, values, 32);
}

// ---- IQ4_XS: S4 with the IQ4 codebook, 8 groups of 32, each with `scale = d*(ls - 32)`.
inline bool encode_iq4_xs(const uint8_t* raw, std::size_t blocks, CanonicalParts& out) {
    out.codes.resize(blocks * 256);
    out.scales.resize(blocks * 8);
    for (std::size_t b = 0; b < blocks; ++b) {
        const uint8_t* p = raw + b * 136;
        const float d = f16_at(p);
        const uint16_t sh = (uint16_t)((uint16_t)p[2] | ((uint16_t)p[3] << 8));   // scales_h, one uint16
        const uint8_t* sl = p + 4;                                               // scales_l: 4 bytes
        const uint8_t* qs = p + 8;
        // The 6-bit scale is split across TWO parallel arrays by element index: `scales_l` holds one nibble per
        // 32-element group (two groups per byte, low nibble first) and `scales_h` the top two bits at 2*ib.
        // Reading it out of scales_l alone is right for values under 16 and silently wrong above it.
        for (int ib = 0; ib < 8; ++ib) {
            const int ls = ((sl[ib / 2] >> (4 * (ib % 2))) & 0xF) | (((sh >> (2 * ib)) & 3) << 4);
            out.scales[b * 8 + (std::size_t)ib] = d * (float)(ls - 32);
        }
        for (int ib = 0; ib < 8; ++ib)
            for (int j = 0; j < 16; ++j) {
                const uint8_t byte = qs[(std::size_t)(ib * 16 + j)];
                out.codes[b * 256 + (std::size_t)(ib * 32 + j)] = (uint8_t)(byte & 0x0F);
                out.codes[b * 256 + (std::size_t)(ib * 32 + 16 + j)] = (uint8_t)(byte >> 4);
            }
    }
    return true;
}

inline bool decode_iq4_xs(const CanonicalParts& parts, std::vector<float>& values) {
    if (parts.codes.size() != parts.blocks * 256 || parts.scales.size() != parts.blocks * 8) return false;
    const float* table = iq4nl_table();
    values.resize(parts.blocks * 256);
    for (std::size_t b = 0; b < parts.blocks; ++b)
        for (std::size_t e = 0; e < 256; ++e)
            values[b * 256 + e] = table[parts.codes[b * 256 + e]] * parts.scales[b * 8 + e / 32];
    return true;
}

} // namespace detail

/// The mapping for `source_type`, or nullptr when the port has not reached it (or it is not a source type).
inline const CanonicalMapping* canonical_mapping(const std::string& source_type) {
    static const CanonicalMapping table[] = {
        // source_type, block_bytes, block_elems, group_elems, form, codebook, code_bias, has_offset, enc, dec
        {"Q2_0", 18, 64, 64, "S2", nullptr, -1, false, detail::encode_q2_0, detail::decode_q2_0},
        {"Q8_0", 34, 32, 32, "S8", nullptr, -128, false, detail::encode_q8_0, detail::decode_q8_0},
        {"Q4_0", 18, 32, 32, "S4", nullptr, -8, false, detail::encode_q4_0, detail::decode_q4_0},
        {"Q4_K", 144, 256, 32, "S4", nullptr, 0, true, detail::encode_q4_k, detail::decode_k256},
        {"Q5_K", 176, 256, 32, "S8", nullptr, 0, true, detail::encode_q5_k, detail::decode_k256},
        {"IQ4_NL", 18, 32, 32, "S4", "IQ4NL", 0, false, detail::encode_iq4_nl, detail::decode_iq4_nl},
        {"Q3_K", 110, 256, 16, "S4", nullptr, -4, false, detail::encode_q3_k, detail::decode_q3_k},
        {"Q5_0", 22, 32, 32, "S8", nullptr, -16, false, detail::encode_q5_0, detail::decode_q5_0},
        {"Q6_K", 210, 256, 16, "S8", nullptr, -32, false, detail::encode_q6_k, detail::decode_q6_k},
        {"IQ4_XS", 136, 256, 32, "S4", "IQ4NL", 0, false, detail::encode_iq4_xs, detail::decode_iq4_xs},
        {"F32", 4, 1, 1, "P32", nullptr, 0, false, detail::encode_p32_f32, detail::decode_p32_f32},
        {"BF16", 2, 1, 1, "P32", nullptr, 0, false, detail::encode_p32_bf16, detail::decode_p32_bf16},
        {"F16", 2, 1, 1, "P16", nullptr, 0, false, detail::encode_p16_f16, detail::decode_p16_f16},
    };
    for (const CanonicalMapping& m : table)
        if (source_type == m.source_type) return &m;
    return nullptr;
}

/// Encode `raw_bytes` of whole blocks into the canonical planes.  False - with `err` set - when the type has no
/// mapping yet, or the byte count is not whole blocks.  The reference refuses to build in both cases.
inline bool canonical_encode(const std::string& source_type, const uint8_t* raw, std::size_t raw_bytes,
                             CanonicalParts& out, std::string& err) {
    const CanonicalMapping* m = canonical_mapping(source_type);
    if (!m) {
        err = "no mapping for " + source_type + " - refusing to build";
        return false;
    }
    if (raw_bytes == 0 || raw_bytes % m->block_bytes) {
        err = source_type + ": " + std::to_string(raw_bytes) + " bytes is not whole blocks of " +
              std::to_string(m->block_bytes);
        return false;
    }
    out = CanonicalParts{};
    out.blocks = raw_bytes / m->block_bytes;
    return m->encode(raw, out.blocks, out);
}

/// The values the planes decode to.  False when the type has no mapping, or the planes do not match its geometry.
inline bool canonical_decode(const std::string& source_type, const CanonicalParts& parts, std::vector<float>& values,
                             std::string& err) {
    const CanonicalMapping* m = canonical_mapping(source_type);
    if (!m) {
        err = "no mapping for " + source_type;
        return false;
    }
    if (!m->decode(parts, values)) {
        err = source_type + ": the planes do not match the mapping's geometry";
        return false;
    }
    return true;
}

} // namespace strata
