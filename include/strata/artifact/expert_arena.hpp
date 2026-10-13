// include/strata/artifact/expert_arena.hpp - the expert arena's blob layout.
//
// `tools/pack_layer.py` defines one expert blob (1,382,400 bytes for the real model):
//
//     gate_up.codes   2 x 640 rows x 640 B    rows interleaved g0,u0,g1,u1,...  (share the input x)
//     down.codes      2,560 rows x 160 B
//     gate_up.scales  2 x 640 x 40 x FP16
//     down.scales     2,560 x 10 x FP16
//
// totalling exactly 819,200 + 409,600 + 102,400 + 51,200 = 1,382,400 B.  The conversion is a PURE RELAYOUT: the
// Q2_0 codes and their fp16 block scale are copied unchanged, so dequant(pack) == dequant(ggml) bit-exactly.  The
// interleaving is the point of the layout - the GGUF stores gate, up and down as three separate tensors, so a
// fused gate/up pass would read two streams from three arrays; interleaved ROWS let one pass over the activation
// serve both.
//
// Two statements, kept apart on purpose.  `pack_expert_blob` follows `pack_layer.build` (plane by plane: the
// codes plane, then the scales plane, with the gate/up rows interleaved) and `unpack_expert_role` follows
// `strata_pack.py`'s `rebuild_role` (block by block: this block's scale, then its codes).  The reference keeps
// them separate because "if it and pack_layer were the same code the arena would be unverified by construction" -
// and what verifies the pair here is the reference's own experts.bin in the golden, not either of them.
//
// The geometry is a parameter, not a constant: the real model's numbers are the defaults, and the golden runs the
// reference over a fixture-sized model so the layout can be checked without a 34 GB arena.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace strata {

/// Where the four planes of one blob start.
struct ExpertArenaOffsets {
    uint64_t gate_up_codes = 0;
    uint64_t down_codes = 0;
    uint64_t gate_up_scales = 0;
    uint64_t down_scales = 0;
};

/// The model's expert geometry.  H*FF must divide by QK, and H and FF by QK, for the planes to be whole blocks.
struct ExpertGeometry {
    uint64_t H = 2560;            ///< hidden
    uint64_t FF = 640;            ///< the gate/up output width, and down's input width
    uint64_t NE = 512;            ///< experts
    uint64_t QK = 64;             ///< Q2_0 block: elements
    uint64_t block_bytes = 18;    ///< Q2_0 block: 2 bytes of fp16 scale + 16 of 2-bit codes

    uint64_t row_bytes_gu() const { return H * 2 / 8; }        ///< 640 B of codes for one 2,560-wide row
    uint64_t row_bytes_d() const { return FF * 2 / 8; }        ///< 160 B of codes for one 640-wide row
    uint64_t scales_gu() const { return H / QK; }              ///< 40 blocks per gate/up row
    uint64_t scales_d() const { return FF / QK; }              ///< 10 blocks per down row
    uint64_t role_bytes() const { return H * FF * block_bytes / QK; }   ///< 460,800 B per role per expert
    uint64_t blob_bytes() const { return 3 * role_bytes(); }            ///< 1,382,400

    ExpertArenaOffsets offsets() const {
        ExpertArenaOffsets o;
        o.gate_up_codes = 0;
        o.down_codes = o.gate_up_codes + 2 * FF * row_bytes_gu();
        o.gate_up_scales = o.down_codes + H * row_bytes_d();
        o.down_scales = o.gate_up_scales + 2 * FF * scales_gu() * 2;
        return o;
    }
};

/// The three roles of one expert, as the GGUF stores them: each `role_bytes()` long, block by block
/// (`block_bytes` per block, the fp16 scale first).
enum ExpertRole { kGate = 0, kUp = 1, kDown = 2 };

/// One expert's blob from its three roles' Q2_0 bytes.  A pure relayout - every byte is copied, none computed -
/// which is what makes `dequant(pack) == dequant(ggml)` hold by construction.
inline void pack_expert_blob(const uint8_t* gate, const uint8_t* up, const uint8_t* down, const ExpertGeometry& g,
                             uint8_t* blob) {
    const ExpertArenaOffsets o = g.offsets();
    const uint64_t rb_gu = g.row_bytes_gu(), rb_d = g.row_bytes_d();
    const uint64_t sg = g.scales_gu(), sd = g.scales_d();
    // gate and up: FF rows each, written into interleaved slots 2*row + {0,1}
    for (int r = 0; r < 2; ++r) {
        const uint8_t* src = r == 0 ? gate : up;
        for (uint64_t row = 0; row < g.FF; ++row) {
            const uint64_t slot = 2 * row + (uint64_t)r;
            for (uint64_t b = 0; b < sg; ++b) {
                const uint8_t* block = src + (row * sg + b) * g.block_bytes;
                std::memcpy(blob + o.gate_up_scales + (slot * sg + b) * 2, block, 2);
                std::memcpy(blob + o.gate_up_codes + slot * rb_gu + b * 16, block + 2, 16);
            }
        }
    }
    // down: H rows
    for (uint64_t row = 0; row < g.H; ++row)
        for (uint64_t b = 0; b < sd; ++b) {
            const uint8_t* block = down + (row * sd + b) * g.block_bytes;
            std::memcpy(blob + o.down_scales + (row * sd + b) * 2, block, 2);
            std::memcpy(blob + o.down_codes + row * rb_d + b * 16, block + 2, 16);
        }
}

/// One role's Q2_0 bytes back out of a blob, for the whole expert: `role_bytes()` written to `out`.  The inverse
/// of `pack_expert_blob`, written the other way round (block by block) so the two are not one statement.
inline void unpack_expert_role(const uint8_t* blob, const ExpertGeometry& g, ExpertRole role, uint8_t* out) {
    const ExpertArenaOffsets o = g.offsets();
    const uint64_t rb_gu = g.row_bytes_gu(), rb_d = g.row_bytes_d();
    const uint64_t sg = g.scales_gu(), sd = g.scales_d();
    const uint64_t n_blk = g.H * g.FF / g.QK;
    for (uint64_t blk = 0; blk < n_blk; ++blk) {
        uint64_t co = 0, so = 0;
        if (role == kDown) {
            const uint64_t row = blk / sd, b_in = blk % sd;
            co = o.down_codes + row * rb_d + b_in * 16;
            so = o.down_scales + (row * sd + b_in) * 2;
        } else {
            const uint64_t row = blk / sg, b_in = blk % sg;
            const uint64_t slot = 2 * row + (role == kGate ? 0 : 1);
            co = o.gate_up_codes + slot * rb_gu + b_in * 16;
            so = o.gate_up_scales + (slot * sg + b_in) * 2;
        }
        std::memcpy(out + blk * g.block_bytes, blob + so, 2);          // fp16 d first, as block_q2_0 has it
        std::memcpy(out + blk * g.block_bytes + 2, blob + co, 16);     // then the 16 packed code bytes
    }
}

} // namespace strata
