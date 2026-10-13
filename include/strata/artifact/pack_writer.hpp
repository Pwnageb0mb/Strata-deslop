// include/strata/artifact/pack_writer.hpp - the pack's index.txt and dense.bin, written the way the reference
// packer writes them.
//
// A pack is two files the engine reads: index.txt, one row per tensor saying what kind it is, where in dense.bin
// it lies and where in the arena the loader will put it, and dense.bin, the float tensors in the form the kernels
// read them.  tools/iq_pack.py has been the only writer of both.  A second writer has to produce the same bytes,
// or two packs of one model would differ for no reason and every "is this pack current" check would have to know
// which tool made it.
//
// So the conversions are NOT reimplemented here.  f32 -> bf16 is strata::bf16_from_f32 and f32 -> f16 is
// strata::f16_from_f32 (include/strata/kernels/), the engine's own, and the dequantizers are
// include/strata/artifact/dequant.hpp, transcribed from ggml.  What this file owns is the format: which form each
// tensor is written in (the FORM table below), the index rows, the dense.bin layout and the arena arithmetic.
//
// Proof: tests/artifact/pack_test.cpp packs the synthetic model of tools/gen_pack_golden.py and compares both
// files against the reference's own output for it.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "strata/artifact/gguf_reader.hpp"

namespace strata {

/// The float form the engine reads a pack tensor in, keyed by the name without its `blk.N.` prefix; nullptr when
/// the tensor is not in the table and is therefore written as it is stored (or served from its GGUF).
const char* pack_form(const std::string& name);

/// Whether `name` is a mixture-of-experts tensor: written by the expert path, never into the index.
bool pack_is_expert(const std::string& name);

/// One conversion, as conversions.json records it.
struct PackConversion {
    std::string name;
    std::string shard;
    std::string src_type;
    std::string dst_type;
    std::string method;
    bool exact = false;
    double max_abs_err = 0.0;
    std::string source_sha256;      ///< of the tensor's bytes as stored in its GGUF
    bool method_last = false;       ///< see the note on record order in pack_writer.cpp
};

struct PackOutput {
    std::string index;               ///< index.txt, verbatim
    std::string conversions_json;    ///< conversions.json, verbatim
    std::vector<uint8_t> dense;      ///< dense.bin, verbatim
    std::vector<PackConversion> conversions;
    uint64_t pool_bytes = 0;         ///< the arena the index's pool offsets address
    std::size_t served_native = 0;   ///< tensors left in their GGUF encoding
};

/// Every non-expert tensor of `model`, packed into `out`: the floats the engine reads from the pack, in the form
/// it reads them, converted when stored otherwise; quantized tensors served from their GGUF.  `compat_bf16`
/// allows the conversions that round (--compat-bf16 in the reference).  Returns "" on success, or the message
/// naming what is refused - in which case `out` is not touched.
std::string pack_standalone(const GgufModel& model, PackOutput& out, bool compat_bf16);

/// One role (gate, up or down) of one layer's experts: the tensor, the shard it lives in, and its absolute offset
/// in THAT shard - two shards do not start their data section at the same byte, so one role's data_start must not
/// be used for another's (per-role data_start as in #255, gopinath87607).
struct PackExpertRole {
    const TensorInfo* tensor = nullptr;
    std::size_t shard = 0;
    uint64_t offset = 0;
    std::string shard_name;   ///< empty when the role is in the source shard
};

struct PackExpertLayer {
    int64_t layer = 0;
    uint32_t gu_type = 0;     ///< gate and up share a type; the layout refuses it otherwise
    uint32_t d_type = 0;
    uint64_t offset = 0;      ///< into experts.bin: blob * n_expert per layer, in layer order
    uint64_t blob = 0;        ///< bytes of one expert: gate | up | down
    std::vector<PackExpertRole> roles;   ///< gate, up, down
};

struct PackExpertTable {
    std::vector<PackExpertLayer> layers;
    std::string text;         ///< native_experts.txt, verbatim
    int64_t n_expert = 0;
    uint64_t total_bytes = 0;
};

/// The pack's expert table (native_experts.txt) for `model`, or the refusal message.  The source shard is shard 0:
/// a role in any other shard is named in the row's shard column.  Refused rather than written when a layer is
/// incomplete, when the routers disagree on the expert count, or when gate and up are not the same type - all
/// before any file of the pack changes.
std::string pack_expert_layout(const GgufModel& model, PackExpertTable& out);

/// The bytes of one layer's experts as experts.bin holds them: for each expert in turn, its gate, up and down
/// bytes.  `blob` must be the layer's blob (the three per-expert sizes added up).  False when a tensor's bytes do
/// not divide into `n_expert` whole parts, or the model does not hold one of the roles.
bool pack_expert_layer_bytes(const GgufModel& model, const PackExpertLayer& layer, int64_t n_expert,
                             std::vector<uint8_t>& out);

/// The sidecar beside experts.bin (experts.bin.src.json): what the file was cut from.  The shard names and sizes
/// plus the sha256 of the table - every per-role file and offset, the formats and the blob sizes - so an
/// experts.bin of the same size but another model or another packing is not mistaken for this one.
std::string pack_experts_source_json(const GgufModel& model, const std::string& native_experts_text,
                                     uint64_t total_bytes);

} // namespace strata
