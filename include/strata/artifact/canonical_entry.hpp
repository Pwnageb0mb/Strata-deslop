// include/strata/artifact/canonical_entry.hpp - a tensor's planes in the pack, and the manifest entry for them.
//
// `tools/strata_pack.py`'s `tensor_entry` turns one tensor's source bytes into the bytes the pack holds plus the
// manifest entry that describes them; `decode_entry` reads a tensor back using ONLY what that entry says.  The
// pair is the format, and the second half is what makes it one: if the entry does not carry enough to decode the
// tensor, `decode_entry` cannot fall back on knowing the source type, so an omission becomes a failure there
// rather than a silent wrong number.
//
// The layout rule is `put()`: every plane starts on a 64-byte boundary, and a plane's `offset` is ABSOLUTE in the
// segment the tensor is written into (base = seg_offset), not relative to the tensor's own bytes.  That is the
// part a port is most likely to get wrong while still decoding its own output.
//
// P16/P32 tensors are a single value plane with no codes, scale or offset - and P16 only where the round-trip is
// exact, because the alternative is a silently lossy tensor; the reference raises "P16 would lose precision, so
// this tensor needs P32" instead, and so does this.
//
// Header-only, like the two headers beside it.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "strata/artifact/canonical_codes.hpp"
#include "strata/artifact/canonical_mapping.hpp"
#include "strata/artifact/dequant_dispatch.hpp"
#include "strata/artifact/gguf_reader.hpp"   // TensorInfo: the shape and source offset the entry records
#include "strata/kernels/f16_bits.hpp"

namespace strata {

constexpr uint64_t kPackAlign = 64;   // every plane starts on this boundary (P1.S7)

/// One plane of one tensor: where it starts in the segment, and how long it is.
struct PackPlane {
    uint64_t offset = 0;
    uint64_t bytes = 0;
    bool present = false;
};

/// The manifest entry for one tensor - everything `pack_decode_entry` is allowed to use.
struct PackTensorEntry {
    std::string name;
    std::string source_type;
    std::string form;
    std::string codebook;        ///< empty when the codes are affine
    int code_bias = 0;
    bool has_offset = false;
    uint32_t block_elems = 0;
    uint32_t block_bytes = 0;
    uint32_t group_elems = 0;
    std::vector<uint64_t> shape;
    uint64_t source_offset = 0;
    uint64_t elements = 0;
    int code_bits = -1;          ///< -1 when the form has no codes
    PackPlane codes, scales, offsets, values;
    int scales_fp16 = -1;        ///< -1 absent, 0 f32, 1 fp16 (the same for offsets and values)
    int offsets_fp16 = -1;
    int values_fp16 = -1;

    /// The entry projected to the one order-fixed string the golden records.
    std::string digest() const;
};

inline std::string PackTensorEntry::digest() const {
    auto plane = [](const PackPlane& p) {
        return p.present ? (std::to_string(p.offset) + ":" + std::to_string(p.bytes)) : std::string("-");
    };
    auto flag = [](int v) { return v < 0 ? std::string("-") : std::to_string(v); };
    return "form=" + form + ",codebook=" + (codebook.empty() ? std::string("-") : codebook) +
           ",code_bias=" + std::to_string(code_bias) + ",has_offset=" + (has_offset ? "1" : "0") +
           ",block_elems=" + std::to_string(block_elems) + ",block_bytes=" + std::to_string(block_bytes) +
           ",group_elems=" + std::to_string(group_elems) + ",elements=" + std::to_string(elements) +
           ",code_bits=" + (code_bits < 0 ? std::string("-") : std::to_string(code_bits)) +
           ",scales_fp16=" + flag(scales_fp16) + ",offsets_fp16=" + flag(offsets_fp16) +
           ",values_fp16=" + flag(values_fp16) + ",codes=" + plane(codes) + ",scales=" + plane(scales) +
           ",offsets=" + plane(offsets) + ",values=" + plane(values);
}

/// Encode one tensor of `source_type` into `blob` (the tensor's own bytes, starting at segment offset `seg_offset`)
/// and fill `entry`.  Returns "" on success, or the refusal message - the reference's, including the P16
/// precision one.
inline std::string pack_tensor_entry(const std::string& name, const TensorInfo& meta,
                                     const std::string& source_type, const uint8_t* raw, std::size_t raw_bytes,
                                     uint64_t seg_offset, std::vector<uint8_t>& blob, PackTensorEntry& entry) {
    const CanonicalMapping* m = canonical_mapping(source_type);
    if (!m) return "no mapping for " + source_type + " - refusing to build";
    if (raw_bytes == 0 || raw_bytes % m->block_bytes)
        return source_type + ": " + std::to_string(raw_bytes) + " bytes is not whole blocks of " +
               std::to_string(m->block_bytes);

    entry = PackTensorEntry{};
    entry.name = name;
    entry.source_type = source_type;
    entry.form = m->form;
    entry.codebook = m->codebook ? m->codebook : "";
    entry.code_bias = m->code_bias;
    entry.has_offset = m->has_offset;
    entry.block_elems = m->block_elems;
    entry.block_bytes = m->block_bytes;
    entry.group_elems = m->group_elems;
    entry.shape = meta.shape;
    entry.source_offset = meta.offset;

    blob.clear();
    auto put = [&](const std::vector<uint8_t>& b, PackPlane& plane) {
        const std::size_t pad = (std::size_t)((kPackAlign - blob.size() % kPackAlign) % kPackAlign);
        blob.insert(blob.end(), pad, 0);
        plane.offset = seg_offset + blob.size();
        plane.bytes = b.size();
        plane.present = true;
        blob.insert(blob.end(), b.begin(), b.end());
    };

    CanonicalParts parts;
    std::string err;
    if (!canonical_encode(source_type, raw, raw_bytes, parts, err)) return err;

    // ---- P16/P32: one value plane, no codes, no scale, no offset
    if (entry.form == "P16" || entry.form == "P32") {
        entry.elements = parts.values.size();
        if (entry.form == "P32") {
            std::vector<uint8_t> vb(parts.values.size() * 4);
            std::memcpy(vb.data(), parts.values.data(), vb.size());
            put(vb, entry.values);
            entry.values_fp16 = 0;
            return "";
        }
        std::vector<uint8_t> vb(parts.values.size() * 2);
        for (std::size_t i = 0; i < parts.values.size(); ++i) {
            const uint16_t h = kernels::f16_from_f32(parts.values[i]);
            if (kernels::f32_from_f16(h) != parts.values[i])
                return name + ": P16 would lose precision, so this tensor needs P32";
            vb[2 * i] = (uint8_t)(h & 0xFF);
            vb[2 * i + 1] = (uint8_t)(h >> 8);
        }
        put(vb, entry.values);
        entry.values_fp16 = 1;
        return "";
    }

    // ---- the S forms: codes, then scales, then (where the mapping has one) offsets
    int width = 0;
    if (entry.form == "S2") width = 2;
    else if (entry.form == "S4") width = 4;
    else if (entry.form == "S8") width = 8;
    else return name + ": unknown form " + entry.form;
    if (parts.codes.size() % (std::size_t)(8 / width))
        return name + ": its code count is not a whole number of bytes";

    std::vector<uint8_t> codes_b;
    if (!pack_codes(parts.codes.data(), parts.codes.size(), width, codes_b)) return name + ": cannot pack its codes";
    std::vector<uint8_t> scales_b;
    const bool sc16 = choose_scale_dtype(parts.scales.data(), parts.scales.size(), scales_b);

    entry.elements = parts.codes.size();
    entry.code_bits = width;
    entry.scales_fp16 = sc16 ? 1 : 0;
    put(codes_b, entry.codes);
    put(scales_b, entry.scales);

    if (m->has_offset) {
        std::vector<uint8_t> off_b;
        const bool o16 = choose_scale_dtype(parts.offsets.data(), parts.offsets.size(), off_b);
        entry.offsets_fp16 = o16 ? 1 : 0;
        put(off_b, entry.offsets);
    }
    return "";
}

/// Decode a tensor from the segment using only `entry`.  False - with `err` set - when the entry does not describe
/// a decodable tensor, which is the failure mode this half of the format exists to make loud.
inline bool pack_decode_entry(const uint8_t* segment, std::size_t segment_bytes, const PackTensorEntry& entry,
                              std::vector<float>& values, std::string& err) {
    auto at = [&](const PackPlane& p, const char* what) -> const uint8_t* {
        if (!p.present || p.offset > segment_bytes || p.bytes > segment_bytes - p.offset) {
            err = std::string("the entry's ") + what + " plane is not inside the segment";
            return nullptr;
        }
        return segment + p.offset;
    };
    auto floats = [](const uint8_t* p, uint64_t bytes, bool fp16, std::vector<float>& out) {
        out.resize((std::size_t)(bytes / (fp16 ? 2 : 4)));
        for (std::size_t i = 0; i < out.size(); ++i) {
            if (fp16)
                out[i] = kernels::f32_from_f16((uint16_t)((uint16_t)p[2 * i] | ((uint16_t)p[2 * i + 1] << 8)));
            else
                std::memcpy(&out[i], p + 4 * i, 4);
        }
    };

    if (entry.form == "P16" || entry.form == "P32") {
        const uint8_t* p = at(entry.values, "values");
        if (!p) return false;
        floats(p, entry.values.bytes, entry.values_fp16 == 1, values);
        return true;
    }

    const uint8_t* cp = at(entry.codes, "codes");
    const uint8_t* sp = at(entry.scales, "scales");
    if (!cp || !sp) return false;
    if (entry.code_bits <= 0) {
        err = "the entry says it has codes but not their width";
        return false;
    }
    std::vector<uint8_t> codes;
    unpack_codes(cp, (std::size_t)entry.codes.bytes, entry.code_bits, (std::size_t)entry.elements, codes);
    if (codes.size() != entry.elements) {
        err = "the codes plane is shorter than the entry's element count";
        return false;
    }
    std::vector<float> scales;
    floats(sp, entry.scales.bytes, entry.scales_fp16 == 1, scales);

    std::vector<float> offsets;
    if (entry.offsets.present) {
        const uint8_t* op = at(entry.offsets, "offsets");
        if (!op) return false;
        floats(op, entry.offsets.bytes, entry.offsets_fp16 == 1, offsets);
    }

    const float* table = entry.codebook == "IQ4NL" ? iq4nl_table() : nullptr;
    const uint64_t g = entry.group_elems ? entry.group_elems : 1;
    if (g > 1 && scales.size() < (entry.elements + g - 1) / g) {
        err = "the scales plane has fewer entries than the entry's group count";
        return false;
    }
    values.resize((std::size_t)entry.elements);
    for (std::size_t i = 0; i < values.size(); ++i) {
        const float v = table ? table[codes[i]] : (float)((int)codes[i] + entry.code_bias);
        const std::size_t grp = g == 1 ? 0 : (std::size_t)(i / g);
        values[i] = g == 1 ? v * scales[0] : v * scales[grp];
        if (!offsets.empty()) values[i] += offsets[grp];
    }
    return true;
}

/// What one tensor's verification found.
struct PackVerifyResult {
    uint64_t elements = 0;
    bool exact = false;
    double max_abs_err = 0.0;   ///< only meaningful when !exact
    std::string detail;         ///< why it failed, when it did
};

/// The rule the whole tool is organised around: for every tensor, `decode(pack) == dequant_ggml(source)` BIT FOR
/// BIT.  `dequantize_any` is the engine's own dequantization of the source bytes, so this is the pack against the
/// model rather than the pack against itself - a pack that is merely CLOSE is a pack whose weights differ from the
/// model's, and the comparison is therefore exact and not a tolerance.
///
/// A shape difference is a failure too, and not a lesser one: it means the entry and the source disagree about how
/// many elements the tensor has, which is the mistake Q2_0's packed-code representation once made (1/4 of the
/// truth, and undetectable by its own decoder).
inline PackVerifyResult pack_verify_tensor(uint32_t source_type, const uint8_t* source, uint64_t source_bytes,
                                           const uint8_t* segment, std::size_t segment_bytes,
                                           const PackTensorEntry& entry) {
    PackVerifyResult r;
    std::vector<float> ref;
    if (!dequantize_any(source_type, source, source_bytes, ref)) {
        r.detail = "this port has no dequantizer for the source type";
        return r;
    }
    std::vector<float> got;
    std::string err;
    if (!pack_decode_entry(segment, segment_bytes, entry, got, err)) {
        r.detail = err;
        return r;
    }
    if (got.size() != ref.size()) {
        r.detail = "the pack decodes " + std::to_string(got.size()) + " elements, the source holds " +
                   std::to_string(ref.size());
        return r;
    }
    r.elements = ref.size();
    r.exact = true;
    for (std::size_t i = 0; i < ref.size(); ++i) {
        if (got[i] != ref[i]) {
            r.exact = false;
            const double d = (double)got[i] - (double)ref[i];
            if (d < 0 ? -d > r.max_abs_err : d > r.max_abs_err) r.max_abs_err = d < 0 ? -d : d;
        }
    }
    if (!r.exact) r.detail = "the pack does not decode to the source's values";
    return r;
}

} // namespace strata
