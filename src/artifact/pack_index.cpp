// src/artifact/pack_index.cpp - see include/strata/artifact/pack_index.hpp for what this owns and why.
#include "strata/artifact/pack_index.hpp"

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

namespace strata {
namespace {

constexpr int kVerbatim = 0, kBF16InF32 = 1, kF32Copy = 2, kF16InF32 = 3;
constexpr int kActQ8_0 = 0, kActQ8_K = 1;

int file_id(const std::string& f) {
    if (f == "dense.bin") return 0;
    if (f == "embd.bin") return 1;
    if (f == "experts.bin") return 2;
    return -1;
}

/// Which activation this weight wants.  A property of the SOURCE TYPE, and not recoverable from the S-form: Q5_0
/// and Q5_K are both 8-bit with bias -16, and IQ4_NL and IQ4_XS are both 4-bit with the IQ4NL codebook and no
/// offset, so no combination of the form's fields tells the pairs apart.
int act_kind_of(const std::string& source_type) {
    if (source_type == "Q3_K" || source_type == "Q4_K" || source_type == "Q5_K" || source_type == "Q6_K" ||
        source_type == "IQ4_XS")
        return kActQ8_K;
    return kActQ8_0;
}

} // namespace

std::string pack_index_text(const std::vector<PackIndexEntry>& entries, uint64_t align, std::string& err) {
    err.clear();
    if (align == 0) align = 64;
    auto align_up = [align](uint64_t n) { return (n + align - 1) / align * align; };

    std::string rows;
    uint64_t dst = 0;
    for (const PackIndexEntry& e : entries) {
        const PackTensorEntry& v = e.tensor;
        const int fid = file_id(e.file);
        if (fid < 0) {
            err = "unknown file '" + e.file + "' for " + e.name;
            return {};
        }
        const uint64_t ne0 = v.shape.empty() ? 0 : v.shape[0];
        const uint64_t ne1 = v.shape.size() > 1 ? v.shape[1] : 0;
        const uint64_t elems = v.elements;

        uint64_t codes_bytes = 0, scales_bytes = 0, offset_bytes = 0, widen_here = 0;
        uint64_t src_off = 0, src_bytes = 0;
        int scales_fp16 = 0;
        int act_kind = kActQ8_0;

        if (v.values.present) {
            src_off = v.values.offset;
            src_bytes = v.values.bytes;
        } else {
            const PackPlane* planes[3] = {&v.codes, &v.scales, &v.offsets};
            const char* names[3] = {"codes", "scales", "offsets"};
            uint64_t lo = UINT64_MAX, hi = 0;
            for (int i = 0; i < 3; ++i)
                if (planes[i]->present) {
                    lo = std::min(lo, planes[i]->offset);
                    hi = std::max(hi, planes[i]->offset + planes[i]->bytes);
                }
            if (lo == UINT64_MAX) {
                err = e.name + ": a quantized tensor with no planes";
                return {};
            }
            src_off = lo;
            src_bytes = hi - lo;

            // The planes must be CONTIGUOUS, in order, because the loader reads them as one span.  If a pack
            // interleaved them this refuses; the alternative is reading the scale plane as offsets and decoding a
            // plausible weight.
            uint64_t expect = src_off;
            for (int i = 0; i < 3; ++i) {
                if (!planes[i]->present) continue;
                if (planes[i]->offset != expect) {
                    err = e.name + ": plane '" + names[i] + "' starts at " + std::to_string(planes[i]->offset) +
                          " but " + std::to_string(expect) + " was expected - planes are not contiguous";
                    return {};
                }
                expect += planes[i]->bytes;
            }
            if (expect != src_off + src_bytes) {
                err = e.name + ": plane span " + std::to_string(expect) + " does not end at " +
                      std::to_string(src_off + src_bytes);
                return {};
            }
            if (v.has_offset && !v.offsets.present) {
                err = e.name + " says has_offset but has no `offsets` plane";
                return {};
            }

            act_kind = act_kind_of(e.source_type);
            const int code_bits = v.code_bits > 0 ? v.code_bits : 0;
            const uint64_t group_elems = v.group_elems;
            if (code_bits == 0 || group_elems == 0) {
                err = e.name + ": a quantized tensor with code_bits=" + std::to_string(code_bits) +
                      " group_elems=" + std::to_string(group_elems);
                return {};
            }
            if (ne0 % group_elems) {
                err = e.name + ": ne0 " + std::to_string(ne0) + " is not a multiple of group_elems " +
                      std::to_string(group_elems);
                return {};
            }
            const uint64_t n_groups = ne0 / group_elems;
            codes_bytes = elems * (uint64_t)code_bits / 8;
            scales_fp16 = v.scales_fp16 == 1 ? 1 : 0;
            const int offs_fp16 = v.offsets_fp16 == 1 ? 1 : 0;
            // Engine-form plane sizes: ALWAYS 4 bytes per scale.  fp16 in the pack is widened, and the load is
            // exact, so it costs bytes and no precision.
            scales_bytes = ne1 * n_groups * 4;
            offset_bytes = v.has_offset ? ne1 * n_groups * 4 : 0;
            const uint64_t src_planes = codes_bytes + ne1 * n_groups * (scales_fp16 ? 2 : 4) +
                                        (v.has_offset ? ne1 * n_groups * (offs_fp16 ? 2 : 4) : 0);
            const uint64_t manifest_planes =
                v.codes.bytes + v.scales.bytes + (v.offsets.present ? v.offsets.bytes : 0);
            if (src_planes != manifest_planes) {
                err = e.name + ": derived planes " + std::to_string(src_planes) +
                      " disagree with the manifest's own plane bytes " + std::to_string(manifest_planes);
                return {};
            }
            if (src_planes != src_bytes) {
                err = e.name + ": derived planes " + std::to_string(src_planes) + " do not fill the span " +
                      std::to_string(src_bytes);
                return {};
            }
            if (scales_fp16) widen_here = scales_bytes - ne1 * n_groups * 2;
        }

        // ---- the engine form, as an explicit KIND rather than a flag to interpret
        const bool vf = v.values_fp16 == 1;
        int kind = kVerbatim;
        uint64_t dst_bytes = 0;
        if (e.source_type == "BF16") {
            kind = kBF16InF32;
            dst_bytes = elems * 2;
        } else if (e.source_type == "F16" && !vf) {
            kind = kF16InF32;
            dst_bytes = elems * 2;
        } else if (e.source_type == "F16") {
            kind = kVerbatim; // already 2 bytes on disk
            dst_bytes = src_bytes;
        } else if (e.source_type == "F32") {
            kind = kF32Copy;
            dst_bytes = elems * 4;
        } else {
            kind = kVerbatim;
            dst_bytes = codes_bytes + scales_bytes + offset_bytes;
            // The engine form of a quantized tensor differs from the pack by exactly the widening, and saying so
            // here is what makes the two plane sizes a CHECK rather than two independent guesses.
            //
            // NOTE: only the SCALE plane's widening is counted, which is the reference's arithmetic - the offset
            // plane's is not.  That gap is unreachable while `offsets_fp16` is false for every tensor with an
            // offset plane (all 54 of them, per pack_index.py's own note), and it is reproduced rather than
            // "fixed" so that this refuses exactly what the reference refuses.
            if (dst_bytes != src_bytes + widen_here) {
                err = e.name + ": engine form " + std::to_string(dst_bytes) + " != pack " +
                      std::to_string(src_bytes) + " + widening " + std::to_string(widen_here);
                return {};
            }
        }

        rows += e.name + " " + std::to_string(fid) + " " + std::to_string(kind) + " " + std::to_string(src_off) +
                " " + std::to_string(src_bytes) + " " + std::to_string(dst) + " " + std::to_string(dst_bytes) + " " +
                std::to_string(ne0) + " " + std::to_string(ne1) + " " +
                std::to_string(v.code_bits > 0 ? v.code_bits : 0) + " " + std::to_string(v.code_bias) + " " +
                std::to_string(v.group_elems) + " " + (v.codebook == "IQ4NL" ? "1" : "0") + " " +
                (v.has_offset ? "1" : "0") + " " + std::to_string(codes_bytes) + " " +
                std::to_string(scales_bytes) + " " + std::to_string(offset_bytes) + " " +
                std::to_string(scales_fp16) + " " + std::to_string(act_kind) + "\n";
        dst = align_up(dst + dst_bytes);
    }

    std::string out;
    out += "# strata pack index v3  --  generated by tools/strata_pack.cpp; do not hand-edit\n";
    out += "# align " + std::to_string(align) + " pool " + std::to_string(dst) + " tensors " +
           std::to_string(entries.size()) + "\n";
    out += "# name file kind src_off src_bytes dst_off dst_bytes ne0 ne1 code_bits code_bias group_elems codebook "
           "has_offset codes_bytes scales_bytes offset_bytes scales_fp16 act_kind\n";
    out += rows;
    return out;
}

} // namespace strata
