// src/artifact/pack_writer.cpp - see include/strata/artifact/pack_writer.hpp for what this owns and why.
#include "strata/artifact/pack_writer.hpp"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <set>
#include <string>

#include "strata/artifact/dequant.hpp"
#include "strata/artifact/dequant_dispatch.hpp"
#include "strata/artifact/sha256.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/f16_bits.hpp"

namespace strata {
namespace {

// The engine's own conversions, which is the point: the packer does not reimplement rounding.
using strata::kernels::bf16_from_f32;
using strata::kernels::f16_from_f32;
using strata::kernels::f32_from_bf16;
using strata::kernels::f32_from_f16;

constexpr uint64_t kAlign = 64;                        // the index's arena alignment, in bytes
constexpr const char* kPleKey = "blk.1.ple_key.weight";

// The 28.8 GB PLE table: read from its GGUF by the engine, never in the pack.
const char* kNotInPack[] = {"per_layer_token_embd.weight"};

// The float form the engine reads each pack tensor in (name without `blk.N.`; from eddoursul/Strata's FORM table).
// The residual, router, GDN, QSA and PLE kernels read these small projections as BF16, the norms as F32 and the PLE
// conv as F16.  GSQ-RCO files already store them that way; other files do not (Unsloth's UD-Q4_K_XL: the routers,
// SSM gates and injections F32, the hyper-connection projections and the PLE value Q8_0, the PLE conv F32), so a
// tensor stored otherwise is converted, and every conversion is recorded:
//   F32 -> BF16 whose values are exactly BF16 (low 16 bits zero)    exact, by default
//   F32 with other values, F16 or a quantized type -> BF16          rounds (nearest-even): only with --compat-bf16
//   F32 -> F16 (ple_conv1d)                                         index kind 3, narrowed by the loader (#255),
//                                                                   by default - the kernel reads nothing else
//   F16 / BF16 -> F32                                               exact widening, by default
// Tensors not named here are written as stored (floats) or served from the GGUF (quantized), as before.
const std::map<std::string, const char*>& form_table() {
    static const std::map<std::string, const char*> t = {
        {"ffn_gate_inp.weight", "BF16"}, {"ffn_gate_inp_shexp.weight", "BF16"},
        {"hc_attn_down.weight", "BF16"}, {"hc_attn_up.weight", "BF16"}, {"hc_attn_inject.weight", "BF16"},
        {"hc_ffn_down.weight", "BF16"}, {"hc_ffn_up.weight", "BF16"}, {"hc_ffn_inject.weight", "BF16"},
        {"output_hc_down.weight", "BF16"}, {"output_hc_up.weight", "BF16"},
        {"indexer.k_proj.weight", "BF16"}, {"indexer.q_proj.weight", "BF16"},
        {"ssm_alpha.weight", "BF16"}, {"ssm_beta.weight", "BF16"},
        {"ple_key.weight", "BF16"}, {"ple_value.weight", "BF16"}, {"ple_conv1d.weight", "F16"},
        {"attn_q_norm.weight", "F32"}, {"attn_k_norm.weight", "F32"}, {"hc_attn_norm.weight", "F32"},
        {"hc_ffn_norm.weight", "F32"}, {"output_hc_norm.weight", "F32"}, {"indexer.q_norm.weight", "F32"},
        {"indexer.k_norm.weight", "F32"}, {"ple_norm_conv.weight", "F32"}, {"ple_norm_key.weight", "F32"},
        {"ple_norm_query.weight", "F32"}, {"ssm_a", "F32"}, {"ssm_conv1d.weight", "F32"}, {"ssm_dt.bias", "F32"},
        {"ssm_norm.weight", "F32"},
    };
    return t;
}

// PLE key encodings left in the GGUF for the engine's native PLE key: the formats NativeDense serves (Q2_0; Q8_0 in
// UD-Q4_K_XL; IQ3_XXS and IQ4_XS, #381).  Other quantized keys take the BF16 path (--compat-bf16), as before.
const std::set<std::string>& native_ple_key() {
    static const std::set<std::string> s = {"Q2_0", "Q8_0", "IQ3_XXS", "IQ4_XS"};
    return s;
}

/// The index kind for each form the engine reads from dense.bin.
const char* kind_of(const std::string& form) {
    if (form == "BF16") return "4";
    if (form == "F16") return "5";
    return "2"; // F32
}

bool is_float_type(const std::string& type_name) {
    return type_name == "BF16" || type_name == "F32" || type_name == "F16";
}

/// `blk.12.attn_q_norm.weight` -> `attn_q_norm.weight`; anything else is returned unchanged (the reference's
/// `re.sub(r"^blk\.\d+\.", "", name)`).
std::string strip_block(const std::string& name) {
    if (name.compare(0, 4, "blk.") != 0) return name;
    std::size_t i = 4;
    while (i < name.size() && name[i] >= '0' && name[i] <= '9') ++i;
    if (i == 4 || i >= name.size() || name[i] != '.') return name;
    return name.substr(i + 1);
}

uint32_t f32_bits(float f) {
    uint32_t u = 0;
    std::memcpy(&u, &f, 4);
    return u;
}

/// The values as BF16, ggml's round-to-nearest-even (strata::bf16_from_f32 is the engine's copy of it).
std::vector<uint8_t> bf16_bytes(const std::vector<float>& values) {
    std::vector<uint8_t> out(values.size() * 2);
    for (std::size_t i = 0; i < values.size(); ++i) {
        const uint16_t h = bf16_from_f32(values[i]);
        out[2 * i] = (uint8_t)(h & 0xFF);
        out[2 * i + 1] = (uint8_t)(h >> 8);
    }
    return out;
}

/// The values as F16, nearest-even (the engine loader's f16_from_f32); false when a value is not finite or overflows
/// F16, which the reference refuses too.
bool f16_bytes(const std::vector<float>& values, std::vector<uint8_t>& out) {
    out.assign(values.size() * 2, 0);
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (!std::isfinite(values[i])) return false;
        const uint16_t h = f16_from_f32(values[i]);
        if (!std::isfinite(f32_from_f16(h))) return false; // beyond F16's range, which the engine reads it as
        out[2 * i] = (uint8_t)(h & 0xFF);
        out[2 * i + 1] = (uint8_t)(h >> 8);
    }
    return true;
}

struct Converted {
    std::string kind;
    std::vector<uint8_t> data;
    uint64_t dst_bytes = 0;
    PackConversion rec;
    bool has_rec = false;
};

/// The pack form of one tensor the engine reads from dense.bin.  Returns "" on success (with `got` filled), or the
/// refusal message - the reference's `convert()`, with its wording.
std::string convert(const std::string& name, uint32_t type, const std::string& type_name, const uint8_t* raw,
                    uint64_t raw_bytes, bool compat_bf16, Converted& got) {
    const char* form_c = pack_form(name);
    const std::string form = form_c ? form_c : "";
    if (form.empty() || form == type_name) {
        // Written as it is stored: a float tensor the FORM table does not name, or one already in its form.
        got.kind = kind_of(form.empty() ? type_name : form);
        got.data.assign(raw, raw + raw_bytes);
        got.dst_bytes = raw_bytes;
        return "";
    }
    std::vector<float> values;
    if (!dequantize_any(type, raw, raw_bytes, values)) return name + ": no dequantizer for " + type_name;
    got.rec.src_type = type_name;
    got.rec.dst_type = form;
    got.has_rec = true;

    if (form == "BF16") {
        const bool f32 = type_name == "F32";
        bool low_bits_clear = f32;
        if (f32) {
            for (float v : values)
                if (f32_bits(v) & 0xFFFF) {
                    low_bits_clear = false;
                    break;
                }
        }
        if (f32 && low_bits_clear) {
            for (float v : values)
                if (!std::isfinite(v)) return name + " is F32 with non-finite values; refused rather than converted";
            got.data.resize(values.size() * 2);
            for (std::size_t i = 0; i < values.size(); ++i) {
                const uint16_t h = (uint16_t)(f32_bits(values[i]) >> 16);
                got.data[2 * i] = (uint8_t)(h & 0xFF);
                got.data[2 * i + 1] = (uint8_t)(h >> 8);
            }
            got.kind = "4";
            got.dst_bytes = got.data.size();
            got.rec.method = "f32->bf16: the values are BF16's (low 16 bits zero)";
            got.rec.exact = true;
            got.rec.max_abs_err = 0.0;
            return "";
        }
        if (!compat_bf16) {
            const std::string what = f32 ? "F32 with values that are not BF16" : type_name;
            return name + " is " + what + ", but the engine requires BF16; use --compat-bf16 (rounds to nearest-even)";
        }
        got.data = bf16_bytes(values);
        got.kind = "4";
        got.dst_bytes = got.data.size();
        std::string lower = type_name;
        for (char& c : lower) c = (char)std::tolower((unsigned char)c);
        got.rec.method = is_float_type(type_name)
                            ? lower + "->bf16 round-to-nearest-even (--compat-bf16)"
                            : "dequantize->bf16 round-to-nearest-even (--compat-bf16)";
        bool exact = true;
        double err = 0.0;
        for (std::size_t i = 0; i < values.size(); ++i) {
            const uint16_t h = (uint16_t)((uint16_t)got.data[2 * i] | ((uint16_t)got.data[2 * i + 1] << 8));
            const float back = f32_from_bf16(h);
            if (back != values[i]) exact = false;
            err = std::max(err, std::abs((double)back - (double)values[i]));
        }
        got.rec.exact = exact;
        got.rec.max_abs_err = values.empty() ? 0.0 : err;
        return "";
    }

    if (form == "F16") {
        if (type_name != "F32" && !compat_bf16)
            return name + " is " + type_name + ", but the engine reads F16; use --compat-bf16 (rounds to nearest-even)";
        std::vector<uint8_t> h;
        if (!f16_bytes(values, h)) return name + ": cannot convert non-finite weights to F16";
        bool exact = true;
        double err = 0.0;
        for (std::size_t i = 0; i < values.size(); ++i) {
            const uint16_t bits = (uint16_t)((uint16_t)h[2 * i] | ((uint16_t)h[2 * i + 1] << 8));
            const float back = f32_from_f16(bits);
            if (back != values[i]) exact = false;
            err = std::max(err, std::abs((double)back - (double)values[i]));
        }
        got.rec.exact = exact;
        got.rec.max_abs_err = values.empty() ? 0.0 : err;
        if (type_name == "F32") {
            // The F32 bytes go into dense.bin as index kind 3 and the loader narrows them to F16 (nearest-even), the
            // form the PLE conv kernel reads (#255, gopinath87607): 4 B/elem in, 2 B/elem out.
            got.rec.method = "f32->f16 by the loader (index kind 3, round-to-nearest-even)";
            got.rec.method_last = true; // the reference updates the record twice on this path, so `method` lands last
            got.kind = "3";
            got.data.assign(raw, raw + raw_bytes);
            got.dst_bytes = raw_bytes / 2;
            return "";
        }
        std::string lower = type_name;
        for (char& c : lower) c = (char)std::tolower((unsigned char)c);
        got.rec.method = lower + "->f16 round-to-nearest-even (--compat-bf16)";
        got.kind = "5";
        got.data = h;
        got.dst_bytes = h.size();
        return "";
    }

    // form == "F32": a widening
    if (type_name != "F16" && type_name != "BF16" && !compat_bf16)
        return name + " is " + type_name + ", but the engine reads F32; use --compat-bf16";
    std::string lower = type_name;
    for (char& c : lower) c = (char)std::tolower((unsigned char)c);
    got.rec.method = lower + "->f32 widening";
    got.rec.exact = true;
    got.rec.max_abs_err = 0.0;
    got.kind = "2";
    got.data.resize(values.size() * 4);
    std::memcpy(got.data.data(), values.data(), values.size() * 4);
    got.dst_bytes = got.data.size();
    return "";
}

std::string basename_of(const std::string& path) {
    const std::size_t s = path.find_last_of("/\\");
    return s == std::string::npos ? path : path.substr(s + 1);
}

// ---- conversions.json.  Not a general JSON writer: one shape, written the way Python's json.dumps(indent=1) lays
// it out, because that is what the reference writes and this file is compared against it byte for byte.  Nothing in
// the engine reads it - it is the record of what the packer changed and why - but a pack whose two writers disagree
// about it is a pack whose two writers have to be distinguished.

std::string json_string(const std::string& s) {
    std::string o = "\"";
    for (unsigned char c : s) {
        switch (c) {
        case '"':
            o += "\\\"";
            break;
        case '\\':
            o += "\\\\";
            break;
        case '\n':
            o += "\\n";
            break;
        case '\r':
            o += "\\r";
            break;
        case '\t':
            o += "\\t";
            break;
        default:
            if (c < 0x20) {
                char b[8];
                std::snprintf(b, sizeof b, "\\u%04x", c);
                o += b;
            } else {
                o += (char)c;
            }
        }
    }
    return o + "\"";
}

/// A float the way Python's repr writes it: shortest round-trip, and always with a fraction or an exponent, so
/// 0.0 is "0.0" and not "0".
std::string json_number(double v) {
    if (!std::isfinite(v)) return "null"; // the reference never records one
    char buf[48];
    const std::to_chars_result r = std::to_chars(buf, buf + sizeof(buf), v);
    std::string s(buf, r.ptr);
    if (s.find_first_of(".eE") == std::string::npos) s += ".0";
    return s;
}

std::string conversions_json(const GgufModel& model, bool compat_bf16,
                             const std::vector<PackConversion>& recs) {
    std::string o = "{\n";
    o += " \"schema\": 1,\n";
    o += " \"tool\": \"tools/strata_pack.cpp\",\n";
    o += std::string(" \"compat_bf16\": ") + (compat_bf16 ? "true" : "false") + ",\n";
    o += " \"source_shards\": [";
    if (model.size() == 0) {
        o += "],\n";
    } else {
        o += "\n";
        for (std::size_t i = 0; i < model.size(); ++i) {
            o += "  {\n";
            o += "   \"name\": " + json_string(basename_of(model.shard(i).path())) + ",\n";
            o += "   \"size\": " + std::to_string(model.shard(i).file_size()) + "\n";
            o += std::string("  }") + (i + 1 < model.size() ? "," : "") + "\n";
        }
        o += " ],\n";
    }
    o += " \"tensors\": [";
    if (recs.empty()) {
        o += "]\n}\n";
        return o;
    }
    o += "\n";
    for (std::size_t i = 0; i < recs.size(); ++i) {
        const PackConversion& r = recs[i];
        const std::string exact = r.exact ? "true" : "false";
        o += "  {\n";
        o += "   \"name\": " + json_string(r.name) + ",\n";
        o += "   \"shard\": " + json_string(r.shard) + ",\n";
        o += "   \"src_type\": " + json_string(r.src_type) + ",\n";
        o += "   \"dst_type\": " + json_string(r.dst_type) + ",\n";
        // The reference's record order, which is its update order and not a choice: on the path where the LOADER
        // narrows a tensor (kind 3) it sets `method` after `exact`/`max_abs_err`, so the key lands last.  Reproduced
        // so the file is byte-identical to the reference's; json.loads does not care.
        if (r.method_last) {
            o += "   \"exact\": " + exact + ",\n";
            o += "   \"max_abs_err\": " + json_number(r.max_abs_err) + ",\n";
            o += "   \"method\": " + json_string(r.method) + ",\n";
        } else {
            o += "   \"method\": " + json_string(r.method) + ",\n";
            o += "   \"exact\": " + exact + ",\n";
            o += "   \"max_abs_err\": " + json_number(r.max_abs_err) + ",\n";
        }
        o += "   \"source_sha256\": " + json_string(r.source_sha256) + "\n";
        o += std::string("  }") + (i + 1 < recs.size() ? "," : "") + "\n";
    }
    o += " ]\n}\n";
    return o;
}

} // namespace

const char* pack_form(const std::string& name) {
    const auto it = form_table().find(strip_block(name));
    return it == form_table().end() ? nullptr : it->second;
}

bool pack_is_expert(const std::string& name) {
    static const std::string suffix = "_exps.weight";
    return name.compare(0, 4, "blk.") == 0 && name.size() >= suffix.size() &&
           name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::string pack_standalone(const GgufModel& model, PackOutput& out, bool compat_bf16) {
    struct Job {
        const TensorInfo* t;
        std::size_t shard;
        bool native;
    };
    std::vector<Job> todo;
    std::vector<std::string> problems;

    // The order the reference walks: shard by shard, tensors in the order the file lists them.
    for (std::size_t s = 0; s < model.size(); ++s) {
        for (const auto& t : model.shard(s).tensors()) {
            if (pack_is_expert(t.name)) continue;
            bool skip = false;
            for (const char* n : kNotInPack)
                if (t.name == n) skip = true;
            if (skip) continue;
            if (t.shape.size() > 2)
                return "tensor " + t.name + " has " + std::to_string(t.shape.size()) +
                       " dimensions; the index holds two";
            const char* form = pack_form(t.name);
            const std::string type_name = t.type_name();
            const bool native = !is_float_type(type_name) &&
                                (!form || (t.name == kPleKey && native_ple_key().count(type_name)));
            if (!is_float_type(type_name) && !native && !compat_bf16)
                problems.push_back(t.name + " is " + type_name + ", but the engine requires " +
                                   (form ? form : "?") + "; use --compat-bf16");
            todo.push_back({&t, s, native});
        }
    }
    // refused before anything is written: the previous pack stays as it was
    if (!problems.empty()) {
        std::string msg;
        for (std::size_t i = 0; i < problems.size() && i < 8; ++i) msg += problems[i] + "\n";
        if (problems.size() > 8) msg += "... and " + std::to_string(problems.size() - 8) + " more\n";
        return msg;
    }

    std::vector<std::vector<std::string>> rows;
    uint64_t at = 0;
    for (const Job& j : todo) {
        const TensorInfo& t = *j.t;
        const std::string ne0 = std::to_string(t.shape.empty() ? 0 : t.shape[0]);
        const std::string ne1 = std::to_string(t.shape.size() > 1 ? t.shape[1] : 0);
        if (j.native) {
            ++out.served_native;
            rows.push_back({t.name, "0", "0", "0", "0", "0", "0", ne0, ne1, "8", "0", "32"});
            rows.back().insert(rows.back().end(), 7, "0");
            continue;
        }
        const GgufFile& g = model.shard(j.shard);
        if (!model.in_bounds(t, j.shard)) return "unsupported or truncated tensor " + t.name;
        const uint8_t* raw = g.tensor_data(t);
        const uint64_t raw_bytes = tensor_payload_bytes(t);
        Converted got;
        const std::string err =
            convert(t.name, t.type, t.type_name(), raw, raw_bytes, compat_bf16, got);
        if (!err.empty()) return err;
        if (got.has_rec) {
            got.rec.name = t.name;
            got.rec.shard = basename_of(g.path());
            got.rec.source_sha256 = Sha256::hex(raw, (std::size_t)raw_bytes);
            out.conversions.push_back(got.rec);
        }
        rows.push_back({t.name, "0", got.kind, std::to_string(at), std::to_string(got.data.size()), "0",
                        std::to_string(got.dst_bytes), ne0, ne1, "0", "0", "1"});
        rows.back().insert(rows.back().end(), 7, "0");
        out.dense.insert(out.dense.end(), got.data.begin(), got.data.end());
        const std::size_t pad = (std::size_t)((kAlign - got.data.size() % kAlign) % kAlign);
        out.dense.insert(out.dense.end(), pad, 0);
        at += got.data.size() + pad;
    }

    // The arena the pool offsets address: the running sum of the DESTINATION sizes, which for the tensors the loader
    // narrows (kind 3) is half the bytes dense.bin holds for them.
    uint64_t pool = 0;
    for (auto& r : rows) {
        r[5] = std::to_string(pool);
        pool += (std::stoull(r[6]) + kAlign - 1) / kAlign * kAlign;
    }
    out.pool_bytes = pool;
    out.index = "# strata pack index v3 -- generated by tools/strata_pack.cpp (native experts) from " +
                basename_of(model.shard(0).path()) + "\n";
    out.index += "# align " + std::to_string(kAlign) + " pool " + std::to_string(pool) + " tensors " +
                 std::to_string(rows.size()) + "\n";
    for (const auto& r : rows) {
        for (std::size_t i = 0; i < r.size(); ++i) {
            if (i) out.index += ' ';
            out.index += r[i];
        }
        out.index += '\n';
    }
    out.conversions_json = conversions_json(model, compat_bf16, out.conversions);
    return "";
}

} // namespace strata

namespace strata {
namespace {

/// "blk.12.ffn_gate_exps.weight" -> 12; -1 when the name is not a numbered block tensor.
int64_t block_index(const std::string& name) {
    if (name.compare(0, 4, "blk.") != 0) return -1;
    std::size_t i = 4;
    int64_t v = 0;
    bool any = false;
    while (i < name.size() && name[i] >= '0' && name[i] <= '9') {
        v = v * 10 + (name[i] - '0');
        ++i;
        any = true;
        if (v > 1000000) return -1;
    }
    if (!any || i >= name.size() || name[i] != '.') return -1;
    return v;
}

} // namespace

std::string pack_expert_layout(const GgufModel& model, PackExpertTable& out) {
    std::map<std::string, PackExpertRole> by_name;
    int64_t n_layers = 0;
    for (std::size_t s = 0; s < model.size(); ++s) {
        const GgufFile& g = model.shard(s);
        for (const auto& t : g.tensors()) {
            if (!pack_is_expert(t.name)) continue;
            PackExpertRole r;
            r.tensor = &t;
            r.shard = s;
            r.offset = g.data_start() + t.offset;             // absolute in THIS shard
            r.shard_name = s == 0 ? std::string() : basename_of(g.path());
            by_name.emplace(t.name, r);
            const int64_t l = block_index(t.name);
            if (l >= 0 && l + 1 > n_layers) n_layers = l + 1;
        }
    }
    if (by_name.empty())
        return "the model has no expert tensors (blk.N.ffn_{gate,up,down}_exps.weight)";

    // The expert count comes off the router: its second dimension is the experts this model kept, and a pruned
    // model legitimately ships fewer than the canonical 512.
    const TensorInfo* router0 = model.find("blk.0.ffn_gate_inp.weight");
    if (!router0 || router0->shape.size() < 2)
        return "blk.0.ffn_gate_inp.weight is missing or is not two-dimensional; the expert count is read from it";
    const int64_t n_expert = (int64_t)router0->shape[1];
    if (n_expert <= 0) return "blk.0.ffn_gate_inp.weight has no experts in its second dimension";
    for (int64_t l = 0; l < n_layers; ++l) {
        const TensorInfo* r = model.find("blk." + std::to_string(l) + ".ffn_gate_inp.weight");
        if (!r || r->shape.size() < 2 || (int64_t)r->shape[1] != n_expert)
            return "the routers disagree on the expert count; a per-layer pruned model cannot be packed";
    }

    out.layers.clear();
    uint64_t offset = 0;
    int64_t n_split = 0;
    std::vector<std::string> lines;
    for (int64_t l = 0; l < n_layers; ++l) {
        PackExpertLayer L;
        L.layer = l;
        std::string missing;
        for (const char* role : {"gate", "up", "down"}) {
            const std::string name = "blk." + std::to_string(l) + ".ffn_" + role + "_exps.weight";
            const auto it = by_name.find(name);
            if (it == by_name.end()) missing += (missing.empty() ? "" : ", ") + name;
            else L.roles.push_back(it->second);
        }
        if (!missing.empty()) return "layer " + std::to_string(l) + ": missing " + missing;

        std::vector<uint64_t> per;
        for (const PackExpertRole& r : L.roles) {
            const uint64_t bytes = tensor_payload_bytes(*r.tensor);
            if (bytes == 0 || r.tensor->shape.size() != 3 || (int64_t)r.tensor->shape[2] != n_expert)
                return "layer " + std::to_string(l) + ": an expert tensor is not [*, *, " +
                       std::to_string(n_expert) + "] of whole blocks";
            per.push_back(bytes / (uint64_t)n_expert);
        }
        if (per[0] != per[1] || L.roles[0].tensor->type != L.roles[1].tensor->type)
            return "layer " + std::to_string(l) + ": gate and up differ in type";
        L.gu_type = L.roles[0].tensor->type;
        L.d_type = L.roles[2].tensor->type;
        L.blob = per[0] + per[1] + per[2];
        L.offset = offset;

        // The shard column: one name when all three roles share a shard (empty when that is the source shard), or
        // "gate,up,down" when a shard boundary falls inside the layer (v4).
        std::vector<std::string> files;
        for (const PackExpertRole& r : L.roles) files.push_back(r.shard_name);
        std::string column = files[0];
        if (!(files[0] == files[1] && files[1] == files[2])) {
            column = files[0] + "," + files[1] + "," + files[2];
            ++n_split;
        }
        std::string line = std::to_string(l) + " " + std::to_string(L.gu_type) + " " + std::to_string(L.d_type) + " " +
                           std::to_string(L.offset) + " " + std::to_string(L.blob) + " " +
                           std::to_string(L.roles[0].offset) + " " + std::to_string(L.roles[1].offset) + " " +
                           std::to_string(L.roles[2].offset);
        if (!column.empty()) line += " " + column;
        lines.push_back(line);

        offset += L.blob * (uint64_t)n_expert;
        out.layers.push_back(std::move(L));
    }

    const std::string src_name = basename_of(model.shard(0).path());
    std::string head;
    if (n_split) {
        head = "# strata native experts v4: layer gu_type d_type offset blob_bytes gate_off up_off down_off "
               "[shard | gate,up,down] (n_expert " + std::to_string(n_expert) + ", total " + std::to_string(offset) +
               "; absolute offsets in " + src_name + ", or in the named shard beside it - per role where the column "
               "is gate,up,down)\n";
    } else {
        head = "# strata native experts v3: layer gu_type d_type offset blob_bytes gate_off up_off down_off [shard] "
               "(n_expert " + std::to_string(n_expert) + ", total " + std::to_string(offset) + "; absolute offsets "
               "in " + src_name + ", or in the named shard beside it)\n";
    }
    out.text = head;
    for (const std::string& l : lines) out.text += l + "\n";
    out.n_expert = n_expert;
    out.total_bytes = offset;
    return "";
}

bool pack_expert_layer_bytes(const GgufModel& model, const PackExpertLayer& layer, int64_t n_expert,
                             std::vector<uint8_t>& out) {
    out.clear();
    if (n_expert <= 0 || layer.roles.size() != 3) return false;
    std::vector<const uint8_t*> base;
    std::vector<uint64_t> per;
    uint64_t blob = 0;
    for (const PackExpertRole& r : layer.roles) {
        if (!r.tensor || !model.in_bounds(*r.tensor, r.shard)) return false;
        const uint64_t bytes = tensor_payload_bytes(*r.tensor);
        if (bytes == 0 || bytes % (uint64_t)n_expert) return false;
        per.push_back(bytes / (uint64_t)n_expert);
        blob += per.back();
        base.push_back(model.shard(r.shard).tensor_data(*r.tensor));
    }
    if (blob != layer.blob) return false;
    out.reserve((std::size_t)(blob * (uint64_t)n_expert));
    // For each expert in turn, its gate, then up, then down - the reference reshapes each tensor's bytes to
    // (n_expert, -1) and concatenates along the experts, so the split is by byte position within the tensor.
    for (int64_t e = 0; e < n_expert; ++e)
        for (std::size_t i = 0; i < base.size(); ++i) {
            const uint8_t* from = base[i] + (std::size_t)(per[i] * (uint64_t)e);
            out.insert(out.end(), from, from + per[i]);
        }
    return true;
}

} // namespace strata

namespace strata {

std::string pack_experts_source_json(const GgufModel& model, const std::string& native_experts_text,
                                     uint64_t total_bytes) {
    // Python's json.dumps(indent=2) here, not the indent=1 of conversions.json: the reference writes this one at
    // two spaces.  Reproduced because the sidecar is compared byte for byte; a JSON reader does not care.
    std::string o = "{\n";
    o += "  \"schema\": 1,\n";
    o += "  \"shards\": [";
    if (model.size() == 0) {
        o += "],\n";
    } else {
        o += "\n";
        for (std::size_t i = 0; i < model.size(); ++i) {
            o += "    {\n";
            o += "      \"name\": " + json_string(basename_of(model.shard(i).path())) + ",\n";
            o += "      \"size\": " + std::to_string(model.shard(i).file_size()) + "\n";
            o += std::string("    }") + (i + 1 < model.size() ? "," : "") + "\n";
        }
        o += "  ],\n";
    }
    o += "  \"native_experts_sha256\": " + json_string(Sha256::hex(native_experts_text.data(),
                                                                 native_experts_text.size())) + ",\n";
    o += "  \"bytes\": " + std::to_string(total_bytes) + "\n";
    o += "}\n";
    return o;
}

} // namespace strata
