// src/artifact/mtp_pack_main.cpp - `strata-mtp-pack`: the C++ port of tools/mtp_pack.py.
//
// The MTP draft head as a Strata GGUF, from the BF16 tensors tools/mtp_fetch.py wrote:
//
//   strata-mtp-pack --src <mtp-bf16 dir> --experts q2_0|q4_0|q8_0 --out <mtp.gguf> [--check-experts N]
//
// Dense tensors (attention, indexer, hyper-connections, shared expert, router, fc/norms) stay BF16 (F32 for the
// 1-D norms).  The routed experts keep the checkpoint's fused layout and are quantized along the last (input)
// axis; the format is picked by MEASURED draft acceptance elsewhere, and the per-tensor reconstruction error in
// the report is a sanity check, not the criterion.
//
// The container is llama.cpp's gguf-py writer, so this reproduces ITS layout: `general.architecture` is the first
// KV (the writer's own constructor adds it), the tensor directory carries a RUNNING offset of the aligned tensor
// sizes, and the data section is the tensors in insertion order with alignment padding after each.  gguf_write.hpp
// is the GGUF-writing half of this directory and supplies the primitives.
//
// The three quantizers are in mtp_pack_quant.hpp - transcriptions of THIS reference's arithmetic, not ggml's.
#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "strata/artifact/checkpoint_index.hpp"
#include "strata/artifact/dequant.hpp"
#include "strata/artifact/gguf_write.hpp"
#include "strata/artifact/json.hpp"
#include "strata/artifact/mtp_pack_quant.hpp"
#include "strata/artifact/sha256.hpp"

namespace {

using strata::artifact::as_count;
using strata::artifact::field;
using strata::artifact::read_file;

const char* kExpertTensors[] = {"mtp.layers.0.mlp.experts.gate_up_proj",
                                "mtp.layers.0.mlp.experts.down_proj"};

struct Format {
    const char* name;
    uint64_t block;        ///< elements per block
    uint64_t block_bytes;  ///< bytes per block
    std::uint32_t ggml_type;
    const char* quantizer;
};

const Format kQ2_0 = {"q2_0", strata::kQ2_0BlockValues, strata::kQ2_0BlockBytes, 42, "per-block MSE scale search"};
const Format kQ4_0 = {"q4_0", strata::kQ4_0BlockValues, strata::kQ4_0BlockBytes, 2, "ggml reference"};
const Format kQ8_0 = {"q8_0", 32, 34, 8, "ggml reference"};

void quantize(const Format& f, const float* x, std::size_t n, uint8_t* out) {
    if (f.ggml_type == 42) strata::q2_0_from_f32(x, n, out);
    else if (f.ggml_type == 2) strata::q4_0_from_f32(x, n, out);
    else strata::q8_0_rint_from_f32(x, n, out);
}

/// The reference's `dequant`, for the report only.  The scalar dequantizers are the engine's own (dequant.hpp),
/// transcribed from ggml; they are the inverse of the quantizers above and the reference's numpy `dequant` is
/// the same statement.
void dequant(const Format& f, const uint8_t* blob, std::size_t n, float* out) {
    const std::size_t block = f.block, bb = f.block_bytes;
    for (std::size_t b = 0; b < n / block; ++b) {
        if (f.ggml_type == 42) strata::dequantize_q2_0(blob + b * bb, out + b * block);
        else if (f.ggml_type == 2) strata::dequantize_q4_0(blob + b * bb, out + b * block);
        else strata::dequantize_q8_0(blob + b * bb, out + b * block);
    }
}

int usage() {
    std::fprintf(stderr, "usage: strata-mtp-pack --src <dir> --experts q2_0|q4_0|q8_0 --out <gguf> "
                         "[--check-experts N]\n");
    return 2;
}

std::string hex_of(const unsigned char* p, std::size_t n) {
    static const char* digits = "0123456789abcdef";
    std::string s;
    for (std::size_t i = 0; i < n; ++i) {
        s += digits[p[i] >> 4];
        s += digits[p[i] & 0xF];
    }
    return s;
}

/// Python's `repr(round(v, n))`, which is what `json.dumps` writes for a float: correctly rounded to n decimal
/// places (half to even, as both `round` and printf's default mode are), then the SHORTEST string that
/// round-trips - with a `.0` for a whole number, as `repr` writes one.
std::string json_number(double v, int decimals) {
    char buf[64];
    std::snprintf(buf, sizeof buf, "%.*f", decimals, v);
    const double r = std::strtod(buf, nullptr);
    char out[64];
    const std::to_chars_result c = std::to_chars(out, out + sizeof out, r);
    std::string s(out, c.ptr);
    if (s.find_first_of(".eE") == std::string::npos) s += ".0";
    return s;
}

/// One tensor's entry in the sidecar report.
struct ReportEntry {
    std::string tensor;
    std::string format;
    uint64_t bytes = 0;
    bool has_error = false;
    double error = 0.0;
    double seconds = 0.0;
};

/// The report, laid out the way `json.dumps(report, indent=1)` lays it out - one shape, so one emitter.
std::string report_json(const std::vector<ReportEntry>& entries, uint64_t total) {
    std::string s = "{\n";
    s += " \"total_bytes\": " + std::to_string(total) + ",\n";
    s += " \"tensors\": [";
    if (entries.empty()) {
        s += "]\n";
    } else {
        s += "\n";
        for (std::size_t i = 0; i < entries.size(); ++i) {
            const ReportEntry& e = entries[i];
            s += "  {\n";
            s += "   \"tensor\": \"" + e.tensor + "\",\n";
            s += "   \"format\": \"" + e.format + "\",\n";
            s += "   \"bytes\": " + std::to_string(e.bytes);
            if (e.has_error) {
                s += ",\n   \"relative_rms_error_first_experts\": " + json_number(e.error, 5);
                s += ",\n   \"seconds\": " + json_number(e.seconds, 1);
            }
            s += "\n  }";
            if (i + 1 < entries.size()) s += ",";
            s += "\n";
        }
        s += " ]\n";
    }
    s += "}";   // no trailing newline: json.dumps writes none, and this file is compared byte for byte
    return s;
}

/// One tensor of the output, resolved from the manifest before anything is written.
struct Tensor {
    std::string name;
    std::vector<uint64_t> dims;    ///< GGUF order: dim 0 varies fastest (the reference writes ti.shape reversed)
    std::uint32_t type = 0;
    uint64_t nbytes = 0;
    int kind = 0;                  ///< 0 = expert, 1 = 1-D F32, 2 = BF16
    std::vector<uint64_t> shape;   ///< the manifest's own (numpy) order
    std::string file;              ///< the bf16 file, relative to --src
};

/// `n` little-endian u16 values from `path`, widened to f32 - the reference's
/// `(raw.astype(np.uint32) << 16).view(np.float32)`.
void read_bf16_f32(const std::string& path, uint64_t skip, uint64_t n, std::vector<float>& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    f.seekg((std::streamoff) (skip * 2));
    std::vector<uint16_t> raw((std::size_t) n);
    f.read((char*) raw.data(), (std::streamsize) (n * 2));
    if ((uint64_t) f.gcount() != n * 2) throw std::runtime_error("short read in " + path);
    out.resize((std::size_t) n);
    for (std::size_t i = 0; i < raw.size(); ++i) {
        const std::uint32_t bits = (std::uint32_t) raw[i] << 16;
        std::memcpy(&out[i], &bits, 4);
    }
}

void write_pad(std::ofstream& w, uint64_t n) {
    static const char zeros[32] = {0};
    const std::size_t pad = strata::artifact::gguf_pad(n);
    if (pad != 0) w.write(zeros, (std::streamsize) pad);
}

} // namespace

int main(int argc, char** argv) {
    std::string src, out, experts = "q2_0";
    int check_experts = 8;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "-h" || a == "--help") return usage();
        if (a == "--src" && i + 1 < argc) src = argv[++i];
        else if (a == "--out" && i + 1 < argc) out = argv[++i];
        else if (a == "--experts" && i + 1 < argc) experts = argv[++i];
        else if (a == "--check-experts" && i + 1 < argc) check_experts = std::atoi(argv[++i]);
        else {
            std::fprintf(stderr, "strata-mtp-pack: unknown option %s\n", a.c_str());
            return usage();
        }
    }
    if (src.empty() || out.empty()) return usage();
    if (experts != "q2_0" && experts != "q4_0" && experts != "q8_0") {
        std::fprintf(stderr, "strata-mtp-pack: --experts must be q2_0, q4_0 or q8_0\n");
        return usage();
    }
    const Format& fmt = experts == "q2_0" ? kQ2_0 : (experts == "q4_0" ? kQ4_0 : kQ8_0);

    try {
        const strata::json::Value manifest = strata::json::parse(read_file(src + "/mtp-manifest.json"));
        if (!manifest.is_arr()) throw std::runtime_error("mtp-manifest.json is not an array");

        // ---- the tensors, sorted by name (the reference iterates `sorted(manifest, key=name)`)
        std::vector<Tensor> tensors;
        std::string hash_src = "{";
        std::vector<std::pair<std::string, std::string>> named;
        for (const strata::json::Value& t : manifest.items) {
            if (!t.is_obj()) throw std::runtime_error("a manifest entry is not an object");
            const strata::json::Value& name = field(t, "name", "a manifest entry");
            const strata::json::Value& file = field(t, "file", name.text);
            const strata::json::Value& shape = field(t, "shape", name.text);
            const strata::json::Value& sha = field(t, "sha256", name.text);
            if (!shape.is_arr() || shape.items.empty()) throw std::runtime_error(name.text + ": no shape");
            named.emplace_back(name.text, sha.text);

            Tensor o;
            o.name = name.text;
            o.file = src + "/" + file.text;
            for (const strata::json::Value& d : shape.items) o.shape.push_back(as_count(d, name.text + " shape"));
            uint64_t elems = 1;
            for (uint64_t d : o.shape) elems *= d;

            const bool is_expert = o.name == kExpertTensors[0] || o.name == kExpertTensors[1];
            if (is_expert) {
                if (o.shape.size() < 2 || o.shape.back() % fmt.block != 0) {
                    throw std::runtime_error(o.name + ": inner dim " + std::to_string(o.shape.back()) +
                                             " is not a multiple of " + std::to_string(fmt.block));
                }
                // the reference passes the BYTE shape with the quantized type, and gguf-py converts it back to
                // the element shape - so the tensor directory carries the model's own dims, reversed
                o.kind = 0;
                o.type = fmt.ggml_type;
                o.nbytes = elems / fmt.block * fmt.block_bytes;
            } else if (o.shape.size() == 1) {
                o.kind = 1;
                o.type = strata::artifact::kGgmlTypeF32;
                o.nbytes = elems * 4;
            } else {
                o.kind = 2;
                o.type = strata::artifact::kGgmlTypeBf16;
                o.nbytes = elems * 2;
            }
            o.dims.assign(o.shape.rbegin(), o.shape.rend());
            tensors.push_back(std::move(o));
        }
        std::sort(tensors.begin(), tensors.end(),
                  [](const Tensor& a, const Tensor& b) { return a.name < b.name; });
        std::sort(named.begin(), named.end());
        for (std::size_t i = 0; i < named.size(); ++i)
            hash_src += (i ? ", " : "") + std::string("\"") + named[i].first + "\": \"" + named[i].second + "\"";
        hash_src += "}";
        strata::Sha256 hasher;
        hasher.update(hash_src.data(), hash_src.size());
        const std::array<unsigned char, 32> digest = hasher.finish();

        // ---- the header: gguf-py's KV order is the writer's own `general.architecture` first, then ours
        std::string kvs = strata::artifact::kv_string("general.architecture", "qwen4exp-mtp") +
                          strata::artifact::kv_string("strata.mtp.source",
                                                      "Qwen/Qwen3.8-Flash-Next BF16 checkpoint, mtp.* tensors") +
                          strata::artifact::kv_string("strata.mtp.source_sha256", hex_of(digest.data(), digest.size())) +
                          strata::artifact::kv_string("strata.mtp.expert_format", experts) +
                          strata::artifact::kv_string("strata.mtp.expert_quantizer", fmt.quantizer);
        std::string head = strata::artifact::gguf_head(tensors.size(), 5) + kvs;
        uint64_t offset = 0;
        for (const Tensor& t : tensors) {
            head += strata::artifact::gguf_tensor_info(t.name, t.dims.data(), t.dims.size(), t.type, offset);
            offset += t.nbytes + strata::artifact::gguf_pad(t.nbytes);
        }
        strata::artifact::gguf_align(head);

        std::ofstream w(out, std::ios::binary);
        if (!w) throw std::runtime_error("cannot write " + out);
        w.write(head.data(), (std::streamsize) head.size());

        std::vector<ReportEntry> report;
        uint64_t total = 0;
        for (const Tensor& t : tensors) {
            const auto t0 = std::chrono::steady_clock::now();
            ReportEntry e;
            e.tensor = t.name;
            e.bytes = t.nbytes;
            if (t.kind == 0) {
                e.format = experts;
                uint64_t per_expert = 1;
                for (std::size_t i = 1; i < t.shape.size(); ++i) per_expert *= t.shape[i];
                const std::size_t bytes = (std::size_t) (per_expert / fmt.block * fmt.block_bytes);
                std::vector<float> x;
                std::vector<uint8_t> blob(bytes);
                std::vector<std::vector<float>> refs;
                std::vector<std::vector<uint8_t>> parts;
                for (uint64_t ex = 0; ex < t.shape[0]; ++ex) {
                    read_bf16_f32(t.file, ex * per_expert, per_expert, x);
                    quantize(fmt, x.data(), x.size(), blob.data());
                    w.write((const char*) blob.data(), (std::streamsize) bytes);
                    if (ex < (uint64_t) check_experts) {   // the error report only looks at the first few
                        refs.push_back(x);
                        parts.push_back(blob);
                    }
                }
                if (!refs.empty()) {
                    std::vector<double> errs;
                    std::vector<float> got, diff;
                    for (std::size_t i = 0; i < refs.size(); ++i) {
                        const float* ref = refs[i].data();
                        got.resize(refs[i].size());
                        dequant(fmt, parts[i].data(), refs[i].size(), got.data());
                        diff.resize(refs[i].size());
                        for (std::size_t j = 0; j < diff.size(); ++j) {
                            const float d = got[j] - ref[j];
                            diff[j] = d * d;
                        }
                        const float num = std::sqrt(strata::np_mean_f32(diff.data(), diff.size()));
                        for (std::size_t j = 0; j < diff.size(); ++j) diff[j] = ref[j] * ref[j];
                        const double den = (double) std::sqrt(strata::np_mean_f32(diff.data(), diff.size())) + 1e-12;
                        errs.push_back((double) num / den);
                    }
                    e.has_error = true;
                    e.error = strata::np_mean_f64(errs.data(), errs.size());
                }
            } else if (t.kind == 1) {
                e.format = "F32";
                uint64_t n = 1;
                for (uint64_t d : t.shape) n *= d;
                std::vector<float> x;
                read_bf16_f32(t.file, 0, n, x);
                w.write((const char*) x.data(), (std::streamsize) (n * 4));
            } else {
                e.format = "BF16";
                std::ifstream f(t.file, std::ios::binary);
                if (!f) throw std::runtime_error("cannot open " + t.file);
                std::vector<char> buf(64u << 20);
                uint64_t left = t.nbytes;
                while (left > 0) {
                    const std::size_t want = (std::size_t) std::min<uint64_t>(buf.size(), left);
                    f.read(buf.data(), (std::streamsize) want);
                    if ((std::size_t) f.gcount() != want) throw std::runtime_error("short read in " + t.file);
                    w.write(buf.data(), (std::streamsize) want);
                    left -= want;
                }
            }
            write_pad(w, t.nbytes);
            e.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            total += t.nbytes;
            report.push_back(e);
            std::printf("%s: %s %.1f MB\n", t.name.c_str(), e.format.c_str(), (double) e.bytes / 1e6);
        }
        w.close();
        if (!w) throw std::runtime_error("write failed");

        const std::string rep = report_json(report, total);
        std::ofstream r(out + ".report.json", std::ios::binary);
        if (!r) throw std::runtime_error("cannot write " + out + ".report.json");
        r.write(rep.data(), (std::streamsize) rep.size());
        r.close();

        std::printf("wrote %s: %.3f GB of tensor data\n", out.c_str(), (double) total / 1e9);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "strata-mtp-pack: %s\n", e.what());
        return 1;
    }
    return 0;
}
