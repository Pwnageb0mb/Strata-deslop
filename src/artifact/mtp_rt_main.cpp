// src/artifact/mtp_rt_main.cpp - `strata-mtp-rt`: the C++ port of tools/mtp_rt.py.
//
// The MTP draft layer's runtime files, from the packed MTP GGUF: `experts.bin` (the blob layout), `dense.bin`
// (every other tensor: Q8_0 / BF16 / F32) and `dense.txt` (the index).
//
//    strata-mtp-rt --gguf MTP.gguf --out DIR
//
// Every piece is reused rather than re-derived: `expert_arena::pack_expert_blob` for the blobs (proven against
// mtp_rt.py's own `blob_of` at production geometry), `q8_0_from_f32` for the Q8_0 projections (proven against its
// `q8_0`), and `gguf_reader` in place of gguf-py.  The GEOMETRY IS READ FROM THE FILE, not hardcoded: the
// reference's H/FF/NE are module constants it asserts the tensors against, and a tool that reads them cannot
// disagree with the file it was given.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <set>
#include <string>
#include <vector>

#include "strata/artifact/expert_arena.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include "strata/artifact/q8_0_quant.hpp"

namespace {

/// The projections the reference quantizes to Q8_0 so the engine's multi-column MMVQ can run them.  The rest of
/// the BF16 tensors are carried through as BF16.
const std::set<std::string>& q8_set() {
    static const std::set<std::string> s = {
        "fc_embedding.weight", "fc_hidden.weight", "self_attn.q_proj.weight", "self_attn.k_proj.weight",
        "self_attn.v_proj.weight", "self_attn.o_proj.weight", "self_attn.indexer.index_qk_proj.weight",
        "mlp.shared_expert.gate_proj.weight", "mlp.shared_expert.up_proj.weight",
        "mlp.shared_expert.down_proj.weight"};
    return s;
}

std::string short_of(const std::string& name) {
    std::string s = name;
    if (s.rfind("mtp.", 0) == 0) s = s.substr(4);
    if (s.rfind("layers.0.", 0) == 0) s = s.substr(9);
    return s;
}

bool write_bytes(const std::string& path, const std::vector<uint8_t>& b) {
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (f == nullptr) return false;
    const bool ok = b.empty() || std::fwrite(b.data(), 1, b.size(), f) == b.size();
    return std::fclose(f) == 0 && ok;
}

int usage() {
    std::fprintf(stderr, "usage: strata-mtp-rt --gguf MTP.gguf --out DIR\n");
    return 2;
}

} // namespace

int main(int argc, char** argv) {
    std::string gguf_path, out_dir;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "-h" || a == "--help") return usage();
        if (a == "--gguf" && i + 1 < argc) gguf_path = argv[++i];
        else if (a == "--out" && i + 1 < argc) out_dir = argv[++i];
        else {
            std::fprintf(stderr, "strata-mtp-rt: unknown option %s\n", a.c_str());
            return usage();
        }
    }
    if (gguf_path.empty() || out_dir.empty()) return usage();

    strata::GgufFile g(gguf_path);
    const strata::TensorInfo* gu = g.find("mtp.layers.0.mlp.experts.gate_up_proj");
    const strata::TensorInfo* dn = g.find("mtp.layers.0.mlp.experts.down_proj");
    if (gu == nullptr || dn == nullptr) {
        std::fprintf(stderr, "strata-mtp-rt: %s carries no mtp.layers.0.mlp.experts.{gate_up,down}_proj\n",
                     gguf_path.c_str());
        return 1;
    }
    // the shapes are the GGUF's own order: gate_up is [H, 2*FF, NE] and down is [FF, H, NE]
    strata::ExpertGeometry geom;
    if (gu->shape.size() != 3 || dn->shape.size() != 3 || gu->shape[1] % 2 != 0) {
        std::fprintf(stderr, "strata-mtp-rt: the expert tensors are not [H, 2*FF, NE] and [FF, H, NE]\n");
        return 1;
    }
    geom.H = gu->shape[0];
    geom.FF = gu->shape[1] / 2;
    geom.NE = gu->shape[2];
    if (dn->shape[0] != geom.FF || dn->shape[1] != geom.H || dn->shape[2] != geom.NE) {
        std::fprintf(stderr, "strata-mtp-rt: the down projection is %llux%llux%llu, not FFxHxNE (%llux%llux%llu)\n",
                     (unsigned long long) dn->shape[0], (unsigned long long) dn->shape[1],
                     (unsigned long long) dn->shape[2], (unsigned long long) geom.FF,
                     (unsigned long long) geom.H, (unsigned long long) geom.NE);
        return 1;
    }

    std::error_code ec;
    std::filesystem::create_directories(out_dir, ec);

    // ---- experts.bin: the blob layout, one expert after another
    const uint64_t rb = geom.role_bytes(), bb = geom.blob_bytes();
    const uint8_t* gub = g.tensor_data(*gu);
    const uint8_t* dnb = g.tensor_data(*dn);
    std::vector<uint8_t> experts((std::size_t) (geom.NE * bb));
    for (uint64_t e = 0; e < geom.NE; ++e) {
        // gate is the first FF rows of gu and up the next FF: one role's bytes each
        strata::pack_expert_blob(gub + e * 2 * rb, gub + e * 2 * rb + rb, dnb + e * rb, geom,
                                 experts.data() + e * bb);
    }
    if (!write_bytes(out_dir + "/experts.bin", experts)) {
        std::fprintf(stderr, "strata-mtp-rt: cannot write %s/experts.bin\n", out_dir.c_str());
        return 1;
    }

    // ---- dense.bin + dense.txt: every other tensor, in the file's own order
    std::vector<uint8_t> dense;
    std::string index;
    std::size_t n_tensors = 0;
    std::vector<float> f32;
    std::vector<uint8_t> raw_out;
    for (const strata::TensorInfo& t : g.tensors()) {
        if (t.name.find("experts.") != std::string::npos) continue;
        const std::string short_name = short_of(t.name);
        const uint8_t* raw = g.tensor_data(t);
        const uint64_t payload = strata::tensor_payload_bytes(t);
        std::string kind;
        uint64_t rows = 1, cols = 0;
        raw_out.clear();
        if (t.type == 0) {                          // F32: a norm weight, raw GemmaRMSNorm w -> 1 + w
            const std::size_t n = (std::size_t) (payload / 4);
            f32.resize(n);
            for (std::size_t i = 0; i < n; ++i) {
                float v = 0.0f;
                std::memcpy(&v, raw + i * 4, 4);
                f32[i] = v + 1.0f;
            }
            raw_out.assign((const uint8_t*) f32.data(), (const uint8_t*) f32.data() + n * 4);
            kind = "f32";
            cols = n;
        } else {                                    // BF16 (type 30)
            // gguf-py presents the data with the dims reversed, which is where rows/cols come from; the BYTES are
            // the file's own order either way, so nothing is reordered here
            rows = t.shape.size() > 1 ? t.shape[1] : 1;
            cols = t.shape.empty() ? 0 : t.shape[0];
            const std::size_t n = (std::size_t) (payload / 2);
            if (q8_set().count(short_name) != 0) {
                f32.resize(n);
                for (std::size_t i = 0; i < n; ++i) {
                    uint16_t u = 0;
                    std::memcpy(&u, raw + i * 2, 2);
                    const uint32_t bits = (uint32_t) u << 16;   // the reference's u16 << 16, viewed as f32
                    std::memcpy(&f32[i], &bits, 4);
                }
                raw_out.resize(n / strata::kQ8_0BlockValues * strata::kQ8_0BlockBytes);
                strata::q8_0_from_f32(f32.data(), n, raw_out.data());
                kind = "q8_0";
            } else {
                raw_out.assign(raw, raw + payload);
                kind = "bf16";
            }
        }
        // every tensor starts on a 256-byte boundary in dense.bin
        const std::size_t pad = (256 - dense.size() % 256) % 256;
        dense.insert(dense.end(), pad, 0);
        const std::size_t offset = dense.size();
        dense.insert(dense.end(), raw_out.begin(), raw_out.end());
        index += short_name + " " + kind + " " + std::to_string(rows) + " " + std::to_string(cols) + " " +
                 std::to_string(offset) + " " + std::to_string(raw_out.size()) + "\n";
        ++n_tensors;
    }
    if (!write_bytes(out_dir + "/dense.bin", dense)) {
        std::fprintf(stderr, "strata-mtp-rt: cannot write %s/dense.bin\n", out_dir.c_str());
        return 1;
    }
    {
        std::FILE* f = std::fopen((out_dir + "/dense.txt").c_str(), "wb");
        if (f == nullptr || std::fwrite(index.data(), 1, index.size(), f) != index.size()) {
            if (f != nullptr) std::fclose(f);
            std::fprintf(stderr, "strata-mtp-rt: cannot write %s/dense.txt\n", out_dir.c_str());
            return 1;
        }
        std::fclose(f);
    }

    std::printf("experts.bin %llu B, dense.bin %zu B, %zu tensors -> %s\n",
                (unsigned long long) experts.size(), dense.size(), n_tensors, out_dir.c_str());
    for (const std::string& line : [&] {
             std::vector<std::string> v;
             std::size_t at = 0;
             while (at < index.size()) {
                 const std::size_t nl = index.find('\n', at);
                 v.push_back(index.substr(at, nl - at));
                 at = nl + 1;
             }
             return v;
         }())
        std::printf("  %s\n", line.c_str());
    return 0;
}
