// src/artifact/pack_build.cpp - see include/strata/artifact/pack_build.hpp for what this owns and why.
#include "strata/artifact/pack_build.hpp"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "strata/artifact/canonical_entry.hpp"
#include "strata/artifact/canonical_mapping.hpp"
#include "strata/artifact/pack_writer.hpp"   // pack_is_expert
#include "strata/artifact/sha256.hpp"

namespace strata {
namespace {

bool write_bytes(const std::string& path, const std::vector<uint8_t>& bytes) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) return false;
    if (!bytes.empty()) f.write((const char*)bytes.data(), (std::streamsize)bytes.size());
    return f.good();
}

bool write_text(const std::string& path, const std::string& text) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) return false;
    f.write(text.data(), (std::streamsize)text.size());
    return f.good();
}

} // namespace

std::string sha256_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return {};
    Sha256 h;
    std::vector<char> buf(1u << 20);
    while (f) {
        f.read(buf.data(), (std::streamsize)buf.size());
        const std::streamsize n = f.gcount();
        if (n > 0) h.update(buf.data(), (std::size_t)n);
    }
    const std::array<uint8_t, 32> d = h.finish();
    static const char* digits = "0123456789abcdef";
    std::string s(64, '0');
    for (int i = 0; i < 32; ++i) {
        s[(std::size_t)(2 * i)] = digits[d[(std::size_t)i] >> 4];
        s[(std::size_t)(2 * i + 1)] = digits[d[(std::size_t)i] & 0xF];
    }
    return s;
}

std::string pack_build(const GgufModel& model, const std::string& out_dir, const PackBuildOptions& opts,
                       PackBuildResult& out) {
    out = PackBuildResult{};

    // ---- 1. the marker check, before anything is opened or written
    if (!opts.force) {
        const std::string marker = pack_occupied(out_dir);
        if (!marker.empty())
            return out_dir + " already holds a pack (" + marker +
                   " is there) - refusing to build; delete the directory, pass another --out, or --force to "
                   "rebuild it";
    }
    std::error_code ec;
    std::filesystem::create_directories(out_dir, ec);
    if (ec && !std::filesystem::is_directory(out_dir)) return "cannot create " + out_dir + ": " + ec.message();
    if (opts.force) {
        // The old markers go before a byte is written, so a forced build that stops part-way leaves an unfinished
        // build rather than the old manifest and index over partly new data.
        for (const char* name : {"manifest.json", "index.txt", "native_experts.txt"})
            std::remove((out_dir + "/" + name).c_str());
    }

    const ExpertGeometry& geom = opts.geometry;
    const uint64_t rb = geom.role_bytes(), bb = geom.blob_bytes();

    // ---- 2. experts.bin: one blob per expert per layer, from the three role tensors
    std::vector<uint8_t> experts;
    experts.reserve((std::size_t)((uint64_t)opts.n_layers * geom.NE * bb));
    for (int64_t layer = 0; layer < opts.n_layers; ++layer) {
        const TensorInfo* roles[3] = {nullptr, nullptr, nullptr};
        const uint8_t* base[3] = {nullptr, nullptr, nullptr};
        static const char* role_names[3] = {"gate", "up", "down"};
        for (int r = 0; r < 3; ++r) {
            const std::string name =
                "blk." + std::to_string(layer) + ".ffn_" + role_names[r] + "_exps.weight";
            std::size_t shard = 0;
            roles[r] = model.find(name, &shard);
            if (!roles[r]) return "layer " + std::to_string(layer) + ": missing " + name;
            if (!model.in_bounds(*roles[r], shard)) return name + ": SHORT READ - refusing to build";
            base[r] = model.shard(shard).tensor_data(*roles[r]);
        }
        PackManifestExpertLayer L;
        L.layer = layer;
        L.offset = experts.size();
        for (int r = 0; r < 3; ++r) L.role_offsets[r] = roles[r]->offset;
        out.manifest.expert_layers.push_back(L);

        for (uint64_t e = 0; e < geom.NE; ++e) {
            const std::size_t at = experts.size();
            experts.resize(at + (std::size_t)bb);
            pack_expert_blob(base[0] + e * rb, base[1] + e * rb, base[2] + e * rb, geom, experts.data() + at);
        }
    }

    // ---- 3. the two segments, in the model's own tensor order (embd.bin is the one tensor that goes elsewhere)
    std::vector<uint8_t> dense, embd;
    for (std::size_t s = 0; s < model.size(); ++s) {
        const GgufFile& gf = model.shard(s);
        for (const TensorInfo& t : gf.tensors()) {
            if (pack_is_expert(t.name)) continue;
            const bool is_embd = t.name == "token_embd.weight";
            std::vector<uint8_t>& seg = is_embd ? embd : dense;
            if (!canonical_mapping(t.type_name()))
                return t.name + ": NO MAPPING for " + t.type_name() + " - refusing to build";
            if (!model.in_bounds(t, s)) return t.name + ": SHORT READ - refusing to build";
            PackIndexEntry e;
            e.name = t.name;
            e.file = is_embd ? "embd.bin" : "dense.bin";
            e.source_type = t.type_name();
            std::vector<uint8_t> blob;
            const std::string err = pack_tensor_entry(t.name, t, t.type_name(), gf.tensor_data(t),
                                                      tensor_payload_bytes(t), seg.size(), blob, e.tensor);
            if (!err.empty()) return err;
            seg.insert(seg.end(), blob.begin(), blob.end());
            out.tensors.push_back(std::move(e));
        }
    }

    // ---- 4. the data files
    if (!write_bytes(out_dir + "/experts.bin", experts)) return "cannot write experts.bin";
    if (!write_bytes(out_dir + "/dense.bin", dense)) return "cannot write dense.bin";
    if (!write_bytes(out_dir + "/embd.bin", embd)) return "cannot write embd.bin";

    // ---- 5. the manifest, and the manifest LAST
    PackManifest& m = out.manifest;
    m.shard1 = model.shard(0).path();
    m.shard2 = opts.shard2;
    m.n_layers = opts.n_layers;
    m.align = opts.align;
    m.geometry = geom;
    m.n_experts_per_layer = geom.NE;
    m.tensors = out.tensors;
    m.files_experts = experts.size();
    m.files_dense = dense.size();
    m.files_embd = embd.size();
    if (!opts.skip_hash) {
        m.shard1_sha256 = sha256_file(m.shard1);
        m.shard2_sha256 = opts.shard2.empty() ? std::string() : sha256_file(opts.shard2);
    }
    std::string err;
    const std::string text = pack_manifest_json(m, err);
    if (!err.empty()) return err;
    if (!write_text(out_dir + "/manifest.json", text)) return "cannot write manifest.json";
    return "";
}

} // namespace strata
