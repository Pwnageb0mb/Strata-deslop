// src/artifact/pack_layer_main.cpp - `strata-pack-layer`: the C++ port of tools/pack_layer.py.
//
// The pack-layout expert arena, cut from a model's own GGUF: for every layer, each expert's three Q2_0 role
// tensors (`ffn_gate_exps`, `ffn_up_exps`, `ffn_down_exps`) are relaid out into one blob, and the blobs are
// written end to end, plus the sidecar manifest `build()` writes next to them.
//
//   strata-pack-layer --gguf <shard1.gguf> --out <arena.bin> [--layers N]
//
// The relayout itself is NOT reimplemented here: it is `expert_arena::pack_expert_blob`, already proven byte for
// byte against the reference's own `experts.bin` by `expert_arena_test`.  What this file owns is the tool around
// it - the GGUF read, the per-layer loop, the running offset, and the manifest.
//
// The geometry is READ FROM THE FILE rather than hardcoded.  The reference's H/FF/NE are module constants it
// ASSERTS the tensors against; a tool that derives them from the tensors cannot disagree with the file it was
// given, and the same code runs over the fixture-sized model in the test.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "strata/artifact/expert_arena.hpp"
#include "strata/artifact/gguf_reader.hpp"

namespace {

int usage() {
    std::fprintf(stderr, "usage: strata-pack-layer --gguf <shard1.gguf> --out <arena.bin> [--layers N]\n");
    return 2;
}

/// One layer's entry in the manifest: where its blobs start, and the three role tensors it was cut from.
struct LayerEntry {
    uint64_t layer = 0;
    uint64_t offset = 0;
    uint64_t role_offsets[3] = {0, 0, 0};   // gate, up, down
};

/// The sidecar manifest, laid out the way Python's `json.dumps(manifest, indent=2)` lays it out - the file is
/// compared byte for byte and two writers of one format have to agree.  One shape, so one emitter.
std::string manifest_json(const strata::ExpertGeometry& g, const std::vector<LayerEntry>& layers, uint64_t total) {
    const strata::ExpertArenaOffsets o = g.offsets();
    std::string s = "{\n";
    s += "  \"blob_bytes\": " + std::to_string(g.blob_bytes()) + ",\n";
    s += "  \"offsets\": {\n";
    s += "    \"gate_up_codes\": " + std::to_string(o.gate_up_codes) + ",\n";
    s += "    \"down_codes\": " + std::to_string(o.down_codes) + ",\n";
    s += "    \"gate_up_scales\": " + std::to_string(o.gate_up_scales) + ",\n";
    s += "    \"down_scales\": " + std::to_string(o.down_scales) + "\n";
    s += "  },\n";
    s += "  \"layers\": [";
    if (layers.empty()) {
        s += "],\n";
    } else {
        s += "\n";
        for (std::size_t i = 0; i < layers.size(); ++i) {
            const LayerEntry& L = layers[i];
            s += "    {\n";
            s += "      \"layer\": " + std::to_string(L.layer) + ",\n";
            s += "      \"offset\": " + std::to_string(L.offset) + ",\n";
            s += "      \"role_offsets\": {\n";
            s += "        \"gate\": " + std::to_string(L.role_offsets[0]) + ",\n";
            s += "        \"up\": " + std::to_string(L.role_offsets[1]) + ",\n";
            s += "        \"down\": " + std::to_string(L.role_offsets[2]) + "\n";
            s += "      }\n";
            s += std::string("    }") + (i + 1 < layers.size() ? "," : "") + "\n";
        }
        s += "  ],\n";
    }
    s += "  \"n_layers\": " + std::to_string(layers.size()) + ",\n";
    s += "  \"total_bytes\": " + std::to_string(total) + "\n";
    s += "}";   // no trailing newline: json.dumps writes none, and this file is compared byte for byte
    return s;
}

std::string layer_name(uint64_t layer, const char* role) {
    return "blk." + std::to_string(layer) + ".ffn_" + role + "_exps.weight";
}

/// The three role tensors of one layer, checked to be the Q2_0 geometry the blob layout needs.
void check_role(const strata::TensorInfo* t, const std::string& name, const uint64_t* want_shape,
                const strata::ExpertGeometry& g) {
    if (t == nullptr) throw std::runtime_error("the GGUF carries no " + name);
    if (t->type != 42) throw std::runtime_error(name + " is " + t->type_name() + ", not Q2_0");
    if (t->shape.size() != 3 || t->shape[0] != want_shape[0] || t->shape[1] != want_shape[1] ||
        t->shape[2] != want_shape[2]) {
        throw std::runtime_error(name + " is not the geometry the other role tensors declare");
    }
    if (strata::tensor_payload_bytes(*t) != g.NE * g.role_bytes()) {
        throw std::runtime_error(name + ": its payload is not NE expert runs of role_bytes");
    }
}

} // namespace

int main(int argc, char** argv) {
    std::string gguf_path, out;
    uint64_t n_layers = 1;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "-h" || a == "--help") return usage();
        if (a == "--gguf" && i + 1 < argc) gguf_path = argv[++i];
        else if (a == "--out" && i + 1 < argc) out = argv[++i];
        else if (a == "--layers" && i + 1 < argc) n_layers = std::strtoull(argv[++i], nullptr, 10);
        else {
            std::fprintf(stderr, "strata-pack-layer: unknown option %s\n", a.c_str());
            return usage();
        }
    }
    if (gguf_path.empty() || out.empty()) return usage();

    try {
        const strata::GgufFile g(gguf_path);

        // ---- the geometry, from the file's own first layer
        const strata::TensorInfo* gate0 = g.find(layer_name(0, "gate"));
        if (gate0 == nullptr) throw std::runtime_error(gguf_path + " carries no blk.0.ffn_gate_exps.weight");
        if (gate0->shape.size() != 3) throw std::runtime_error("blk.0.ffn_gate_exps.weight is not 3-D");
        strata::ExpertGeometry geom;
        geom.H = gate0->shape[0];
        geom.FF = gate0->shape[1];
        geom.NE = gate0->shape[2];
        if (geom.H % geom.QK != 0 || geom.FF % geom.QK != 0) {
            throw std::runtime_error("H and FF must be whole Q2_0 blocks (QK=" + std::to_string(geom.QK) + ")");
        }
        if (geom.H * geom.FF * geom.block_bytes % geom.QK != 0) throw std::runtime_error("bad blob geometry");

        const uint64_t rb = geom.role_bytes(), bb = geom.blob_bytes();
        const std::uint64_t gate_shape[3] = {geom.H, geom.FF, geom.NE};
        const std::uint64_t down_shape[3] = {geom.FF, geom.H, geom.NE};

        const std::filesystem::path out_path(out);
        if (out_path.has_parent_path()) std::filesystem::create_directories(out_path.parent_path());
        std::filesystem::path man_path = out_path;
        man_path.replace_extension(".json");

        std::ofstream w(out, std::ios::binary);
        if (!w) throw std::runtime_error("cannot write " + out);

        std::vector<LayerEntry> layers;
        std::vector<uint8_t> blob((std::size_t) bb);
        uint64_t written = 0;
        for (uint64_t layer = 0; layer < n_layers; ++layer) {
            const strata::TensorInfo* gate = g.find(layer_name(layer, "gate"));
            const strata::TensorInfo* up = g.find(layer_name(layer, "up"));
            const strata::TensorInfo* down = g.find(layer_name(layer, "down"));
            check_role(gate, layer_name(layer, "gate"), gate_shape, geom);
            check_role(up, layer_name(layer, "up"), gate_shape, geom);
            check_role(down, layer_name(layer, "down"), down_shape, geom);

            const uint8_t* gb = g.tensor_data(*gate);
            const uint8_t* ub = g.tensor_data(*up);
            const uint8_t* db = g.tensor_data(*down);

            LayerEntry entry;
            entry.layer = layer;
            entry.offset = written;
            entry.role_offsets[0] = gate->offset;
            entry.role_offsets[1] = up->offset;
            entry.role_offsets[2] = down->offset;
            layers.push_back(entry);

            // expert e is a CONTIGUOUS run inside each role tensor (the expert index is the last axis)
            for (uint64_t e = 0; e < geom.NE; ++e) {
                strata::pack_expert_blob(gb + e * rb, ub + e * rb, db + e * rb, geom, blob.data());
                w.write((const char*) blob.data(), (std::streamsize) bb);
                if (!w) throw std::runtime_error("write failed at layer " + std::to_string(layer));
            }
            written += geom.NE * bb;
            std::printf("  layer %llu: wrote %llu blobs at %llu\n", (unsigned long long) layer,
                        (unsigned long long) geom.NE, (unsigned long long) entry.offset);
        }
        w.close();
        if (!w) throw std::runtime_error("write failed");

        const std::string manifest = manifest_json(geom, layers, written);
        std::ofstream m(man_path, std::ios::binary);
        if (!m) throw std::runtime_error("cannot write " + man_path.string());
        m.write(manifest.data(), (std::streamsize) manifest.size());
        m.close();
        if (!m) throw std::runtime_error("write failed");
    } catch (const std::exception& e) {
        std::fprintf(stderr, "strata-pack-layer: %s\n", e.what());
        return 1;
    }
    return 0;
}
