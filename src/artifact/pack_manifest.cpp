// src/artifact/pack_manifest.cpp - see include/strata/artifact/pack_manifest.hpp for what this owns and why.
#include "strata/artifact/pack_manifest.hpp"

#include <charconv>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

namespace strata {
namespace {

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

/// A float the way Python's repr writes it: shortest round-trip, always with a fraction or an exponent.
std::string json_number(double v) {
    char buf[48];
    const std::to_chars_result r = std::to_chars(buf, buf + sizeof(buf), v);
    std::string s(buf, r.ptr);
    if (s.find_first_of(".eE") == std::string::npos) s += ".0";
    return s;
}

std::string json_bool(bool b) { return b ? "true" : "false"; }

std::string pad(int n) { return std::string((std::size_t)n, ' '); }

/// One {"offset": n, "bytes": m} plane object.  The opening brace is written by the caller, right after the key.
std::string plane_json(const PackPlane& p, int ind) {
    return "{\n" + pad(ind + 1) + "\"offset\": " + std::to_string(p.offset) + ",\n" + pad(ind + 1) +
           "\"bytes\": " + std::to_string(p.bytes) + "\n" + pad(ind) + "}";
}

/// One tensor's manifest entry, in the order `tensor_entry` builds it: the common fields, then the element count
/// and the form's own, then the planes, then the file the build put it in.
std::string entry_json(const PackIndexEntry& e, int ind) {
    const PackTensorEntry& v = e.tensor;
    const int k = ind + 1;
    // The opening brace is written by the caller, right after the tensor's name: `"name": {`.
    std::string o = "{\n";
    auto kv = [&](const std::string& key, const std::string& val, bool comma = true) {
        o += pad(k) + "\"" + key + "\": " + val + (comma ? "," : "") + "\n";
    };
    kv("name", json_string(v.name));
    kv("source_type", json_string(v.source_type));
    kv("form", json_string(v.form));
    kv("codebook", v.codebook.empty() ? "null" : json_string(v.codebook));
    kv("code_bias", std::to_string(v.code_bias));
    kv("has_offset", json_bool(v.has_offset));
    kv("block_elems", std::to_string(v.block_elems));
    kv("block_bytes", std::to_string(v.block_bytes));
    kv("group_elems", std::to_string(v.group_elems));
    o += pad(k) + "\"shape\": [";
    if (v.shape.empty()) {
        o += "],\n";
    } else {
        o += "\n";
        for (std::size_t i = 0; i < v.shape.size(); ++i)
            o += pad(k + 1) + std::to_string(v.shape[i]) + (i + 1 < v.shape.size() ? "," : "") + "\n";
        o += pad(k) + "],\n";
    }
    kv("source_offset", std::to_string(v.source_offset));
    kv("elements", std::to_string(v.elements));
    if (v.values.present) {
        kv("values_fp16", json_bool(v.values_fp16 == 1));
        o += pad(k) + "\"values\": " + plane_json(v.values, k) + ",\n";
    } else {
        kv("code_bits", std::to_string(v.code_bits));
        o += pad(k) + "\"codes\": " + plane_json(v.codes, k) + ",\n";
        o += pad(k) + "\"scales\": " + plane_json(v.scales, k) + ",\n";
        kv("scales_fp16", json_bool(v.scales_fp16 == 1));
        if (v.offsets.present) {
            o += pad(k) + "\"offsets\": " + plane_json(v.offsets, k) + ",\n";
            kv("offsets_fp16", json_bool(v.offsets_fp16 == 1));
        }
    }
    kv("file", json_string(e.file), false);
    o += pad(ind) + "}";
    return o;
}

} // namespace

std::string pack_manifest_json(const PackManifest& m, std::string& err) {
    err.clear();
    const ExpertArenaOffsets off = m.geometry.offsets();
    std::string o = "{\n";
    o += " \"format\": \"strata-pack\",\n";
    o += " \"format_version\": 1,\n";
    o += " \"source\": {\n";
    o += "  \"shard1\": " + json_string(m.shard1) + ",\n";
    o += "  \"shard2\": " + json_string(m.shard2) + ",\n";
    o += "  \"shard1_sha256\": " + (m.shard1_sha256.empty() ? "null" : json_string(m.shard1_sha256)) + ",\n";
    o += "  \"shard2_sha256\": " + (m.shard2_sha256.empty() ? "null" : json_string(m.shard2_sha256)) + "\n";
    o += " },\n";
    o += " \"n_layers\": " + std::to_string(m.n_layers) + ",\n";
    o += " \"align\": " + std::to_string(m.align) + ",\n";

    // The IQ4 codebook is recorded because the pack's codes are indices into it: a reader that assumed a
    // different table would decode a plausible weight.
    o += " \"codebooks\": {\n";
    o += "  \"IQ4NL\": [\n";
    const float* table = iq4nl_table();
    for (int i = 0; i < 16; ++i) o += "   " + json_number((double)table[i]) + (i < 15 ? "," : "") + "\n";
    o += "  ]\n";
    o += " },\n";

    o += " \"tensors\": {";
    if (m.tensors.empty()) {
        o += "},\n";
    } else {
        o += "\n";
        for (std::size_t i = 0; i < m.tensors.size(); ++i)
            o += "  " + json_string(m.tensors[i].name) + ": " + entry_json(m.tensors[i], 2) +
                 (i + 1 < m.tensors.size() ? "," : "") + "\n";
        o += " },\n";
    }

    o += " \"experts\": {\n";
    o += "  \"blob_bytes\": " + std::to_string(m.geometry.blob_bytes()) + ",\n";
    o += "  \"offsets\": {\n";
    o += "   \"gate_up_codes\": " + std::to_string(off.gate_up_codes) + ",\n";
    o += "   \"down_codes\": " + std::to_string(off.down_codes) + ",\n";
    o += "   \"gate_up_scales\": " + std::to_string(off.gate_up_scales) + ",\n";
    o += "   \"down_scales\": " + std::to_string(off.down_scales) + "\n";
    o += "  },\n";
    o += "  \"layers\": [";
    if (m.expert_layers.empty()) {
        o += "],\n";
    } else {
        o += "\n";
        for (std::size_t i = 0; i < m.expert_layers.size(); ++i) {
            const PackManifestExpertLayer& L = m.expert_layers[i];
            o += "   {\n";
            o += "    \"layer\": " + std::to_string(L.layer) + ",\n";
            o += "    \"offset\": " + std::to_string(L.offset) + ",\n";
            o += "    \"role_offsets\": {\n";
            o += "     \"gate\": " + std::to_string(L.role_offsets[0]) + ",\n";
            o += "     \"up\": " + std::to_string(L.role_offsets[1]) + ",\n";
            o += "     \"down\": " + std::to_string(L.role_offsets[2]) + "\n";
            o += "    }\n";
            o += std::string("   }") + (i + 1 < m.expert_layers.size() ? "," : "") + "\n";
        }
        o += "  ],\n";
    }
    o += "  \"total_bytes\": " + std::to_string((uint64_t)m.n_layers * m.geometry.NE * m.geometry.blob_bytes()) +
         ",\n";
    o += "  \"scales_fp16\": true,\n";
    o += "  \"source_type\": \"Q2_0\",\n";
    o += "  \"form\": \"S2\"\n";
    o += " },\n";
    o += " \"n_experts_per_layer\": " + std::to_string(m.n_experts_per_layer) + ",\n";
    o += " \"files\": {\n";
    o += "  \"experts.bin\": " + std::to_string(m.files_experts) + ",\n";
    o += "  \"dense.bin\": " + std::to_string(m.files_dense) + ",\n";
    o += "  \"embd.bin\": " + std::to_string(m.files_embd) + "\n";
    o += " }\n";
    o += "}";   // no trailing newline: json.dumps writes none, and this file is compared byte for byte
    return o;
}

std::string pack_occupied(const std::string& dir) {
    // The order matters only for which name is reported; any one of them means the directory holds a pack.
    static const char* markers[] = {"manifest.json", "index.txt", "native_experts.txt"};
    for (const char* name : markers) {
        std::ifstream f(dir + "/" + name);
        if (f.good()) return name;
    }
    return {};
}

} // namespace strata
