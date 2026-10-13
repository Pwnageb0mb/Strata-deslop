// src/artifact/ple_fp8_pack_main.cpp - `strata-ple-fp8-pack`: the port of tools/ple_fp8_pack.py.
//
// The PLE (per-layer embedding) n-gram table, as an FP8 E4M3 GGUF the engine reads.  The shards are concatenated
// in shard order and copied straight through - nothing is decoded or rounded - so the output is a header plus
// the checkpoint's own bytes.
//
//   strata-ple-fp8-pack --model <checkpoint dir> --out <ple-fp8.gguf>
//
// The GGUF writing half is gguf_write.hpp, shared with the embedding tool.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "strata/artifact/checkpoint_index.hpp"
#include "strata/artifact/gguf_write.hpp"
#include "strata/artifact/json.hpp"
#include "strata/artifact/safetensors.hpp"

namespace {

constexpr std::uint64_t kChunk = 64ull << 20;   // the reference's CHUNK

// the file read, the suffix test and the JSON accessors are shared with the embedding tool
using strata::artifact::as_count;
using strata::artifact::ends_with;
using strata::artifact::field;
using strata::artifact::read_file;

/// a tensor's dtype, 2-D shape and data offsets, checked to be what the caller expects
struct Entry {
    std::uint64_t a = 0, b = 0;
    std::uint64_t start = 0, size = 0;
    std::string dtype;
};

Entry entry_of(const strata::artifact::SafetensorsFile& sf, const std::string& name, const char* want_dtype,
               bool allow_1d) {
    const strata::json::Value& t = field(sf.header, name, "the shard's header");
    const strata::json::Value& dtype = field(t, "dtype", name);
    const strata::json::Value& shape = field(t, "shape", name);
    const strata::json::Value& offs = field(t, "data_offsets", name);
    if (!offs.is_arr() || offs.items.size() != 2) throw std::runtime_error(name + ": data_offsets is not a pair");
    Entry e;
    e.dtype = dtype.text;
    e.start = as_count(offs.items[0], name + " data_offsets[0]");
    e.size = as_count(offs.items[1], name + " data_offsets[1]") - e.start;
    if (dtype.text != want_dtype) throw std::runtime_error(name + " is " + dtype.text + ", not " + want_dtype);
    if (allow_1d && shape.is_arr() && shape.items.size() <= 1) {
        e.a = shape.items.empty() ? 1 : as_count(shape.items[0], name + " shape[0]");
        e.b = 1;
        return e;
    }
    if (!shape.is_arr() || shape.items.size() != 2) throw std::runtime_error(name + ": shape is not 2-D");
    e.a = as_count(shape.items[0], name + " shape[0]");
    e.b = as_count(shape.items[1], name + " shape[1]");
    return e;
}

} // namespace

int main(int argc, char** argv) {
    std::string model, out;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--model" && i + 1 < argc) model = argv[++i];
        else if (a == "--out" && i + 1 < argc) out = argv[++i];
        else {
            std::fprintf(stderr, "usage: strata-ple-fp8-pack --model <dir> --out <gguf>\n");
            return 2;
        }
    }
    if (model.empty() || out.empty()) {
        std::fprintf(stderr, "usage: strata-ple-fp8-pack --model <dir> --out <gguf>\n");
        return 2;
    }

    try {
        const strata::json::Value index = strata::json::parse(read_file(model + "/model.safetensors.index.json"));
        const strata::json::Value& wmap = field(index, "weight_map", "the index");
        if (!wmap.is_obj()) throw std::runtime_error("weight_map is not an object");

        std::map<long, std::string> shards;
        std::string scale_name;
        for (const auto& m : wmap.members) {
            const std::string& n = m.first;
            const std::size_t at = n.find(".ngram_embedding.shard_");
            if (at != std::string::npos && ends_with(n, ".weight")) {
                const std::size_t from = at + std::strlen(".ngram_embedding.shard_");
                const std::size_t dot = n.find('.', from);
                if (dot == std::string::npos) throw std::runtime_error(n + ": the shard number is unreadable");
                shards[std::stol(n.substr(from, dot - from))] = n;
            } else if (ends_with(n, ".ngram_embedding.weight_scale")) {
                scale_name = n;
            }
        }
        if (shards.empty() || scale_name.empty()) {
            throw std::runtime_error("no ngram_embedding shards / weight_scale in " + model);
        }
        for (std::size_t k = 0; k < shards.size(); ++k) {
            if (shards.count((long) k) == 0) {
                throw std::runtime_error("the n-gram shards are not numbered 0.." +
                                         std::to_string(shards.size() - 1) + " without gaps");
            }
        }

        // the scale: a bf16 scalar, widened by shifting it into the top half of an f32
        const strata::json::Value& scale_shard = field(wmap, scale_name.c_str(), "weight_map");
        const std::string scale_path = model + "/" + scale_shard.text;
        const strata::artifact::SafetensorsFile ssf = strata::artifact::read_safetensors_header(scale_path);
        const Entry se = entry_of(ssf, scale_name, "BF16", true);
        if (se.a != 1 || se.size != 2) throw std::runtime_error("unexpected weight_scale: " + se.dtype);
        float scale = 0.0f;
        {
            std::ifstream f(scale_path, std::ios::binary);
            if (!f) throw std::runtime_error("cannot open " + scale_path);
            f.seekg((std::streamoff) (ssf.base + se.start));
            unsigned char two[2] = {0, 0};
            f.read(reinterpret_cast<char*>(two), 2);
            if (f.gcount() != 2) throw std::runtime_error("short read in " + scale_path);
            const std::uint32_t bits = (((std::uint32_t) two[1] << 8) | two[0]) << 16;
            std::memcpy(&scale, &bits, 4);
        }

        struct Part {
            std::string path;
            std::uint64_t offset = 0, size = 0;
        };
        std::vector<Part> parts;
        std::uint64_t rows = 0, dim = 0;
        bool have_dim = false;
        for (const auto& kv : shards) {
            const std::string& name = kv.second;
            const strata::json::Value& shard = field(wmap, name.c_str(), "weight_map");
            const std::string path = model + "/" + shard.text;
            const strata::artifact::SafetensorsFile sf = strata::artifact::read_safetensors_header(path);
            const Entry e = entry_of(sf, name, "F8_E4M3", false);
            if (!have_dim) {
                dim = e.b;
                have_dim = true;
            }
            if (e.b != dim || e.size != e.a * dim) throw std::runtime_error(name + ": inconsistent shape");
            parts.push_back({path, sf.base + e.start, e.a * dim});
            rows += e.a;
        }
        std::printf("%d shards, %d rows x %d, scale %.9g, %.2f GB\n", (int) parts.size(), (int) rows, (int) dim,
                    (double) scale, (double) (rows * dim) / 1e9);

        const std::string kvs =
            strata::artifact::kv_string("general.architecture", "strata-ple") +
            strata::artifact::kv_string("general.name", "PLE n-gram table, FP8 E4M3 as shipped") +
            strata::artifact::kv_string("strata.ple.format", "f8_e4m3") +
            strata::artifact::kv_f32("strata.ple.scale", scale) +
            strata::artifact::kv_string("strata.ple.source", std::filesystem::path(model).filename().string());
        std::string head = strata::artifact::gguf_head(1, 5) + kvs;
        const std::uint64_t dims[2] = {dim, rows};
        head += strata::artifact::gguf_tensor_info("per_layer_token_embd.weight", dims, 2,
                                                   strata::artifact::kGgmlTypeI8);
        strata::artifact::gguf_align(head);

        const std::filesystem::path out_path(out);
        if (out_path.has_parent_path()) std::filesystem::create_directories(out_path.parent_path());
        const std::string tmp = out + ".part";
        std::uint64_t written = 0;
        {
            std::ofstream w(tmp, std::ios::binary);
            if (!w) throw std::runtime_error("cannot write " + tmp);
            w.write(head.data(), (std::streamsize) head.size());
            std::vector<char> buf((std::size_t) kChunk);
            for (std::size_t i = 0; i < parts.size(); ++i) {
                std::ifstream src(parts[i].path, std::ios::binary);
                if (!src) throw std::runtime_error("cannot open " + parts[i].path);
                src.seekg((std::streamoff) parts[i].offset);
                std::uint64_t left = parts[i].size;
                while (left > 0) {
                    const std::size_t want = (std::size_t) std::min<std::uint64_t>(kChunk, left);
                    src.read(buf.data(), (std::streamsize) want);
                    if ((std::size_t) src.gcount() != want) throw std::runtime_error("short read in " + parts[i].path);
                    w.write(buf.data(), (std::streamsize) want);
                    left -= want;
                }
                written += parts[i].size;
                if (i % 16 == 15 || i == parts.size() - 1) {
                    std::printf("  %3d/%d shards, %.1f GB\n", (int) (i + 1), (int) parts.size(),
                                (double) written / 1e9);
                }
            }
            w.close();
            if (!w) throw std::runtime_error("cannot write " + tmp);
        }
        if ((std::uint64_t) std::filesystem::file_size(tmp) != head.size() + rows * dim) {
            throw std::runtime_error("size check failed");
        }
        std::filesystem::rename(tmp, out);
        std::printf("wrote %s (%d rows x %d FP8, header %zu B)\n", out.c_str(), (int) rows, (int) dim, head.size());
    } catch (const std::exception& e) {
        std::fprintf(stderr, "strata-ple-fp8-pack: %s\n", e.what());
        return 1;
    }
    return 0;
}
