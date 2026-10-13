// src/artifact/embd_bf16_pack_main.cpp - `strata-embd-bf16-pack`: the port of tools/embd_bf16_pack.py.
//
// The token embedding exactly as the checkpoint ships it, as a one-tensor GGUF the engine reads.  The payload is
// copied straight out of the safetensors file - nothing is decoded or rounded - so the output is a header plus
// the checkpoint's own bytes.
//
//   strata-embd-bf16-pack --model <checkpoint dir> --out <embd-bf16.gguf>
//
// The GGUF writing half is gguf_write.hpp, shared with the PLE tool: the two references are the same tool twice
// and the header is identical in both.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

#include "strata/artifact/checkpoint_index.hpp"
#include "strata/artifact/gguf_write.hpp"
#include "strata/artifact/json.hpp"
#include "strata/artifact/safetensors.hpp"

namespace {

constexpr std::uint64_t kChunk = 64ull << 20;   // the references' CHUNK

// the file read, the suffix test and the JSON accessors are shared with the PLE tool
using strata::artifact::as_count;
using strata::artifact::ends_with;
using strata::artifact::field;
using strata::artifact::read_file;

} // namespace

int main(int argc, char** argv) {
    std::string model, out;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--model" && i + 1 < argc) model = argv[++i];
        else if (a == "--out" && i + 1 < argc) out = argv[++i];
        else {
            std::fprintf(stderr, "usage: strata-embd-bf16-pack --model <dir> --out <gguf>\n");
            return 2;
        }
    }
    if (model.empty() || out.empty()) {
        std::fprintf(stderr, "usage: strata-embd-bf16-pack --model <dir> --out <gguf>\n");
        return 2;
    }

    try {
        const strata::json::Value index = strata::json::parse(read_file(model + "/model.safetensors.index.json"));
        const strata::json::Value& wmap = field(index, "weight_map", "the index");
        if (!wmap.is_obj()) throw std::runtime_error("weight_map is not an object");

        // the ONE text embed_tokens.weight: not the visual tower's, not the MTP layer's
        std::string name;
        int found = 0;
        for (const auto& m : wmap.members) {
            const std::string& n = m.first;
            if (ends_with(n, "embed_tokens.weight") && n.find("visual") == std::string::npos &&
                n.find("mtp") == std::string::npos) {
                name = n;
                ++found;
            }
        }
        if (found != 1) {
            throw std::runtime_error("expected one text embed_tokens.weight in " + model + ", found " +
                                     std::to_string(found));
        }
        const strata::json::Value& shard = field(wmap, name.c_str(), "weight_map");
        if (!shard.is_str()) throw std::runtime_error("weight_map[" + name + "] is not a string");
        const std::string shard_path = model + "/" + shard.text;

        const strata::artifact::SafetensorsFile sf = strata::artifact::read_safetensors_header(shard_path);
        const strata::json::Value& t = field(sf.header, name.c_str(), "the shard's header");
        const strata::json::Value& dtype = field(t, "dtype", name);
        const strata::json::Value& shape = field(t, "shape", name);
        const strata::json::Value& offs = field(t, "data_offsets", name);
        if (!shape.is_arr() || shape.items.size() != 2) throw std::runtime_error(name + ": shape is not 2-D");
        if (!offs.is_arr() || offs.items.size() != 2) throw std::runtime_error(name + ": data_offsets is not a pair");
        if (dtype.text != "BF16") throw std::runtime_error(name + " is " + dtype.text + ", not BF16");

        const std::uint64_t vocab = as_count(shape.items[0], name + " shape[0]");
        const std::uint64_t dim = as_count(shape.items[1], name + " shape[1]");
        const std::uint64_t payload = vocab * dim * 2;
        const std::uint64_t start = as_count(offs.items[0], name + " data_offsets[0]");
        if (as_count(offs.items[1], name + " data_offsets[1]") - start != payload) {
            throw std::runtime_error(name + ": data size does not match its shape");
        }
        std::printf("%s: %llu x %llu BF16, %.2f GB\n", name.c_str(), (unsigned long long) vocab,
                    (unsigned long long) dim, (double) payload / 1e9);

        const std::string kvs = strata::artifact::kv_string("general.architecture", "strata-embd") +
                                strata::artifact::kv_string("general.name", "token embedding, BF16 as shipped") +
                                strata::artifact::kv_string("strata.embd.source",
                                                            std::filesystem::path(model).filename().string());
        std::string head = strata::artifact::gguf_head(1, 3) + kvs;
        const std::uint64_t dims[2] = {dim, vocab};   // GGUF order: the fastest-varying dimension first
        head += strata::artifact::gguf_tensor_info("token_embd.weight", dims, 2, strata::artifact::kGgmlTypeBf16);
        strata::artifact::gguf_align(head);

        const std::filesystem::path out_path(out);
        if (out_path.has_parent_path()) std::filesystem::create_directories(out_path.parent_path());
        const std::string tmp = out + ".part";
        {
            std::ofstream w(tmp, std::ios::binary);
            if (!w) throw std::runtime_error("cannot write " + tmp);
            w.write(head.data(), (std::streamsize) head.size());

            std::ifstream src(shard_path, std::ios::binary);
            if (!src) throw std::runtime_error("cannot open " + shard_path);
            src.seekg((std::streamoff) (sf.base + start));
            std::vector<char> buf((std::size_t) std::min<std::uint64_t>(kChunk, payload));
            std::uint64_t left = payload;
            while (left > 0) {
                const std::size_t want = (std::size_t) std::min<std::uint64_t>(kChunk, left);
                src.read(buf.data(), (std::streamsize) want);
                if ((std::size_t) src.gcount() != want) throw std::runtime_error("short read in " + shard_path);
                w.write(buf.data(), (std::streamsize) want);
                left -= want;
            }
            w.close();
            if (!w) throw std::runtime_error("cannot write " + tmp);
        }
        if ((std::uint64_t) std::filesystem::file_size(tmp) != head.size() + payload) {
            throw std::runtime_error("size check failed");
        }
        std::filesystem::rename(tmp, out);
        std::printf("wrote %s (header %zu B)\n", out.c_str(), head.size());
    } catch (const std::exception& e) {
        std::fprintf(stderr, "strata-embd-bf16-pack: %s\n", e.what());
        return 1;
    }
    return 0;
}
