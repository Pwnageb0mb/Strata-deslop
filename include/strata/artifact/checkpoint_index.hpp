// include/strata/artifact/checkpoint_index.hpp - reading a HuggingFace checkpoint directory.
//
// The two model-preparation tools (tools/ple_fp8_pack.py, tools/embd_bf16_pack.py) both start the same way: read
// model.safetensors.index.json, take its weight_map, and look a tensor up in the shard that names it.  These are
// the pieces of that both need - the whole-file read, the suffix test, and the two JSON accessors - in one place
// rather than transcribed into each tool.
//
// The accessors are deliberately thin.  json.hpp stays a reader; these turn a missing key, or a count that is not
// a non-negative number, into an error that NAMES what was being read, which is the difference between
// "no such member" and "the weight_scale entry has no data_offsets".
#pragma once

#include <cstdint>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <string_view>

#include "strata/artifact/json.hpp"

namespace strata {
namespace artifact {

inline std::string read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

inline bool ends_with(const std::string& s, const std::string& suffix) {
    return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

/// the object member `key`, or a throw naming what was being read
inline const json::Value& field(const json::Value& obj, std::string_view key, const std::string& what) {
    const json::Value* v = obj.find(key);
    if (v == nullptr) throw std::runtime_error(what + " has no \"" + std::string(key) + "\"");
    return *v;
}

/// a non-negative integer, or a throw naming what was being read
inline std::uint64_t as_count(const json::Value& v, const std::string& what) {
    if (!v.is_num() || v.number < 0.0) throw std::runtime_error("expected a non-negative number for " + what);
    return (std::uint64_t) v.number;
}

} // namespace artifact
} // namespace strata
