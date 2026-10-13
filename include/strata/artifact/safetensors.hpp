// include/strata/artifact/safetensors.hpp - the safetensors header, as the model-preparation tools read it.
//
// The format is an 8-byte little-endian length, then that many bytes of JSON; the tensor data starts right
// after.  So a tensor's bytes live at `base + data_offsets[0]`, where `base` is 8 + the length - which is why
// this returns both and not just the parsed document.
//
// Header only, no data: the tools copy the payload straight through in CHUNK-sized pieces rather than reading a
// 100 GB tensor into memory, and that is the point of them.
#pragma once

#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>

#include "strata/artifact/json.hpp"

namespace strata {
namespace artifact {

struct SafetensorsFile {
    json::Value header;    ///< tensor name -> {dtype, shape, data_offsets}
    std::uint64_t base = 0;   ///< where the data section starts
};

constexpr std::uint64_t kMaxHeaderBytes = 100ull * 1024 * 1024;   // the safetensors spec's limit

inline SafetensorsFile read_safetensors_header(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);

    unsigned char len[8] = {0};
    f.read(reinterpret_cast<char*>(len), 8);
    if (f.gcount() != 8) throw std::runtime_error(path + ": too short to be safetensors");
    std::uint64_t n = 0;
    for (int i = 7; i >= 0; --i) n = (n << 8) | len[i];     // the length is little-endian

    if (n > kMaxHeaderBytes) throw std::runtime_error(path + ": the header length is over 100 MiB");

    std::string text((std::size_t) n, '\0');
    f.read(text.data(), (std::streamsize) n);
    if ((std::uint64_t) f.gcount() != n) throw std::runtime_error(path + ": the header is truncated");

    SafetensorsFile out;
    out.header = json::parse(text);
    out.base = 8 + n;
    return out;
}

} // namespace artifact
} // namespace strata
