// include/strata/artifact/gguf_write.hpp - the GGUF header the model-preparation tools hand-roll.
//
// tools/ple_fp8_pack.py and tools/embd_bf16_pack.py do not use gguf-py.  They build the file with struct.pack,
// and they build it the same way: the same gguf_string/kv_string, the same GGUF_TYPE_STRING = 8, the same
// ALIGN = 32, one tensor out.  This is that header once, for both callers - the alternative is two
// transcriptions of the same six lines, which is how two tools drift apart.
//
// The tree READS GGUF in gguf_reader.hpp; this is the writing half, and it is deliberately this small because
// that is all the references are.  Every integer is written little-endian by hand rather than by memcpy, so
// the bytes do not depend on the host's byte order.
#pragma once

#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>

namespace strata {
namespace artifact {

inline constexpr std::uint32_t kGgufTypeString = 8;   ///< GGUF_TYPE_STRING
inline constexpr std::uint32_t kGgufTypeF32 = 6;      ///< GGUF_TYPE_F32
inline constexpr std::size_t kGgufAlign = 32;         ///< the tools' ALIGN

inline void put_u32(std::string& out, std::uint32_t v) {
    for (int i = 0; i < 4; ++i) out += (char) ((v >> (8 * i)) & 0xFF);
}

inline void put_u64(std::string& out, std::uint64_t v) {
    for (int i = 0; i < 8; ++i) out += (char) ((v >> (8 * i)) & 0xFF);
}

inline void put_f32(std::string& out, float v) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &v, 4);
    put_u32(out, bits);
}

/// a GGUF string: a u64 length, then the bytes
inline std::string gguf_string(std::string_view s) {
    std::string out;
    put_u64(out, s.size());
    out.append(s);
    return out;
}

inline std::string kv_string(std::string_view key, std::string_view val) {
    std::string out = gguf_string(key);
    put_u32(out, kGgufTypeString);
    out += gguf_string(val);
    return out;
}

inline std::string kv_f32(std::string_view key, float val) {
    std::string out = gguf_string(key);
    put_u32(out, kGgufTypeF32);
    put_f32(out, val);
    return out;
}

/// the magic and the two counts.  The order is GGUF's, not the caller's: version, then n_tensors, then n_kv -
/// the reference packs `3, 1, len(kvs)`.
inline std::string gguf_head(std::uint64_t n_tensors, std::uint64_t n_kv) {
    std::string out = "GGUF";
    put_u32(out, 3);
    put_u64(out, n_tensors);
    put_u64(out, n_kv);
    return out;
}

/// the ggml type ids these tools write.  Part of the container's tensor-info, so it lives here; the
/// dequantizers for the types the ENGINE reads are in dequant_dispatch.hpp.
inline constexpr std::uint32_t kGgmlTypeF32 = 0;
inline constexpr std::uint32_t kGgmlTypeI8 = 24;
inline constexpr std::uint32_t kGgmlTypeBf16 = 30;

/// one tensor's info: the name, the dimension count, the dimensions, the ggml type, and the tensor's offset
/// within the data section.  A multi-tensor container passes a running offset; the single-tensor tools leave it 0.
inline std::string gguf_tensor_info(std::string_view name, const std::uint64_t* dims, std::size_t n_dims,
                                    std::uint32_t type, std::uint64_t offset = 0) {
    std::string out = gguf_string(name);
    put_u32(out, (std::uint32_t) n_dims);
    for (std::size_t i = 0; i < n_dims; ++i) put_u64(out, dims[i]);
    put_u32(out, type);
    put_u64(out, offset);
    return out;
}

/// the padding `n` bytes need to reach the tools' ALIGN
inline std::size_t gguf_pad(std::uint64_t n) { return (kGgufAlign - (std::size_t) (n % kGgufAlign)) % kGgufAlign; }

/// pad to the tools' ALIGN
inline void gguf_align(std::string& out) {
    out.append((kGgufAlign - out.size() % kGgufAlign) % kGgufAlign, '\0');
}

} // namespace artifact
} // namespace strata
