// tests/artifact/canonical_entry_test.cpp - the plane layout and the manifest-driven decode.
//
// tools/gen_canonical_entry_golden.py records, for every source type, what tools/strata_pack.py's `tensor_entry`
// makes of one tensor: the bytes the pack holds, the manifest entry that describes them, and what `decode_entry`
// gets back from those two alone.  This compares all three.
//
// The layout is the part worth pinning: every plane starts on a 64-byte boundary and its offset is ABSOLUTE in the
// segment the tensor is written into, so a port that made the offsets relative to the tensor's own bytes would
// decode its own output perfectly and disagree with the reference on every tensor but the first.
//
//    canonical_entry_test <tests/artifact/canonical_entry_golden.txt>
#include "strata/artifact/canonical_entry.hpp"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

int failures = 0;

// The segment offset the golden was generated with (gen_canonical_entry_golden.py's SEG_OFFSET).
constexpr uint64_t kSegOffset = 1024;
constexpr uint64_t kSourceOffset = 4096;

std::vector<uint8_t> unhex(const std::string& h) {
    static const char* digits = "0123456789abcdef";
    std::vector<uint8_t> out;
    if (h.size() % 2) {
        std::fprintf(stderr, "bad hex (odd length): %s\n", h.c_str());
        ++failures;
        return out;
    }
    for (std::size_t i = 0; i + 1 < h.size(); i += 2) {
        const std::size_t hi = std::string(digits).find(h[i]);
        const std::size_t lo = std::string(digits).find(h[i + 1]);
        if (hi == std::string::npos || lo == std::string::npos) {
            std::fprintf(stderr, "bad hex: %s\n", h.c_str());
            ++failures;
            return out;
        }
        out.push_back((uint8_t)((hi << 4) | lo));
    }
    return out;
}

std::string hex_of(const uint8_t* p, std::size_t n) {
    static const char* digits = "0123456789abcdef";
    std::string s;
    s.reserve(n * 2);
    for (std::size_t i = 0; i < n; ++i) {
        s += digits[p[i] >> 4];
        s += digits[p[i] & 0xF];
    }
    return s;
}

std::string hex_of_f32(const std::vector<float>& v) {
    std::vector<uint8_t> raw(v.size() * 4);
    if (!v.empty()) std::memcpy(raw.data(), v.data(), raw.size());
    return hex_of(raw.data(), raw.size());
}

// The ggml type ids of the thirteen source types, so the test can ask the engine's dequantizer for the same
// tensor the golden used (the mapping registry keys on names; a caller building a pack has the id off the tensor).
uint32_t type_id_of(const std::string& name) {
    static const std::pair<const char*, uint32_t> table[] = {
        {"F32", 0},    {"F16", 1},    {"Q4_0", 2},   {"Q5_0", 6},   {"Q8_0", 8},    {"Q3_K", 11},
        {"Q4_K", 12},  {"Q5_K", 13},  {"Q6_K", 14},  {"IQ4_NL", 20}, {"IQ4_XS", 23}, {"BF16", 30},
        {"Q2_0", 42},
    };
    for (const auto& e : table)
        if (name == e.first) return e.second;
    return 0xFFFFFFFFu;
}

void check(const std::string& what, const std::string& got, const std::string& want) {
    if (got != want) {
        std::fprintf(stderr, "FAIL %s\n  got  %s\n  want %s\n", what.c_str(), got.c_str(), want.c_str());
        ++failures;
    }
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: canonical_entry_test <tests/artifact/canonical_entry_golden.txt>\n");
        return 2;
    }
    std::ifstream in(argv[1]);
    if (!in) {
        std::fprintf(stderr, "cannot read %s\n", argv[1]);
        return 2;
    }
    std::string line;
    int cases = 0;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        std::string what, tname, raw_hex, blob_hex, want_digest, values_hex, reference_hex;
        if (!(ss >> what >> tname >> raw_hex >> blob_hex >> want_digest >> values_hex >> reference_hex) ||
            what != "case") {
            std::fprintf(stderr, "malformed golden line: %s\n", line.c_str());
            ++failures;
            continue;
        }
        const std::vector<uint8_t> raw = unhex(raw_hex);
        const std::string label = tname + " (" + std::to_string(raw.size()) + " bytes)";

        // The shape the golden was generated with: one dimension, the tensor's element count.
        const strata::CanonicalMapping* m = strata::canonical_mapping(tname);
        if (!m) {
            std::fprintf(stderr, "FAIL %s: no mapping\n", label.c_str());
            ++failures;
            continue;
        }
        strata::TensorInfo meta;
        meta.name = "t." + tname;
        meta.shape.push_back((raw.size() / m->block_bytes) * m->block_elems);
        meta.type = 0;
        meta.offset = kSourceOffset;

        std::vector<uint8_t> blob;
        strata::PackTensorEntry entry;
        const std::string err =
            strata::pack_tensor_entry(meta.name, meta, tname, raw.data(), raw.size(), kSegOffset, blob, entry);
        if (!err.empty()) {
            std::fprintf(stderr, "FAIL %s: tensor_entry refused: %s\n", label.c_str(), err.c_str());
            ++failures;
            continue;
        }
        check(label + " blob", hex_of(blob.data(), blob.size()), blob_hex);
        check(label + " entry", entry.digest(), want_digest);

        // Decoding gets the SEGMENT, not the tensor's own bytes: the entry's offsets are absolute in it.
        std::vector<uint8_t> segment((std::size_t)kSegOffset + blob.size(), 0);
        std::memcpy(segment.data() + kSegOffset, blob.data(), blob.size());
        std::vector<float> values;
        std::string derr;
        if (!strata::pack_decode_entry(segment.data(), segment.size(), entry, values, derr)) {
            std::fprintf(stderr, "FAIL %s: decode_entry refused: %s\n", label.c_str(), derr.c_str());
            ++failures;
            continue;
        }
        check(label + " decoded", hex_of_f32(values), values_hex);

        // ---- verify: the engine's own dequantization of the SOURCE against the pack's decode, bit for bit.
        const uint32_t type_id = type_id_of(tname);
        std::vector<float> engine;
        if (!strata::dequantize_any(type_id, raw.data(), raw.size(), engine)) {
            std::fprintf(stderr, "FAIL %s: the engine has no dequantizer for type id %u\n", label.c_str(), type_id);
            ++failures;
            continue;
        }
        // The two independent sides have to agree before the comparison means anything: this is the engine's
        // transcription against the reference's own dequantization.
        check(label + " engine == reference", hex_of_f32(engine), reference_hex);

        const strata::PackVerifyResult ok =
            strata::pack_verify_tensor(type_id, raw.data(), raw.size(), segment.data(), segment.size(), entry);
        if (!ok.exact || ok.elements != engine.size()) {
            std::fprintf(stderr, "FAIL %s: verify did not pass: %s\n", label.c_str(), ok.detail.c_str());
            ++failures;
            continue;
        }
        // ...and a verifier that cannot fail is worth nothing: one flipped bit in the segment must be caught.
        // The last byte of whichever plane the form has - the S forms carry codes, the P forms values.
        const strata::PackPlane& pl = entry.values.present ? entry.values : entry.codes;
        std::vector<uint8_t> corrupt = segment;
        const std::size_t at = (std::size_t)(pl.offset + pl.bytes - 1);
        corrupt[at] ^= 0xFF;
        const strata::PackVerifyResult bad = strata::pack_verify_tensor(type_id, raw.data(), raw.size(),
                                                                       corrupt.data(), corrupt.size(), entry);
        if (bad.exact) {
            std::fprintf(stderr, "FAIL %s: a corrupted segment verified as exact\n", label.c_str());
            ++failures;
            continue;
        }
        ++cases;
    }

    // An entry that does not describe its planes must be refused, not guessed at - that is the point of decoding
    // from the manifest alone.
    {
        strata::PackTensorEntry e;
        e.form = "S8";
        e.code_bits = 8;
        e.group_elems = 32;
        e.elements = 32;
        e.scales_fp16 = 1;
        e.scales.present = true;
        e.scales.offset = 0;
        e.scales.bytes = 2;
        const std::vector<uint8_t> segment(64, 0);
        std::vector<float> values;
        std::string derr;
        if (strata::pack_decode_entry(segment.data(), segment.size(), e, values, derr)) {
            std::fprintf(stderr, "FAIL an entry with no codes plane was decoded instead of refused\n");
            ++failures;
        }
    }

    if (failures) {
        std::fprintf(stderr, "canonical_entry_test: FAILED (%d)\n", failures);
        return 1;
    }
    std::printf("canonical_entry_test: %d source types' plane layout, manifest-driven decode and verify match the "
                "reference, an incomplete entry is refused, and a corrupted pack is caught\n", cases);
    return 0;
}
