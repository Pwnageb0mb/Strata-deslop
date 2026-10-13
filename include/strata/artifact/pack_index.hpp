// include/strata/artifact/pack_index.hpp - the pack's manifest entries flattened into the engine's index.txt.
//
// `tools/pack_index.py` writes the one line per tensor that the engine's loader reads.  Two things it decides
// THERE rather than in the engine, because the engine cannot recover them:
//
//   * WHICH ACTIVATION a weight wants.  The legacy formats (Q2_0, Q4_0, Q5_0, Q8_0, IQ4_NL) dot against Q8_0;
//     the K-quants and IQ4_XS against Q8_K.  NO COMBINATION OF THE S-FORM FIELDS TELLS THEM APART: Q5_0 and Q5_K
//     are both 8-bit with bias -16 (only has_offset differs), and IQ4_NL and IQ4_XS are both 4-bit with the IQ4NL
//     codebook and no offset.  They differ in nothing the form carries, so the column is emitted here.
//
//   * HOW THE ENGINE'S FORM DIFFERS FROM THE PACK'S, as an explicit `kind`: 0 VERBATIM, 1 BF16_IN_F32 (take the
//     high 16 bits - exact, because a bf16 value IS the top half of its f32 image), 2 F32, 3 F16_IN_F32 (a real
//     round-to-nearest-even conversion, not a truncation).  A tensor stored as BF16/F16 is promoted to 32 bits in
//     the pack because that is lossless and keeps the canonical decoder simple; a running engine does not have to
//     hold it that way, and re-rounding to 16 bits costs nothing because the value came FROM 16 bits.
//
// THE ONE THAT HID FOR A WHOLE ROUND: a scale plane may be 2 bytes in the pack and is ALWAYS 4 in the engine.
// 90 of the pack's 303 quantized tensors carry fp16 scales (that is what the source ggml block stores); the rest
// carry f32.  The engine assumed 4 everywhere, computed a scale plane twice the real size, and the loader refused
// - which is the only reason it was not a silent wrong answer.  So the plane sizes are EMITTED here rather than
// left for the loader to re-derive: two places computing one layout is two places to disagree.
//
// WHY THIS TAKES ENTRIES AND NOT A MANIFEST.  pack_index.py's docstring gives the reason the engine has no JSON
// parser: it would be a new, unaudited component whose failure mode is a wrong byte offset.  A C++ `strata-pack`
// that writes the manifest itself already holds these entries in memory, so it needs no parser either.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "strata/artifact/canonical_entry.hpp"

namespace strata {

/// One tensor as the index needs it: the entry itself, plus where it lives and what it was stored as.
struct PackIndexEntry {
    std::string name;
    std::string file;          ///< dense.bin, embd.bin or experts.bin
    std::string source_type;
    PackTensorEntry tensor;
};

/// The engine's index.txt for `entries`, or the refusal message in `err`.  Every check the reference makes is
/// made here too - a pack whose planes are not contiguous, whose `has_offset` disagrees with its planes, whose
/// element count is not a whole number of groups, or whose derived plane sizes disagree with its own recorded
/// ones is refused rather than written, because each of those decodes to a plausible weight.
std::string pack_index_text(const std::vector<PackIndexEntry>& entries, uint64_t align, std::string& err);

} // namespace strata
