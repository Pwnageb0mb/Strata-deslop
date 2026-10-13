// include/strata/artifact/pack_manifest.hpp - the pack's manifest.json.
//
// `tools/strata_pack.py`'s `build()` writes the manifest LAST, which is what makes it the pack's completion
// marker: `PACK_MARKERS` is (manifest.json, index.txt, native_experts.txt), so experts.bin without a manifest is
// a build that did not finish and is rebuilt - and a directory holding any of the three already holds a pack,
// which `build` refuses to overwrite without `--force`.
//
// The manifest is not read by the engine (that is why pack_index.py exists - the engine reads the flat index.txt
// instead of carrying a JSON parser).  It is the pack's description of itself, for the tools: what each tensor
// was, what form it was stored in, where its planes are, and what the pack was built from.
//
// The key order below is the reference's, and it is load-bearing only in the sense that the file is compared
// byte for byte - `json.dumps(indent=1)` lays it out this way, and two writers of one format have to agree.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "strata/artifact/canonical_entry.hpp"
#include "strata/artifact/expert_arena.hpp"
#include "strata/artifact/pack_index.hpp"

namespace strata {

/// One layer's entry in the manifest's expert table.
struct PackManifestExpertLayer {
    int64_t layer = 0;
    uint64_t offset = 0;              ///< into experts.bin
    uint64_t role_offsets[3] = {0, 0, 0};   ///< gate, up, down: their offsets in the SOURCE GGUF
};

struct PackManifest {
    std::string shard1;               ///< the source shard 1, as given
    std::string shard2;               ///< where shard 2 would be (it is NOT copied into the pack)
    int64_t n_layers = 0;
    uint64_t align = 64;
    std::vector<PackIndexEntry> tensors;    ///< in build order, each already carrying its `file`
    ExpertGeometry geometry;
    std::vector<PackManifestExpertLayer> expert_layers;
    uint64_t n_experts_per_layer = 0;
    std::string shard1_sha256;
    std::string shard2_sha256;        ///< empty writes null
    uint64_t files_experts = 0, files_dense = 0, files_embd = 0;
};

/// manifest.json, verbatim.  Returns "" with `err` set when an entry cannot be described.
std::string pack_manifest_json(const PackManifest& m, std::string& err);

/// The pack marker already in `dir` - "manifest.json", "index.txt" or "native_experts.txt" - or "" when the
/// directory is empty, absent, or holds no pack.  Checked BEFORE anything is opened or written: a build that
/// overwrote whatever pack was there did so silently, which is how the marker list came to exist.
std::string pack_occupied(const std::string& dir);

} // namespace strata
