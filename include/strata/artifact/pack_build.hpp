// include/strata/artifact/pack_build.hpp - building a pack: the orchestration over the pieces.
//
// `tools/strata_pack.py`'s `build()` assembles a pack from a model: the expert arena, the two tensor segments
// (dense.bin and embd.bin), the manifest that describes them, and the hashes of the source.  This is that, over
// the ported pieces - the mappings, the plane layout, the arena - which are each already verified against the
// reference byte for byte.
//
// TWO THINGS ABOUT THE ORDER, both from the reference and both load-bearing:
//
//   * The marker check comes FIRST, before anything is opened or written.  A build used to overwrite
//     experts.bin, dense.bin, embd.bin and manifest.json of whatever pack was in the directory, saying nothing.
//
//   * manifest.json is written LAST, which is what makes it the pack's completion marker: a build that stops
//     part-way leaves experts.bin without a manifest, which the next build (or setup) makes again - never the old
//     manifest over partly new data.
//
// The engine's index.txt is a SEPARATE step in the reference (setup.py runs pack_index.py after the build), and
// stays one here: this fills `PackBuildResult::tensors`, and `pack_index_text` turns that into the file.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "strata/artifact/expert_arena.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include "strata/artifact/pack_index.hpp"
#include "strata/artifact/pack_manifest.hpp"

namespace strata {

struct PackBuildOptions {
    ExpertGeometry geometry;      ///< the model's geometry: 2,560/640/512 in production, scaled in tests
    int64_t n_layers = 48;
    uint64_t align = 64;
    bool skip_hash = false;
    bool force = false;
    std::string shard2;           ///< where shard 2 would be: recorded in the manifest, never copied
};

struct PackBuildResult {
    PackManifest manifest;
    std::vector<PackIndexEntry> tensors;   ///< in build order, each with its file - what pack_index_text needs
};

/// Writes experts.bin, dense.bin, embd.bin and manifest.json into `out_dir`.  Returns "" on success, or the
/// refusal message naming what was refused and why.
std::string pack_build(const GgufModel& model, const std::string& out_dir, const PackBuildOptions& opts,
                       PackBuildResult& out);

/// The sha256 of a file, read in chunks - the source shards are 60 GB, so this cannot be one buffer.  Empty
/// string when the file cannot be read, which the manifest writes as null (the reference does the same for a
/// shard that is not there).
std::string sha256_file(const std::string& path);

} // namespace strata
