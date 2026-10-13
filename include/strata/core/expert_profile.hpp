// include/strata/core/expert_profile.hpp - the expert profile: its file format, its ranking, and the routing
// trace it is built from.  NO GPU DEPENDENCY: this is file I/O and arithmetic, which is why it lives here and not
// in expert_cache.hpp.
//
// It used to live in expert_cache.hpp/.cpp, inside `strata_engine` - a library that is only built when a GPU
// backend is on.  The profile FORMAT has nothing to do with a GPU, and keeping it there meant a CPU-only tool
// (the profile builder, the eviction benchmark) could not link the one implementation of the format that already
// existed, and would have had to write a second.  The engine's `ExpertCache` includes this header and uses these
// functions; `src/core/expert_cache.cpp` keeps only the cache.
#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace strata::core {

// ---------------------------------------------------------------- the profile file ------------------------------

/// **THE STATIC RESIDENCY PLAN, READ BACK FROM THE FILE `tools/make_profile.py` WROTE.**
///
/// `profile.bin` is `STRP`, then `version, n_layers, n_expert, slots, n_ranked`, then `n_ranked` `(layer,
/// expert)` pairs ranked by descending routing frequency, then a `n_layers x n_expert` frequency table.
///
/// **WHY A PROFILE AND NOT THE COMPULSORY-MISS POLICY THIS ENGINE ALREADY HAD.**  Compulsory-miss fills with
/// the first `S` DISTINCT experts the run happens to route, which on this workload is whatever the prompt
/// touched first - it measured `h = 0.4864`.  A profile ranked by frequency over a whole trace is what the
/// plan's `0.6447` refers to, and the CPU's half of the drain is linear in `h`, so the gap is worth about
/// 4 ms of drain.  The point of reading it from a file is that a profile built on ONE prompt can be scored on
/// ANOTHER, which is the only way to know whether it transfers.
///
/// The frequency table is not returned: it is the profile's own working, and the engine measures `h` by
/// running rather than by re-deriving what the file claims.
bool read_expert_profile(const std::string& path, int64_t n_layers, int64_t n_expert,
                         std::vector<std::pair<int32_t, int32_t>>& ranked, int64_t& slots, std::string& err);

/// #477: writes `ranked` in tools/make_profile.py's format, byte for byte: `STRP`, `1, n_layers, n_expert,
/// n_ranked, n_ranked` (uint32), the pairs (uint16 layer, uint16 expert), then the n_layers x n_expert int32 table
/// of each pair's rank (-1: not ranked).  Atomically: a `<path>.tmp` beside it, renamed over `path` once complete,
/// so a reader (the next start) never sees half a file.  False with `err` when it could not be written.
bool write_expert_profile(const std::string& path, int64_t n_layers, int64_t n_expert,
                          const std::vector<std::pair<int32_t, int32_t>>& ranked, std::string& err);

/// #477 (--expert-profile-save): what the adaptive tier learned, as a profile ranking EVERY (layer, expert) pair -
/// the experts resident in VRAM now first (where the swaps left the cache), then the rest; within each, by `heat`
/// (the routing the adaptive tier counted since the start, descending), then by `prior` (the profile the engine
/// started from: an expert this run never routed keeps its old place), then by index.  A start from it begins
/// where this one ended; one with more slots adds the hottest of the rest, one with fewer keeps the hottest.
/// `resident` and `heat`: n_layers x n_expert entries (`heat` may be empty: no counts, the prior decides).
std::vector<std::pair<int32_t, int32_t>> rank_learned_profile(int64_t n_layers, int64_t n_expert,
                                                              const std::vector<uint8_t>& resident,
                                                              const std::vector<double>& heat,
                                                              const std::vector<std::pair<int32_t, int32_t>>& prior);

/// The slots a lookup on `layer` may use: layer `l` owns `[l*q, (l+1)*q)` with `q = slots / n_layers`, and the
/// LAST layer takes the remainder - R4.2g's per-layer admission, which keeps eviction inside a layer's own range.
/// `ExpertCache::layer_slot_range` is this function; the eviction benchmark simulates the same scope, and the two
/// are one rule in one place so they cannot drift.
void profile_slot_range(int64_t layer, int64_t slots, int64_t n_layers, int64_t& lo, int64_t& hi);

// ---------------------------------------------------------------- the routing trace ----------------------------

/// One routed (layer, expert) pair as a uint32: `(layer << 16) | expert`.  A trace is 4 bytes per lookup instead
/// of two objects, and for eviction policy the ORDER is the whole question, so the encoding is what makes a long
/// trace affordable to hold.
uint32_t trace_encode(int32_t layer, int32_t expert);
std::pair<int32_t, int32_t> trace_decode(uint32_t code);

/// The routed pairs of one `--dump-routing` file, IN ORDER, as `trace_encode`d codes.
///
/// The record is what `src/program/generate.cpp` writes: `<int32 layer><int32 k>`, then `k` int32 expert ids, then
/// `k` float32 weights.  A pair outside `0..n_layers-1` x `0..n_expert-1` is SKIPPED, as both reference tools skip
/// it.  A trailing partial record is not an error: the loop is bounded by the file, as the reference's is.
bool read_routing_trace(const std::string& path, int64_t n_layers, int64_t n_expert,
                        std::vector<uint32_t>& codes, std::string& err);

/// The routing frequencies of a trace - what `make_profile.read_trace` returns.  Derived from the ordered codes,
/// which is the consolidation: the reference walks the same records twice, once in `make_profile` for frequencies
/// and once in `bench_eviction` for order, and the two walks agree by construction.
std::vector<std::pair<std::pair<int32_t, int32_t>, uint64_t>> trace_frequencies(const std::vector<uint32_t>& codes);

/// `make_profile.rank_profile`: the base's order, then the traces' pairs most frequent first (a tie by
/// `(layer, expert)`, ascending), then every pair still missing, interleaved across the layers.
///
/// The base is only ever *appended to*, never reordered - so with a base that already ranks every pair, a plain
/// run cannot be moved by any trace.  `reorder` (or `no_base`) puts the traces' pairs first, which is what makes a
/// trace decide the top.  `counts`, when given, receives {from the base, from the traces, filled in}.
std::vector<std::pair<int32_t, int32_t>> rank_profile(
    const std::vector<std::pair<int32_t, int32_t>>& base_pairs,
    const std::vector<std::pair<std::pair<int32_t, int32_t>, uint64_t>>& trace_freq, int64_t n_layers,
    int64_t n_expert, bool no_base, bool reorder, int64_t counts[3]);

/// The profile's default dimensions, as the reference's module constants have them.
inline constexpr int64_t kProfileLayers = 48;
inline constexpr int64_t kProfileExperts = 512;

/// `make_profile_mix.rank`: an expert-cache profile from several traces, each counting the SAME.
///
/// `rank_profile` adds the traces' raw counts, so the longest trace decides the ranking.  A cache profile should
/// rank the experts by how often a user's tokens need them across the kinds of text that user writes, so here
/// every trace is first turned into a distribution (its counts / its total) and multiplied by its weight, and the
/// ranking is the weighted sum.  Weights are per trace, so a kind of text with more traces counts more - give a
/// group of N traces the weight 1/N each to make the group count once.
///
/// `per_trace_freq` is one entry per trace, in the order given, with `weights` alongside it.  A trace whose total
/// is 0 contributes nothing (the reference prints "empty trace, skipped").  The pairs no trace used are appended
/// in `rank_profile`'s fill order.
std::vector<std::pair<int32_t, int32_t>> rank_profile_mix(
    const std::vector<std::vector<std::pair<std::pair<int32_t, int32_t>, uint64_t>>>& per_trace_freq,
    const std::vector<double>& weights, int64_t n_layers, int64_t n_expert);

} // namespace strata::core
