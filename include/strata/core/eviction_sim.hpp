// include/strata/core/eviction_sim.hpp - replaying a routing trace against candidate ExpertCache eviction policies.
//
// `ExpertCache::admit` never evicts: the eviction policy is a MEASURED question (R4.1's LFU-decay vs an LRU
// sweep), and a placeholder would set the hit rate everything downstream is sized against.  This is the harness
// for that measurement without a GPU - a policy's hit rate depends only on the ORDER the run routed (layer,
// expert) pairs, and a `--dump-routing` trace records exactly that order.
//
// The reference is `tools/bench_eviction.py`, and the port is accepted against its own report and CSV.
//
// The policies:
//
//   fill-only   the engine's current admission: compulsory miss, never evict.  The baseline a policy has to beat,
//               and the answer for any run whose trace never exceeds the slot count.
//   lru         evict the least-recently-used resident.
//   lfu-decay   frequency counts halved every `decay_window` lookups, so a hot-once expert fades and the cache
//               can follow a topic shift; evict the lowest count, the oldest on a tie.
//   oracle      Belady's MIN: evict the resident whose next use is furthest in the future.  Not implementable
//               online - it reads the trace ahead - but it is the upper bound any policy can reach on this trace.
//
// The scope is the cache's: one shared pool (the engine's default), or `--per-layer` where layer `l` owns
// `[l*q, (l+1)*q)` and eviction stays inside the layer's own range.  That rule is `profile_slot_range`, the same
// function `ExpertCache::layer_slot_range` calls.
#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace strata::core {

/// One (slots, policy) replay's result: the overall counts and the per-window split.
struct EvictionRow {
    int64_t slots = 0;
    std::string policy;
    int64_t hits = 0;
    int64_t lookups = 0;
    int64_t evictions = 0;
    std::vector<int64_t> w_hits;
    std::vector<int64_t> w_tot;
};

/// The policy names, in the reference's order (which is also the order its `--policies` default lists them).
const std::vector<std::string>& eviction_policy_names();

/// "" or the reason this (slots, policy) pair cannot be replayed: a scope that leaves a layer no slots at all,
/// for a policy that evicts.
///
/// Reachable as `--per-layer --slots < n_layers`, where `q = slots / n_layers` is 0 and every layer but the last
/// has capacity 0.  **The reference CRASHES there** - its `Lru.lookup` tests `len(od) >= capacity` (true for an
/// empty cache at capacity 0) and then pops an empty OrderedDict, `KeyError: 'dictionary is empty'` - so there is
/// no reference output for this configuration to be a port of.  Refusing it is the honest answer; producing
/// numbers where the reference dies would be inventing behaviour it does not define.
std::string eviction_config_error(int64_t n_layers, int64_t slots, bool per_layer, const std::string& policy);

/// One replay per (slots, policy), in the order given.  `prefill` seats the profile's ranked pairs first, which
/// is how the engine actually starts with `--expert-profile`; without it the run starts cold.
std::vector<EvictionRow> eviction_run(const std::vector<uint32_t>& codes, int64_t n_layers, int64_t n_expert,
                                      const std::vector<int64_t>& slots_list, bool per_layer,
                                      const std::vector<std::string>& policies, int64_t windows = 1,
                                      int64_t decay_window = 65536,
                                      const std::vector<std::pair<int32_t, int32_t>>* prefill = nullptr);

/// The report the reference prints, verbatim (including its leading blank line per slot count).
std::string eviction_report(const std::vector<EvictionRow>& rows, int64_t windows);

/// The reference's `--csv` file, verbatim.  A per-window row carries no eviction count, which is why its line
/// ends in a comma.
std::string eviction_csv(const std::vector<EvictionRow>& rows, const std::string& scope);

} // namespace strata::core
