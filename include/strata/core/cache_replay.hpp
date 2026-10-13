// include/strata/core/cache_replay.hpp - replaying a routing trace through the engine's OWN adaptive policy.
//
// `ExpertCache::admit` never evicts; the adaptive tier that swaps experts is `adapt()` inside
// `src/program/generate.cpp`, and it runs against the live cache, the exchange buffers and the device.  This is
// the same policy replayed offline, so a parameter set can be measured on a trace without a GPU.  The reference
// is `tools/replay_cache_policy.py`.
//
// **IT DOES NOT REUSE THE ENGINE'S `adapt()`, AND THAT IS A FINDING, NOT AN OVERSIGHT.**  The engine's copy is a
// lambda closed over the live cache and sorts its candidate and victim lists with `std::sort`; the reference sorts
// with a STABLE order (`np.argsort(..., kind="stable")`) and so does this.  When two experts of a layer have equal
// usage - which a trace with repeated routing produces constantly - the two pick different victims, so the model's
// rate and the engine's can differ.  This port follows the reference, because the reference's own output is what
// it is accepted against; the divergence is recorded in docs/CXX_PORT.md rather than papered over by changing
// either side.
//
// The policy: `usage += 1` per routed entry, a round every `every` windows, per layer the best non-resident
// (`usage >= mincand`) against the worst resident (`candidate >= victim + margin`, stopping at the first pair that
// does not clear it), the best `swaps` over all layers by gain, then `usage *= decay`.
#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace strata::core {

struct ReplayPolicy {
    int64_t every = 4;
    double decay = 0.7;
    int64_t swaps = 96;
    double margin = 1.5;
    double mincand = 2.0;
    double mingain = 0.0;   ///< the reference filters on this after sorting; the engine has no equivalent
};

/// One layer's group of records inside a window: the records that routed this layer before the layer changed.
struct ReplayGroup {
    int32_t layer = 0;
    int64_t n_records = 0;          ///< `tokens/window` is the mean of the FIRST group's record count per window
    std::vector<int32_t> ids;       ///< every record's ids, in order (the reference flattens the group anyway)
};

struct ReplayWindow {
    std::vector<ReplayGroup> groups;
};

struct ReplayStats {
    double share = 0.0;
    int64_t hits = 0, entries = 0;
    int64_t swaps = 0, windows = 0;
    int64_t miss_ent = 0, miss_dist = 0;
};

/// The trace, grouped into windows, as the reference's `load_trace` groups it.
///
/// It reads the file as int32 and takes `k` from the FIRST record's second word, then treats the file as records
/// of a fixed `2 + 2k` words - it does NOT read each record's own `k`, and it does NOT skip out-of-range ids the
/// way `make_profile`/`bench_eviction` do.  A window is a run of groups whose layer RISES; consecutive records
/// with one layer are one group; only windows with one group per layer are kept (`len(w) == n_layers`), so a short
/// trace loads as no windows at all.  An id outside `0..n_expert-1` is refused rather than indexed with, which is
/// what the reference would do (an IndexError).
bool read_replay_trace(const std::string& path, int64_t n_layers, int64_t n_expert,
                       std::vector<ReplayWindow>& windows, int64_t& k, std::string& err);

/// One adaptation round: returns how many swaps it made.  `res` and `usage` are `n_layers x n_expert`.
int64_t replay_adapt_round(std::vector<uint8_t>& res, std::vector<double>& usage, int64_t n_layers,
                           int64_t n_expert, const ReplayPolicy& pol);

/// The replay: `init` seats the cache first (the profile's top `slots`), then the windows run.
ReplayStats replay_simulate(const std::vector<ReplayWindow>& windows,
                            const std::vector<std::pair<int32_t, int32_t>>& init, int64_t n_layers,
                            int64_t n_expert, const ReplayPolicy& pol);

/// Belady-free upper bound the reference reports: the share of all routed entries served by the `slots` most
/// routed experts of THIS trace.
double replay_static_oracle(const std::vector<ReplayWindow>& windows, int64_t n_layers, int64_t n_expert,
                            int64_t slots);

/// The four lines the reference prints, verbatim.  It lives here rather than in the CLI because the printed
/// report IS the acceptance: the test compares against the reference's own stdout, and a second copy of the
/// formatting in the test would be a second thing to get wrong.
std::string replay_report(const std::vector<ReplayWindow>& windows,
                          const std::vector<std::pair<int32_t, int32_t>>& init, int64_t slots, int64_t k,
                          int64_t n_layers, int64_t n_expert, const ReplayPolicy& pol);

/// A double the way Python's repr writes it: shortest round-trip, always with a fraction or an exponent.  The
/// reference prints `np.mean(...)` of a float64, and numpy's repr is Python's repr.
std::string replay_py_float(double v);

/// `printf("%.*f")`, which is what the reference's `%.4f` / `%.2f` / `%.1f` are.
std::string replay_fixed(double v, int places);

} // namespace strata::core
