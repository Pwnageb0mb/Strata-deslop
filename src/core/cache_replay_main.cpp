// src/core/cache_replay_main.cpp - `strata-cache-replay`: the C++ port of tools/replay_cache_policy.py.
//
// Offline replay of a `--dump-routing` trace through the expert cache's adaptive policy, so a parameter set can be
// measured on a trace without a GPU.  The policy itself is `cache_replay.hpp`'s; this is the argument handling and
// the refusal of what the reference crashes on.
//
//    strata-cache-replay TRACE PROFILE SLOTS [--every N] [--decay F] [--swaps N] [--margin F] [--mincand F]
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "strata/core/cache_replay.hpp"
#include "strata/core/expert_profile.hpp"

namespace {

int usage() {
    std::fprintf(stderr,
                 "usage: strata-cache-replay TRACE PROFILE SLOTS [--every N] [--decay F] [--swaps N]\n"
                 "                                             [--margin F] [--mincand F]\n"
                 "\n"
                 "  TRACE       a --dump-routing file (one record per token and layer)\n"
                 "  PROFILE     the profile the run starts from; its top SLOTS pairs are seated first\n"
                 "  SLOTS       how many experts the cache holds\n"
                 "  --every N   windows between adaptation rounds (default 4)\n"
                 "  --decay F   usage counts are multiplied by this after each round (default 0.7)\n"
                 "  --swaps N   most swaps per round (default 96)\n"
                 "  --margin F  a candidate must beat a victim by this (default 1.5)\n"
                 "  --mincand F a non-resident expert needs this usage to be a candidate (default 2.0)\n");
    return 2;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 4) return usage();
    const std::string trace_path = argv[1], profile_path = argv[2];
    const int64_t slots = std::atoll(argv[3]);
    if (slots < 0) {
        std::fprintf(stderr, "strata-cache-replay: SLOTS must not be negative\n");
        return 2;
    }

    strata::core::ReplayPolicy pol;
    for (int i = 4; i + 1 < argc; ++i) {
        const char* a = argv[i];
        if (std::strcmp(a, "--every") == 0) pol.every = (int64_t) std::atoll(argv[i + 1]);
        else if (std::strcmp(a, "--decay") == 0) pol.decay = std::atof(argv[i + 1]);
        else if (std::strcmp(a, "--swaps") == 0) pol.swaps = (int64_t) std::atoll(argv[i + 1]);
        else if (std::strcmp(a, "--margin") == 0) pol.margin = std::atof(argv[i + 1]);
        else if (std::strcmp(a, "--mincand") == 0) pol.mincand = std::atof(argv[i + 1]);
    }

    const int64_t n_layers = strata::core::kProfileLayers;
    const int64_t n_expert = strata::core::kProfileExperts;

    std::vector<strata::core::ReplayWindow> windows;
    int64_t k = 0;
    std::string err;
    if (!strata::core::read_replay_trace(trace_path, n_layers, n_expert, windows, k, err)) {
        std::fprintf(stderr, "strata-cache-replay: %s\n", err.c_str());
        return 1;
    }
    if (windows.empty()) {
        // the reference divides by `windows` in its last line and reads `w[0]` in its first, so an empty load is a
        // crash there: refusing is the honest answer, not printing a rate the reference cannot produce
        std::fprintf(stderr, "strata-cache-replay: %s loaded no complete window (a window is one group per layer, "
                             "with the layer rising; the reference divides by it)\n", trace_path.c_str());
        return 1;
    }

    std::vector<std::pair<int32_t, int32_t>> ranked;
    int64_t profile_slots = 0;
    if (!strata::core::read_expert_profile(profile_path, n_layers, n_expert, ranked, profile_slots, err)) {
        std::fprintf(stderr, "strata-cache-replay: %s\n", err.c_str());
        return 1;
    }
    const int64_t take = std::min<int64_t>(slots, (int64_t) ranked.size());
    const std::vector<std::pair<int32_t, int32_t>> init(ranked.begin(), ranked.begin() + (std::size_t) take);

    std::printf("%s", strata::core::replay_report(windows, init, slots, k, n_layers, n_expert, pol).c_str());
    return 0;
}
