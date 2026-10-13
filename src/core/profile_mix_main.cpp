// src/core/profile_mix_main.cpp - `strata-profile-mix`: the C++ port of tools/make_profile_mix.py.
//
// An expert-cache profile from several routing traces, each trace counting the SAME: every trace becomes a
// distribution (its counts / its total) and is multiplied by its weight, and the ranking is the weighted sum.
// `strata-profile` adds raw counts instead, so there the longest trace decides the ranking - which is why the
// reference has two tools and so does this.
//
//    strata-profile-mix TRACE[:WEIGHT]... --out PATH [--n-expert 512]
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "strata/core/expert_profile.hpp"

namespace {

int usage() {
    std::fprintf(stderr,
                 "usage: strata-profile-mix TRACE[:WEIGHT]... --out PATH [--n-expert N]\n"
                 "\n"
                 "  TRACE[:WEIGHT]  a --dump-routing file (or the SYCL engine's STRATA_USAGE_DUMP file: same\n"
                 "                  record format), optionally with a weight (default 1.0).  Give a group of N\n"
                 "                  traces the weight 1/N each to make the group count once.\n"
                 "  --out PATH      where to write the profile (required)\n"
                 "  --n-expert N    experts per layer (default 512)\n");
    return 2;
}

} // namespace

int main(int argc, char** argv) {
    const int64_t n_layers = strata::core::kProfileLayers;
    int64_t n_expert = strata::core::kProfileExperts;
    std::string out;
    std::vector<std::string> specs;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "-h" || a == "--help") return usage();
        if (a == "--out") {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "strata-profile-mix: --out needs a value\n");
                return 2;
            }
            out = argv[++i];
        } else if (a == "--n-expert") {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "strata-profile-mix: --n-expert needs a value\n");
                return 2;
            }
            n_expert = std::atoll(argv[++i]);
        } else if (!a.empty() && a[0] == '-') {
            std::fprintf(stderr, "strata-profile-mix: unknown option %s\n", a.c_str());
            return usage();
        } else {
            specs.push_back(a);
        }
    }
    if (specs.empty() || out.empty()) return usage();

    std::vector<std::vector<std::pair<std::pair<int32_t, int32_t>, uint64_t>>> per_trace;
    std::vector<double> weights;
    std::vector<uint8_t> scored((std::size_t) (n_layers * n_expert), 0);
    std::size_t n_scored = 0;

    for (const std::string& spec : specs) {
        // TRACE:WEIGHT - a colon whose tail is not a number is part of the path (a Windows drive letter)
        std::string path = spec;
        double weight = 1.0;
        const std::size_t colon = spec.rfind(':');
        if (colon != std::string::npos) {
            const std::string tail = spec.substr(colon + 1);
            char* end = nullptr;
            const double v = std::strtod(tail.c_str(), &end);
            if (end != nullptr && *end == '\0' && !tail.empty()) {
                path = spec.substr(0, colon);
                weight = v;
            }
        }
        std::vector<uint32_t> codes;
        std::string err;
        if (!strata::core::read_routing_trace(path, n_layers, n_expert, codes, err)) {
            std::fprintf(stderr, "strata-profile-mix: %s\n", err.c_str());
            return 1;
        }
        if (codes.empty()) {
            std::printf("%s: empty trace, skipped\n", path.c_str());
            continue;
        }
        const auto freq = strata::core::trace_frequencies(codes);
        for (const auto& [pair, _] : freq) {
            const std::size_t i = (std::size_t) (pair.first * n_expert + pair.second);
            if (!scored[i]) {
                scored[i] = 1;
                ++n_scored;
            }
        }
        per_trace.push_back(freq);
        weights.push_back(weight);
    }

    const auto ranked = strata::core::rank_profile_mix(per_trace, weights, n_layers, n_expert);
    std::string err;
    if (!strata::core::write_expert_profile(out, n_layers, n_expert, ranked, err)) {
        std::fprintf(stderr, "strata-profile-mix: %s\n", err.c_str());
        return 1;
    }
    // the reference asserts this too: a profile that does not read back as the ranking that was written is a
    // format bug, and it is cheaper to find here than in a start that seats the wrong experts
    std::vector<std::pair<int32_t, int32_t>> back;
    int64_t slots = 0;
    if (!strata::core::read_expert_profile(out, n_layers, n_expert, back, slots, err)) {
        std::fprintf(stderr, "strata-profile-mix: %s\n", err.c_str());
        return 1;
    }
    if (back != ranked) {
        std::fprintf(stderr, "strata-profile-mix: the profile did not survive the round trip\n");
        return 1;
    }
    std::printf("wrote %s: %zu ranked pairs, %zu from %zu traces\n", out.c_str(), ranked.size(), n_scored,
                specs.size());
    return 0;
}
