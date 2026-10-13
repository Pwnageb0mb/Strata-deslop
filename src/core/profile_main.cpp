// src/core/profile_main.cpp - `strata-profile`: the C++ port of tools/make_profile.py.
//
// Reads `--dump-routing` traces, ranks every (layer, expert) pair (the base's order, then the traces' pairs most
// frequent first, then the fill) and writes a profile `--expert-profile` reads.
//
// The reference is tools/make_profile.py, and the port is accepted against its own output:
// tests/core/expert_profile_test.cpp compares the profile BYTES for every mode.
//
//    strata-profile TRACE... [--base PATH | --no-base] [--reorder] [--out PATH] [--n-expert 512]
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "strata/core/expert_profile.hpp"

namespace {

int usage() {
    std::fprintf(stderr,
                 "usage: strata-profile TRACE... [--base PATH | --no-base] [--reorder] [--out PATH]\n"
                 "                       [--n-expert N]\n"
                 "\n"
                 "  TRACE...        routing traces from `strata generate --dump-routing`\n"
                 "  --base PATH     the ranking to keep first (default: data/expert-profile.bin)\n"
                 "  --no-base       rank by the traces only\n"
                 "  --reorder       rank the traces' pairs before the base's - needed when the base already ranks\n"
                 "                  every pair, where --base alone cannot change the order\n"
                 "  --out PATH      where to write the profile (default: data/expert-profile.bin)\n"
                 "  --n-expert N    experts per layer (default 512; a pruned model keeps 256 of 512)\n");
    return 2;
}

} // namespace

int main(int argc, char** argv) {
    const int64_t n_layers = strata::core::kProfileLayers;
    int64_t n_expert = strata::core::kProfileExperts;
    std::string base = "data/expert-profile.bin";
    std::string out = "data/expert-profile.bin";
    bool no_base = false, reorder = false;
    std::vector<std::string> traces;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char* what) -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "strata-profile: %s needs a value\n", what);
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "-h" || a == "--help") return usage();
        else if (a == "--base") base = next("--base");
        else if (a == "--no-base") no_base = true;
        else if (a == "--reorder") reorder = true;
        else if (a == "--out") out = next("--out");
        else if (a == "--n-expert") n_expert = std::atoll(next("--n-expert"));
        else if (!a.empty() && a[0] == '-') {
            std::fprintf(stderr, "strata-profile: unknown option %s\n", a.c_str());
            return usage();
        } else {
            traces.push_back(a);
        }
    }
    if (n_expert <= 0 || n_expert > 65535) {
        std::fprintf(stderr, "strata-profile: --n-expert %lld is not a usable expert count\n", (long long) n_expert);
        return 2;
    }

    // the traces' frequencies, summed over the files in the order given (the reference sums the same way)
    std::vector<uint32_t> all;
    for (const std::string& t : traces) {
        std::vector<uint32_t> codes;
        std::string err;
        if (!strata::core::read_routing_trace(t, n_layers, n_expert, codes, err)) {
            std::fprintf(stderr, "strata-profile: %s\n", err.c_str());
            return 1;
        }
        all.insert(all.end(), codes.begin(), codes.end());
    }
    const auto freq = strata::core::trace_frequencies(all);

    std::vector<std::pair<int32_t, int32_t>> base_pairs;
    if (!no_base) {
        int64_t slots = 0;
        std::string err;
        if (!strata::core::read_expert_profile(base, n_layers, n_expert, base_pairs, slots, err)) {
            std::fprintf(stderr, "strata-profile: %s\n", err.c_str());
            return 1;
        }
    }

    int64_t counts[3] = {0, 0, 0};
    const auto ranked =
        strata::core::rank_profile(base_pairs, freq, n_layers, n_expert, no_base, reorder, counts);
    std::string err;
    if (!strata::core::write_expert_profile(out, n_layers, n_expert, ranked, err)) {
        std::fprintf(stderr, "strata-profile: %s\n", err.c_str());
        return 1;
    }

    // the reference asserts this too: a profile that does not read back as the ranking that was written is a
    // format bug, and it is cheaper to find here than in a start that seats the wrong experts
    std::vector<std::pair<int32_t, int32_t>> back;
    int64_t slots = 0;
    if (!strata::core::read_expert_profile(out, n_layers, n_expert, back, slots, err)) {
        std::fprintf(stderr, "strata-profile: %s\n", err.c_str());
        return 1;
    }
    if (back != ranked) {
        std::fprintf(stderr, "strata-profile: the profile did not survive the round trip\n");
        return 1;
    }

    std::printf("wrote %s: %zu ranked pairs (%lld from the base, %lld from the traces, %lld filled in)\n",
                out.c_str(), ranked.size(), (long long) counts[0], (long long) counts[1], (long long) counts[2]);
    return 0;
}
