// src/core/evict_main.cpp - `strata-evict`: the C++ port of tools/bench_eviction.py.
//
// Replays `--dump-routing` traces against candidate ExpertCache eviction policies, without a GPU: a policy's hit
// rate depends only on the ORDER the run routed (layer, expert) pairs, and the trace records exactly that order.
//
// AGENTS.md: a number needs what it was measured on.  The trace files and the lookup and distinct-pair counts it
// ran over are printed, and belong in any report of these rates.  The trace itself comes from a one-shot engine
// run (`--dump-routing` on a prompt typical of the workload) - the model, quant and prompt are the environment.
//
//    strata-evict TRACE... [--slots 4105] [--per-layer] [--windows 4] [--policies ...]
//                          [--decay-window 65536] [--prefill PROFILE] [--n-expert 512] [--csv out.csv]
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "strata/core/eviction_sim.hpp"
#include "strata/core/expert_profile.hpp"

namespace {

int usage() {
    std::fprintf(stderr,
                 "usage: strata-evict TRACE... [--slots 4105] [--per-layer] [--windows 4]\n"
                 "                     [--policies fill-only,lru,lfu-decay,oracle] [--decay-window 65536]\n"
                 "                     [--prefill PROFILE] [--n-expert 512] [--csv OUT.CSV]\n"
                 "\n"
                 "  TRACE...        --dump-routing files, replayed in the order given\n"
                 "  --slots N[,N]   comma-separated slot counts (default: 4105, the R4 reference sizing)\n"
                 "  --per-layer     R4.2g admission: each layer its own slot range, eviction stays in-layer\n"
                 "  --windows N     split the trace into N equal windows and report the hit-rate per window\n"
                 "  --prefill P     profile.bin to start from (the engine's --expert-profile start)\n");
    return 2;
}

/// Python's f"{n:,}": the reference prints the counts grouped, and a report of a hit rate is quoted with them.
std::string grouped(int64_t n) {
    std::string s = std::to_string(n < 0 ? -n : n), out;
    for (std::size_t i = 0; i < s.size(); ++i) {
        if (i > 0 && (s.size() - i) % 3 == 0) out += ',';
        out += s[i];
    }
    return (n < 0 ? "-" : "") + out;
}

std::vector<std::string> split(const std::string& s, char sep) {
    std::vector<std::string> out;
    std::size_t at = 0;
    while (true) {
        const std::size_t next = s.find(sep, at);
        out.push_back(s.substr(at, next == std::string::npos ? std::string::npos : next - at));
        if (next == std::string::npos) return out;
        at = next + 1;
    }
}

} // namespace

int main(int argc, char** argv) {
    const int64_t n_layers = strata::core::kProfileLayers;
    int64_t n_expert = strata::core::kProfileExperts;
    std::string slots_arg = "4105";
    std::string policies_arg;
    std::string prefill_path, csv_path;
    bool per_layer = false;
    int64_t windows = 1, decay_window = 65536;
    std::vector<std::string> traces;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char* what) -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "strata-evict: %s needs a value\n", what);
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "-h" || a == "--help") return usage();
        else if (a == "--slots") slots_arg = next("--slots");
        else if (a == "--per-layer") per_layer = true;
        else if (a == "--windows") windows = std::atoll(next("--windows"));
        else if (a == "--policies") policies_arg = next("--policies");
        else if (a == "--decay-window") decay_window = std::atoll(next("--decay-window"));
        else if (a == "--prefill") prefill_path = next("--prefill");
        else if (a == "--n-expert") n_expert = std::atoll(next("--n-expert"));
        else if (a == "--csv") csv_path = next("--csv");
        else if (!a.empty() && a[0] == '-') {
            std::fprintf(stderr, "strata-evict: unknown option %s\n", a.c_str());
            return usage();
        } else {
            traces.push_back(a);
        }
    }
    if (traces.empty()) return usage();
    if (windows < 1) {
        std::fprintf(stderr, "strata-evict: --windows %lld is not a usable window count\n", (long long) windows);
        return 2;
    }

    std::vector<uint32_t> codes;
    for (const std::string& t : traces) {
        std::vector<uint32_t> part;
        std::string err;
        if (!strata::core::read_routing_trace(t, n_layers, n_expert, part, err)) {
            std::fprintf(stderr, "strata-evict: %s\n", err.c_str());
            return 1;
        }
        codes.insert(codes.end(), part.begin(), part.end());
    }
    if (codes.empty()) {
        std::fprintf(stderr, "strata-evict: no usable lookups in the trace(s)\n");
        return 1;
    }

    std::vector<int64_t> slots_list;
    for (const std::string& s : split(slots_arg, ',')) slots_list.push_back(std::atoll(s.c_str()));
    const std::vector<std::string> policies =
        policies_arg.empty() ? strata::core::eviction_policy_names() : split(policies_arg, ',');

    for (const std::string& p : policies) {
        const std::vector<std::string>& have = strata::core::eviction_policy_names();
        if (std::find(have.begin(), have.end(), p) == have.end()) {
            std::fprintf(stderr, "strata-evict: unknown policy %s (have:", p.c_str());
            for (const std::string& h : have) std::fprintf(stderr, " %s", h.c_str());
            std::fprintf(stderr, ")\n");
            return 1;
        }
        for (int64_t slots : slots_list) {
            // the configuration the reference crashes on: refusing beats reporting numbers it cannot produce
            const std::string bad = strata::core::eviction_config_error(n_layers, slots, per_layer, p);
            if (!bad.empty()) {
                std::fprintf(stderr, "strata-evict: %s\n", bad.c_str());
                return 1;
            }
        }
    }

    std::vector<std::pair<int32_t, int32_t>> prefill;
    const std::vector<std::pair<int32_t, int32_t>>* prefill_ptr = nullptr;
    if (!prefill_path.empty()) {
        int64_t slots = 0;
        std::string err;
        if (!strata::core::read_expert_profile(prefill_path, n_layers, n_expert, prefill, slots, err)) {
            std::fprintf(stderr, "strata-evict: %s\n", err.c_str());
            return 1;
        }
        prefill_ptr = &prefill;
    }

    std::string joined;
    for (const std::string& t : traces) joined += (joined.empty() ? "" : ", ") + t;
    std::printf("trace: %s\n", joined.c_str());

    std::vector<uint32_t> distinct = codes;
    std::sort(distinct.begin(), distinct.end());
    distinct.erase(std::unique(distinct.begin(), distinct.end()), distinct.end());
    const std::string scope = per_layer ? "per-layer" : "global";
    std::printf("%s lookups, %s distinct (layer, expert) pairs, scope: %s, prefill: %s", grouped((int64_t) codes.size()).c_str(),
                grouped((int64_t) distinct.size()).c_str(), scope.c_str(),
                prefill_path.empty() ? "none (cold start)" : prefill_path.c_str());
    const bool has_decay = std::find(policies.begin(), policies.end(), "lfu-decay") != policies.end();
    if (has_decay) std::printf(", decay-window: %lld", (long long) decay_window);
    std::printf("\n");
    if ((int64_t) distinct.size() <= *std::min_element(slots_list.begin(), slots_list.end()))
        std::printf("note: the distinct pairs fit in every slot count given - fill-only cannot be beaten "
                    "on this trace; use a longer trace or fewer slots to see policy differences\n");

    const std::vector<strata::core::EvictionRow> rows =
        strata::core::eviction_run(codes, n_layers, n_expert, slots_list, per_layer, policies, windows,
                                   decay_window, prefill_ptr);
    std::printf("%s", strata::core::eviction_report(rows, windows).c_str());

    if (!csv_path.empty()) {
        const std::string csv = strata::core::eviction_csv(rows, scope);
        std::FILE* f = std::fopen(csv_path.c_str(), "wb");
        if (f == nullptr) {
            std::fprintf(stderr, "strata-evict: cannot write %s\n", csv_path.c_str());
            return 1;
        }
        const bool ok = csv.empty() || std::fwrite(csv.data(), 1, csv.size(), f) == csv.size();
        const bool closed = std::fclose(f) == 0;
        if (!ok || !closed) {
            std::fprintf(stderr, "strata-evict: cannot write %s\n", csv_path.c_str());
            return 1;
        }
        std::printf("\nwrote %s\n", csv_path.c_str());
    }
    return 0;
}
