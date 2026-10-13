// tests/core/cache_replay_test.cpp - the engine's adaptive policy, replayed, against the reference's own stdout.
//
// tools/gen_cache_replay_golden.py builds a trace that forms COMPLETE windows (the reference keeps only windows
// with one group per layer, so a toy trace loads as none), runs tools/replay_cache_policy.py over it for two
// parameter sets and records what it printed - plus the window GROUPING it loaded, which is the subtle part.
//
// The fixture is chosen so the mechanism fires: the profile's experts are routed only in the first window, so
// their usage stays below the candidate threshold and they are the victims the adaptive round swaps out.  A
// profile nothing routes gives `share 0.0000` in every configuration, and a golden like that passes for a port
// that prints zeros.
//
//    cache_replay_test <tests/eviction/cache_replay_golden.txt>
#include "strata/core/cache_replay.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "strata/core/expert_profile.hpp"

namespace {

int failures = 0;

std::string read_text(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return {};
    return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

void check(const std::string& what, const std::string& got, const std::string& want) {
    if (got == want) return;
    std::fprintf(stderr, "FAIL %s\n  got  %s\n  want %s\n", what.c_str(), got.c_str(), want.c_str());
    ++failures;
}

/// One window as the golden writes it: `layer:id,id layer:id,id ...`, the same shape the loader produces.
std::string window_text(const strata::core::ReplayWindow& w) {
    std::string s;
    for (const strata::core::ReplayGroup& g : w.groups) {
        if (!s.empty()) s += " ";
        s += std::to_string(g.layer) + ":";
        for (std::size_t i = 0; i < g.ids.size(); ++i) {
            if (i) s += ",";
            s += std::to_string(g.ids[i]);
        }
    }
    return s;
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: cache_replay_test <tests/eviction/cache_replay_golden.txt>\n");
        return 2;
    }
    const std::string golden_path = argv[1];
    const std::string dir = std::filesystem::path(golden_path).parent_path().string();
    const std::string all = read_text(golden_path);
    if (all.empty()) {
        std::fprintf(stderr, "cannot read %s\n", golden_path.c_str());
        return 2;
    }

    std::string trace_name, profile_name;
    int64_t n_layers = 0, n_expert = 0, want_k = 0, want_windows = 0, slots = 0;
    int64_t loaded_windows = -1, loaded_k = -1;
    std::vector<std::string> want_window_text;
    std::map<std::string, std::string> stdout_by_cfg;
    std::vector<std::string> cfg_order;

    std::istringstream ss(all);
    std::string line;
    while (std::getline(ss, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ls(line);
        std::string what;
        ls >> what;
        if (what == "trace") ls >> trace_name;
        else if (what == "profile") ls >> profile_name;
        else if (what == "n_layers") ls >> n_layers;
        else if (what == "n_expert") ls >> n_expert;
        else if (what == "k") ls >> want_k;
        else if (what == "n_windows") ls >> want_windows;
        else if (what == "slots") ls >> slots;
        else if (what == "loaded_windows") ls >> loaded_windows;
        else if (what == "loaded_k") ls >> loaded_k;
        else if (what == "window") {
            int64_t wi = 0;
            ls >> wi;
            std::string rest;
            std::getline(ls, rest);
            if (!rest.empty() && rest[0] == ' ') rest.erase(0, 1);
            if ((int64_t) want_window_text.size() <= wi) want_window_text.resize((std::size_t) wi + 1);
            want_window_text[(std::size_t) wi] = rest;
        } else if (what == "stdout") {
            std::string name, rest;
            ls >> name;
            std::getline(ls, rest);
            if (!rest.empty() && rest[0] == ' ') rest.erase(0, 1);
            if (stdout_by_cfg.find(name) == stdout_by_cfg.end()) cfg_order.push_back(name);
            stdout_by_cfg[name] += rest + "\n";
        } else {
            std::fprintf(stderr, "unknown golden line: %s\n", line.c_str());
            ++failures;
        }
    }
    if (trace_name.empty() || cfg_order.empty() || loaded_windows < 0) {
        std::fprintf(stderr, "the golden is not the shape this test expects\n");
        return 1;
    }

    // ---- the loader, including the GROUPING (the part a wrong window would hide)
    std::vector<strata::core::ReplayWindow> windows;
    int64_t k = 0;
    std::string err;
    if (!strata::core::read_replay_trace(dir + "/" + trace_name, n_layers, n_expert, windows, k, err)) {
        std::fprintf(stderr, "FAIL read_replay_trace refused %s: %s\n", trace_name.c_str(), err.c_str());
        return 1;
    }
    check("loaded_k", std::to_string(k), std::to_string(loaded_k));
    check("loaded_windows", std::to_string(windows.size()), std::to_string(loaded_windows));
    check("k", std::to_string(k), std::to_string(want_k));
    check("n_windows", std::to_string(windows.size()), std::to_string(want_windows));
    if (windows.size() != want_window_text.size()) {
        std::fprintf(stderr, "FAIL the golden records %zu windows, the loader kept %zu\n", want_window_text.size(),
                     windows.size());
        ++failures;
    }
    for (std::size_t i = 0; i < windows.size() && i < want_window_text.size(); ++i)
        check("window " + std::to_string(i), window_text(windows[i]), want_window_text[i]);

    // ---- the profile, and the replay for each recorded parameter set
    std::vector<std::pair<int32_t, int32_t>> ranked;
    int64_t profile_slots = 0;
    if (!strata::core::read_expert_profile(dir + "/" + profile_name, n_layers, n_expert, ranked, profile_slots, err)) {
        std::fprintf(stderr, "FAIL read_expert_profile refused %s: %s\n", profile_name.c_str(), err.c_str());
        return 1;
    }
    const int64_t take = std::min<int64_t>(slots, (int64_t) ranked.size());
    const std::vector<std::pair<int32_t, int32_t>> init(ranked.begin(), ranked.begin() + (std::size_t) take);

    for (const std::string& name : cfg_order) {
        const std::string& want = stdout_by_cfg[name];
        // the parameters come from the reference's own last line, which prints every / swaps / decay
        strata::core::ReplayPolicy pol;
        long long every = 0, swaps = 0;
        double decay = 0.0;
        const std::size_t at = want.find("policy every ");
        if (at == std::string::npos ||
            std::sscanf(want.c_str() + at, "policy every %lld, swaps %lld, decay %lf", &every, &swaps, &decay) != 3) {
            std::fprintf(stderr, "FAIL %s: the golden's last line does not carry the parameters\n", name.c_str());
            ++failures;
            continue;
        }
        pol.every = (int64_t) every;
        pol.swaps = (int64_t) swaps;
        pol.decay = decay;
        check("stdout " + name,
              strata::core::replay_report(windows, init, slots, k, n_layers, n_expert, pol), want);
    }

    // ---- and the refusal: a trace with no complete window is what the reference divides by
    {
        std::vector<strata::core::ReplayWindow> none;
        int64_t kk = 0;
        // a trace of a single record can never form a window: one group, and 48 are needed
        const std::string tmp = dir + "/.one-record-trace.bin";
        std::ofstream(tmp, std::ios::binary).write("\x00\x00\x00\x00\x01\x00\x00\x00\x05\x00\x00\x00\x00\x00\x00\x00",
                                                   16);
        if (!strata::core::read_replay_trace(tmp, n_layers, n_expert, none, kk, err)) {
            std::fprintf(stderr, "FAIL a one-record trace was refused by the loader: %s\n", err.c_str());
            ++failures;
        } else if (!none.empty()) {
            std::fprintf(stderr, "FAIL a one-record trace loaded %zu windows, and 48 groups are needed\n",
                         none.size());
            ++failures;
        }
        std::remove(tmp.c_str());
    }

    if (failures) {
        std::fprintf(stderr, "cache_replay_test: FAILED (%d)\n", failures);
        return 1;
    }
    std::printf("cache_replay_test: the window grouping and all %zu replayed parameter sets match the reference's "
                "own output\n", cfg_order.size());
    return 0;
}
