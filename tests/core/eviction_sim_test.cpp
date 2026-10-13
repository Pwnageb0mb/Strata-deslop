// tests/core/eviction_sim_test.cpp - the eviction policies against the reference's own report and CSV.
//
// tools/gen_eviction_golden.py runs tools/bench_eviction.py over two committed traces and records the report it
// printed and the CSV it wrote, for configurations that between them cover every policy, a multi-window split,
// both scopes, a profile-seated start and a small decay window.  This replays the same traces with the C++ port
// and compares the whole report and the whole CSV.
//
// The tight-slot configuration is the point: 10 distinct pairs against 4 and 8 slots, so the policies evict and
// diverge.  A fixture that fits every slot count has all four policies reporting the same rate, and a golden like
// that would pass for a port that got every eviction wrong.
//
//    eviction_sim_test <tests/eviction/trace_golden.txt>
#include "strata/core/eviction_sim.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
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
    std::fprintf(stderr, "FAIL %s (%zu vs %zu bytes)\n", what.c_str(), got.size(), want.size());
    std::istringstream g(got), w(want);
    std::string gl, wl;
    int n = 1;
    while (true) {
        const bool have_g = (bool) std::getline(g, gl), have_w = (bool) std::getline(w, wl);
        if (!have_g && !have_w) break;
        if (gl != wl) {
            std::fprintf(stderr, "  line %d:\n    got  %s\n    want %s\n", n, gl.c_str(), wl.c_str());
            break;
        }
        ++n;
    }
    ++failures;
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

struct Cfg {
    std::string name;
    std::vector<int64_t> slots;
    bool per_layer = false;
    std::vector<std::string> policies;
    int64_t windows = 1, decay_window = 65536;
    bool prefill = false;
    std::string report, csv;
};

} // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: eviction_sim_test <tests/eviction/trace_golden.txt>\n");
        return 2;
    }
    const std::string golden_path = argv[1];
    const std::string dir = std::filesystem::path(golden_path).parent_path().string();
    const std::string all = read_text(golden_path);
    if (all.empty()) {
        std::fprintf(stderr, "cannot read %s\n", golden_path.c_str());
        return 2;
    }

    int64_t n_layers = 0, n_expert = 0;
    std::vector<std::string> traces;
    std::string base_name;
    std::vector<Cfg> cfgs;

    std::istringstream ss(all);
    std::string line;
    Cfg* open = nullptr;
    int block = 0;   // 0 none, 1 report, 2 csv
    while (std::getline(ss, line)) {
        if (line.rfind("# report-begin ", 0) == 0) {
            block = 1;
            open = nullptr;
            for (Cfg& c : cfgs)
                if (c.name == line.substr(15)) open = &c;
            if (open == nullptr) {
                std::fprintf(stderr, "FAIL a report block names a configuration the golden never declared\n");
                ++failures;
            }
            continue;
        }
        if (line.rfind("# csv-begin ", 0) == 0) {
            block = 2;
            open = nullptr;
            for (Cfg& c : cfgs)
                if (c.name == line.substr(12)) open = &c;
            continue;
        }
        if (block != 0) {
            if (open != nullptr) {
                std::string& into = block == 1 ? open->report : open->csv;
                into += line + "\n";
            }
            continue;
        }
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ls(line);
        std::string what;
        ls >> what;
        if (what == "n_layers") ls >> n_layers;
        else if (what == "n_expert") ls >> n_expert;
        else if (what == "trace") { std::string n; ls >> n; traces.push_back(n); }
        else if (what == "base") ls >> base_name;
        else if (what == "cfg") {
            std::string name, key, val;
            ls >> name >> key >> val;
            Cfg* c = nullptr;
            for (Cfg& e : cfgs)
                if (e.name == name) c = &e;
            if (c == nullptr) {
                cfgs.push_back(Cfg{});
                c = &cfgs.back();
                c->name = name;
            }
            if (key == "slots") for (const std::string& s : split(val, ',')) c->slots.push_back(std::atoll(s.c_str()));
            else if (key == "per_layer") c->per_layer = val == "1";
            else if (key == "policies") c->policies = split(val, ',');
            else if (key == "windows") c->windows = std::atoll(val.c_str());
            else if (key == "decay_window") c->decay_window = std::atoll(val.c_str());
            else if (key == "prefill") c->prefill = val == "1";
        }
        // the tensor/profile lines belong to expert_profile_test
    }
    if (traces.size() != 2 || cfgs.empty()) {
        std::fprintf(stderr, "the golden is not the shape this test expects\n");
        return 1;
    }

    std::vector<uint32_t> codes;
    for (const std::string& t : traces) {
        std::string err;
        if (!strata::core::read_routing_trace(dir + "/" + t, n_layers, n_expert, codes, err)) {
            std::fprintf(stderr, "FAIL read_routing_trace refused %s: %s\n", t.c_str(), err.c_str());
            return 1;
        }
    }
    std::vector<std::pair<int32_t, int32_t>> base_pairs;
    int64_t base_slots = 0;
    std::string err;
    if (!strata::core::read_expert_profile(dir + "/" + base_name, n_layers, n_expert, base_pairs, base_slots, err)) {
        std::fprintf(stderr, "FAIL read_expert_profile refused %s: %s\n", base_name.c_str(), err.c_str());
        return 1;
    }

    for (const Cfg& c : cfgs) {
        const std::vector<strata::core::EvictionRow> rows = strata::core::eviction_run(
            codes, n_layers, n_expert, c.slots, c.per_layer, c.policies, c.windows, c.decay_window,
            c.prefill ? &base_pairs : nullptr);
        check("report " + c.name, strata::core::eviction_report(rows, c.windows), c.report);
        check("csv " + c.name, strata::core::eviction_csv(rows, c.per_layer ? "per-layer" : "global"), c.csv);
    }

    // ---- the configuration the reference crashes on is REFUSED, not answered
    // (per-layer with fewer slots than layers: q = 0, so every layer but the last has no capacity, and the
    // reference's Lru pops an empty OrderedDict).  fill-only never evicts, so it is allowed.
    if (strata::core::eviction_config_error(n_layers, 4, true, "lru").empty()) {
        std::fprintf(stderr, "FAIL per-layer with 4 slots and lru was not refused\n");
        ++failures;
    }
    if (!strata::core::eviction_config_error(n_layers, 4, true, "fill-only").empty()) {
        std::fprintf(stderr, "FAIL per-layer with 4 slots and fill-only was refused, and the reference runs it\n");
        ++failures;
    }
    if (!strata::core::eviction_config_error(n_layers, 4105, true, "lru").empty() ||
        !strata::core::eviction_config_error(n_layers, 96, true, "lfu-decay").empty() ||
        !strata::core::eviction_config_error(n_layers, 4, false, "oracle").empty()) {
        std::fprintf(stderr, "FAIL a replayable configuration was refused\n");
        ++failures;
    }

    if (failures) {
        std::fprintf(stderr, "eviction_sim_test: FAILED (%d)\n", failures);
        return 1;
    }
    std::printf("eviction_sim_test: all %zu configurations match the reference's own report and CSV, and the "
                "configuration it crashes on is refused\n", cfgs.size());
    return 0;
}
