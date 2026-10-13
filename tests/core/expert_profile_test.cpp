// tests/core/expert_profile_test.cpp - the routing trace and the profile ranking against the reference's own.
//
// tools/gen_eviction_golden.py runs tools/make_profile.py and tools/bench_eviction.py over two committed traces
// and records what their readers produced (the frequencies make_profile computes AND the order bench_eviction
// keeps) and what the ranking produced for every mode.  This reads the same fixtures with the C++ port and
// compares: the ordered codes, the frequencies, and the profile BYTES the reference wrote for each mode.
//
// The fixtures are committed beside the golden: trace-1.bin, trace-2.bin, base.bin and profile-<mode>.bin.
//
//    expert_profile_test <tests/eviction/trace_golden.txt>
#include "strata/core/expert_profile.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

int failures = 0;

std::string read_text(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return {};
    return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

std::string pairs_to_text(const std::vector<std::pair<int32_t, int32_t>>& v) {
    std::string s;
    for (std::size_t i = 0; i < v.size(); ++i) {
        if (i) s += " ";
        s += std::to_string(v[i].first) + "," + std::to_string(v[i].second);
    }
    return s;
}

void check(const std::string& what, const std::string& got, const std::string& want) {
    if (got != want) {
        std::fprintf(stderr, "FAIL %s\n  got  %s\n  want %s\n", what.c_str(), got.c_str(), want.c_str());
        ++failures;
    }
}

struct Mix {
    std::string name;
    std::vector<std::string> specs;
    std::string bytes, count, head, tail, stdout_text, skipped;
};

/// One ranking mode's recorded output: the profile's size, its head and tail, and how many pairs it ranked.
struct Mode {
    std::string name, head, tail, bytes, count;
};

} // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: expert_profile_test <tests/eviction/trace_golden.txt>\n");
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
    std::string want_freq, want_codes;
    std::vector<std::pair<int32_t, int32_t>> want_base_ranked;
    std::vector<Mix> mixes;
    std::vector<Mode> modes;

    std::istringstream ss(all);
    std::string line;
    while (std::getline(ss, line)) {
        // the eviction benchmark's blocks come last and are read by eviction_sim_test: stop at the first one
        // (checked before the comment skip, because the marker is itself a comment)
        if (line.rfind("# report-begin ", 0) == 0) break;
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ls(line);
        std::string what;
        ls >> what;
        if (what == "n_layers") ls >> n_layers;
        else if (what == "n_expert") ls >> n_expert;
        else if (what == "trace") { std::string name; ls >> name; traces.push_back(name); }
        else if (what == "base") ls >> base_name;
        else if (what == "base_ranked") {
            std::string tok;
            while (ls >> tok) {
                const std::size_t comma = tok.find(',');
                want_base_ranked.push_back({std::atoi(tok.substr(0, comma).c_str()),
                                            std::atoi(tok.substr(comma + 1).c_str())});
            }
        } else if (what == "freq") {
            int64_t l = 0, e = 0, c = 0;
            ls >> l >> e >> c;
            if (!want_freq.empty()) want_freq += "\n";
            want_freq += std::to_string(l) + " " + std::to_string(e) + " " + std::to_string(c);
        } else if (what == "codes") {
            std::string tok;
            std::vector<std::pair<int32_t, int32_t>> v;
            while (ls >> tok) {
                const std::size_t comma = tok.find(',');
                v.push_back({std::atoi(tok.substr(0, comma).c_str()), std::atoi(tok.substr(comma + 1).c_str())});
            }
            want_codes = pairs_to_text(v);
        } else if (what == "profile") {
            Mode m;
            ls >> m.name;
            std::string key;
            while (ls >> key) {
                if (key == "bytes") ls >> m.bytes;
                else if (key == "sha256") { std::string sha; ls >> sha; }
            }
            modes.push_back(m);
        } else if (what == "profile_head") {
            Mode m;
            ls >> m.name;
            std::string tok;
            std::vector<std::pair<int32_t, int32_t>> v;
            while (ls >> tok) {
                const std::size_t comma = tok.find(',');
                v.push_back({std::atoi(tok.substr(0, comma).c_str()), std::atoi(tok.substr(comma + 1).c_str())});
            }
            for (Mode& e : modes)
                if (e.name == m.name) e.head = pairs_to_text(v);
        } else if (what == "profile_tail") {
            std::string name, tok;
            ls >> name;
            std::vector<std::pair<int32_t, int32_t>> v;
            while (ls >> tok) {
                const std::size_t comma = tok.find(',');
                v.push_back({std::atoi(tok.substr(0, comma).c_str()), std::atoi(tok.substr(comma + 1).c_str())});
            }
            for (Mode& e : modes)
                if (e.name == name) e.tail = pairs_to_text(v);
        } else if (what == "profile_count") {
            std::string name, count;
            ls >> name >> count;
            for (Mode& e : modes)
                if (e.name == name) e.count = count;
        } else if (what == "mix") {
            Mix m;
            ls >> m.name;
            std::vector<std::string> tok;
            std::string t;
            while (ls >> t) tok.push_back(t);
            std::size_t i = 0;
            if (!tok.empty() && tok[0] == "specs") i = 1;   // the label, then the specs
            for (; i < tok.size() && tok[i] != "bytes"; ++i) m.specs.push_back(tok[i]);
            if (i + 5 < tok.size()) {
                m.bytes = tok[i + 1];      // bytes N sha256 H count N
                m.count = tok[i + 5];
            }
            mixes.push_back(m);
        } else if (what == "mix_head" || what == "mix_tail" || what == "mix_stdout" || what == "mix_skipped") {
            std::string name, rest;
            ls >> name;
            std::getline(ls, rest);
            if (!rest.empty() && rest[0] == ' ') rest.erase(0, 1);
            for (Mix& m : mixes)
                if (m.name == name) {
                    if (what == "mix_head") m.head = rest;
                    else if (what == "mix_tail") m.tail = rest;
                    else if (what == "mix_stdout") m.stdout_text = rest;
                    else m.skipped = rest;
                }
        } else if (what == "summary" || what == "cfg") {
            // the eviction benchmark's section, which eviction_sim_test reads; not this test's
        } else {
            std::fprintf(stderr, "unknown golden line: %s\n", line.c_str());
            ++failures;
        }
    }
    if (traces.size() != 2 || modes.size() != 3 || want_codes.empty()) {
        std::fprintf(stderr, "the golden is not the shape this test expects\n");
        return 1;
    }

    // ---- the reader: the ordered codes, and the frequencies derived from them
    std::vector<uint32_t> codes;
    for (const std::string& t : traces) {
        std::string err;
        if (!strata::core::read_routing_trace(dir + "/" + t, n_layers, n_expert, codes, err)) {
            std::fprintf(stderr, "FAIL read_routing_trace refused %s: %s\n", t.c_str(), err.c_str());
            ++failures;
            return 1;
        }
    }
    std::vector<std::pair<int32_t, int32_t>> got_codes;
    for (uint32_t c : codes) got_codes.push_back(strata::core::trace_decode(c));
    check("the ordered codes", pairs_to_text(got_codes), want_codes);

    std::string got_freq;
    for (const auto& [pair, count] : strata::core::trace_frequencies(codes)) {
        if (!got_freq.empty()) got_freq += "\n";
        got_freq += std::to_string(pair.first) + " " + std::to_string(pair.second) + " " + std::to_string(count);
    }
    check("the frequencies", got_freq, want_freq);

    // ---- the base, read back with the engine's own reader
    int64_t slots = 0;
    std::string err;
    std::vector<std::pair<int32_t, int32_t>> base_pairs;
    if (!strata::core::read_expert_profile(dir + "/" + base_name, n_layers, n_expert, base_pairs, slots, err)) {
        std::fprintf(stderr, "FAIL read_expert_profile refused %s: %s\n", base_name.c_str(), err.c_str());
        return 1;
    }
    check("the base's ranking", pairs_to_text(base_pairs), pairs_to_text(want_base_ranked));

    // ---- the ranking, every mode, against the reference's own profile BYTES
    char tmpl[] = "expert-profile-XXXXXX";
    char* made = mkdtemp(tmpl);
    if (made == nullptr) {
        std::fprintf(stderr, "FAIL cannot make a temporary directory\n");
        return 1;
    }
    const std::string tmp = std::string(made) + "/out.bin";
    const auto freq = strata::core::trace_frequencies(codes);
    for (const Mode& m : modes) {
        int64_t counts[3] = {0, 0, 0};
        const auto ranked = strata::core::rank_profile(base_pairs, freq, n_layers, n_expert,
                                                       m.name == "nobase", m.name == "reorder", counts);
        check("profile_count " + m.name, std::to_string(ranked.size()), m.count);
        check("profile_head " + m.name, pairs_to_text({ranked.begin(), ranked.begin() + 12}), m.head);
        check("profile_tail " + m.name,
              pairs_to_text({ranked.end() - 4, ranked.end()}), m.tail);
        if (!strata::core::write_expert_profile(tmp, n_layers, n_expert, ranked, err)) {
            std::fprintf(stderr, "FAIL write_expert_profile refused: %s\n", err.c_str());
            ++failures;
            continue;
        }
        const std::string got = read_text(tmp);
        const std::string want = read_text(dir + "/profile-" + m.name + ".bin");
        check("profile " + m.name + " bytes", std::to_string(got.size()), m.bytes);
        if (got != want) {
            std::fprintf(stderr, "FAIL profile %s differs from the reference (%zu vs %zu bytes)\n", m.name.c_str(),
                         got.size(), want.size());
            for (std::size_t i = 0; i < got.size() && i < want.size(); ++i)
                if (got[i] != want[i]) {
                    std::fprintf(stderr, "  first difference at byte %zu\n", i);
                    break;
                }
            ++failures;
        }
        std::remove(tmp.c_str());
    }

    // ---- the weighted mix: make_profile_mix's ranking, against the reference's own profile BYTES.  The specs
    // cover a weight, a ZERO weight (a pair present with score 0 is ranked, not filled) and an empty trace.
    for (const Mix& m : mixes) {
        std::vector<std::vector<std::pair<std::pair<int32_t, int32_t>, uint64_t>>> per_trace;
        std::vector<double> weights;
        std::vector<uint8_t> scored((std::size_t) (n_layers * n_expert), 0);
        std::size_t n_scored = 0, n_used = 0, n_skipped = 0;
        for (const std::string& spec : m.specs) {
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
            if (!strata::core::read_routing_trace(dir + "/" + path, n_layers, n_expert, codes, err)) {
                std::fprintf(stderr, "FAIL mix %s: read_routing_trace refused %s: %s\n", m.name.c_str(),
                             path.c_str(), err.c_str());
                ++failures;
                continue;
            }
            if (codes.empty()) {
                ++n_skipped;   // the reference prints "empty trace, skipped" and moves on
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
            ++n_used;
        }
        const auto ranked = strata::core::rank_profile_mix(per_trace, weights, n_layers, n_expert);
        // the reference reports `len(a.traces)`: every spec it was given, the skipped ones included
        n_used = m.specs.size();
        check("mix count " + m.name, std::to_string(ranked.size()), m.count);
        check("mix_head " + m.name, pairs_to_text({ranked.begin(), ranked.begin() + 12}), m.head);
        check("mix_tail " + m.name, pairs_to_text({ranked.end() - 4, ranked.end()}), m.tail);
        check("mix_stdout " + m.name,
              std::to_string(ranked.size()) + " ranked pairs, " + std::to_string(n_scored) + " from " +
                  std::to_string(n_used) + " traces",
              m.stdout_text);
        check("mix_skipped " + m.name, std::to_string(n_skipped), m.skipped);
        if (!strata::core::write_expert_profile(tmp, n_layers, n_expert, ranked, err)) {
            std::fprintf(stderr, "FAIL write_expert_profile refused the mix %s: %s\n", m.name.c_str(), err.c_str());
            ++failures;
            continue;
        }
        const std::string got = read_text(tmp);
        check("mix " + m.name + " bytes", std::to_string(got.size()), m.bytes);
        if (got != read_text(dir + "/mix-" + m.name + ".bin")) {
            std::fprintf(stderr, "FAIL mix %s differs from the reference\n", m.name.c_str());
            ++failures;
        }
        std::remove(tmp.c_str());
    }

    // ---- and the refusals: a profile that is not STRP, and a trace whose record runs past the file
    {
        const std::string bad = std::string(made) + "/bad.bin";
        std::ofstream(bad, std::ios::binary).write("NOPE\x01\x00\x00\x00", 8);
        std::vector<std::pair<int32_t, int32_t>> r;
        int64_t s = 0;
        if (strata::core::read_expert_profile(bad, n_layers, n_expert, r, s, err)) {
            std::fprintf(stderr, "FAIL a profile that is not STRP was accepted\n");
            ++failures;
        }
        // a record claiming more ids than the file holds: the reference's struct.unpack_from raises here
        const std::string short_trace = std::string(made) + "/short.bin";
        std::ofstream(short_trace, std::ios::binary).write("\x00\x00\x00\x00\x40\x00\x00\x00", 8);
        std::vector<uint32_t> c2;
        if (strata::core::read_routing_trace(short_trace, n_layers, n_expert, c2, err)) {
            std::fprintf(stderr, "FAIL a trace whose record runs past the file was accepted\n");
            ++failures;
        }
        std::remove(bad.c_str());
        std::remove(short_trace.c_str());
    }
    std::filesystem::remove_all(std::string(made));

    if (failures) {
        std::fprintf(stderr, "expert_profile_test: FAILED (%d)\n", failures);
        return 1;
    }
    std::printf("expert_profile_test: the trace reader (order and frequencies), the ranking in all %zu modes and "
                "the weighted mix in all %zu spec sets match the reference, byte for byte\n",
                modes.size(), mixes.size());
    return 0;
}
