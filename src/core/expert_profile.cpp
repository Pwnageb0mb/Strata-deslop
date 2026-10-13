// src/core/expert_profile.cpp - see include/strata/core/expert_profile.hpp for what this owns and why.
#include "strata/core/expert_profile.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <system_error>

namespace strata::core {

// ---------------------------------------------------------------- the profile file ------------------------------
// read_expert_profile and write_expert_profile came from src/core/expert_cache.cpp unchanged.  Only their home
// moved: the format is file I/O, and the GPU-gated library they were in is not linked by a CPU-only tool.

bool read_expert_profile(const std::string& path, int64_t n_layers, int64_t n_expert,
                         std::vector<std::pair<int32_t, int32_t>>& ranked, int64_t& slots, std::string& err) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) {
        err = "read_expert_profile: cannot open " + path;
        return false;
    }
    char magic[4] = {0, 0, 0, 0};
    uint32_t hdr[5] = {0, 0, 0, 0, 0};
    if (std::fread(magic, 1, 4, f) != 4 || std::fread(hdr, 4, 5, f) != 5) {
        std::fclose(f);
        err = "read_expert_profile: " + path + " is too short to hold a header";
        return false;
    }
    if (std::memcmp(magic, "STRP", 4) != 0) {
        std::fclose(f);
        err = "read_expert_profile: " + path + " does not start with STRP";
        return false;
    }
    const uint32_t version = hdr[0], nl = hdr[1], ne = hdr[2], want = hdr[3], n_ranked = hdr[4];
    // the version comes first: the fields after it are only known to be these in a version 1 file, so a later
    // format is refused by its number rather than read as if it were this one
    if (version != 1) {
        std::fclose(f);
        char buf[256];
        std::snprintf(buf, sizeof buf,
                      "read_expert_profile: %s is a version %u profile but this engine reads version 1 - it was "
                      "written by a different release's tools/make_profile.py or --expert-profile-save",
                      path.c_str(), version);
        err = buf;
        return false;
    }
    if ((int64_t) nl != n_layers || (int64_t) ne != n_expert) {
        std::fclose(f);
        char buf[256];
        std::snprintf(buf, sizeof buf,
                      "read_expert_profile: %s is %ux%u but this model is %lldx%lld - it is a profile for a "
                      "different artifact", path.c_str(), nl, ne, (long long) n_layers, (long long) n_expert);
        err = buf;
        return false;
    }
    if (n_ranked > want) {
        std::fclose(f);
        err = "read_expert_profile: the header claims more ranked pairs than slots";
        return false;
    }
    // a profile ranks each (layer, expert) pair at most once, so neither count can exceed the model's pair count;
    // checked before the allocation below, which a corrupt header would otherwise size
    if ((int64_t) want > n_layers * n_expert || (int64_t) n_ranked > n_layers * n_expert) {
        std::fclose(f);
        err = "read_expert_profile: the header claims more slots or ranked pairs than the model has experts";
        return false;
    }
    ranked.assign(n_ranked, {0, 0});
    std::vector<uint16_t> raw((size_t) n_ranked * 2);
    if (n_ranked > 0 && std::fread(raw.data(), 2, (size_t) n_ranked * 2, f) != (size_t) n_ranked * 2) {
        std::fclose(f);
        err = "read_expert_profile: the ranked list is truncated";
        return false;
    }
    std::fclose(f);
    for (uint32_t i = 0; i < n_ranked; ++i) {
        const int32_t l = (int32_t) raw[(size_t) i * 2], e = (int32_t) raw[(size_t) i * 2 + 1];
        if (l < 0 || l >= n_layers || e < 0 || e >= n_expert) {
            char buf[256];
            std::snprintf(buf, sizeof buf, "read_expert_profile: pair %u is (layer %d, expert %d), out of range",
                          i, l, e);
            err = buf;
            return false;
        }
        ranked[(size_t) i] = {l, e};
    }
    slots = (int64_t) want;
    return true;
}

bool write_expert_profile(const std::string& path, int64_t n_layers, int64_t n_expert,
                          const std::vector<std::pair<int32_t, int32_t>>& ranked, std::string& err) {
    if (n_layers <= 0 || n_expert <= 0 || n_layers > 65535 || n_expert > 65535) {
        err = "write_expert_profile: the model's layout does not fit the format";
        return false;
    }
    std::vector<int32_t> table((size_t) (n_layers * n_expert), -1);
    std::vector<uint16_t> pairs;
    pairs.reserve(ranked.size() * 2);
    for (size_t r = 0; r < ranked.size(); ++r) {
        const auto [l, e] = ranked[r];
        if (l < 0 || l >= n_layers || e < 0 || e >= n_expert) {
            err = "write_expert_profile: a ranked pair is out of range";
            return false;
        }
        table[(size_t) (l * n_expert + e)] = (int32_t) r;
        pairs.push_back((uint16_t) l);
        pairs.push_back((uint16_t) e);
    }
    const uint32_t hdr[5] = {1u, (uint32_t) n_layers, (uint32_t) n_expert, (uint32_t) ranked.size(),
                             (uint32_t) ranked.size()};
    const std::string tmp = path + ".tmp";
    std::FILE* f = std::fopen(tmp.c_str(), "wb");
    if (f == nullptr) {
        err = "write_expert_profile: cannot create " + tmp;
        return false;
    }
    // the format is little-endian (make_profile.py's "<"): so is every machine this engine runs on
    bool ok = std::fwrite("STRP", 1, 4, f) == 4 && std::fwrite(hdr, 4, 5, f) == 5 &&
              (pairs.empty() || std::fwrite(pairs.data(), 2, pairs.size(), f) == pairs.size()) &&
              std::fwrite(table.data(), 4, table.size(), f) == table.size();
    ok = (std::fclose(f) == 0) && ok;
    std::error_code ec;
    if (ok) std::filesystem::rename(tmp, path, ec);   // replaces an existing file (MoveFileEx / rename(2))
    if (!ok || ec) {
        std::filesystem::remove(tmp, ec);
        err = "write_expert_profile: cannot write " + path;
        return false;
    }
    return true;
}

std::vector<std::pair<int32_t, int32_t>> rank_learned_profile(int64_t n_layers, int64_t n_expert,
                                                              const std::vector<uint8_t>& resident,
                                                              const std::vector<double>& heat,
                                                              const std::vector<std::pair<int32_t, int32_t>>& prior) {
    const size_t n = (size_t) (n_layers * n_expert);
    std::vector<int64_t> prior_rank(n, INT64_MAX);
    for (size_t r = 0; r < prior.size(); ++r) {
        const auto [l, e] = prior[r];
        if (l >= 0 && l < n_layers && e >= 0 && e < n_expert) {
            int64_t& pr = prior_rank[(size_t) (l * n_expert + e)];
            if (pr == INT64_MAX) pr = (int64_t) r;
        }
    }
    std::vector<int64_t> order(n);
    for (size_t i = 0; i < n; ++i) order[i] = (int64_t) i;
    auto res = [&](int64_t i) { return (size_t) i < resident.size() && resident[(size_t) i] != 0; };
    auto ht = [&](int64_t i) { return (size_t) i < heat.size() ? heat[(size_t) i] : 0.0; };
    std::stable_sort(order.begin(), order.end(), [&](int64_t a, int64_t b) {
        if (res(a) != res(b)) return res(a);
        if (ht(a) != ht(b)) return ht(a) > ht(b);
        if (prior_rank[(size_t) a] != prior_rank[(size_t) b]) return prior_rank[(size_t) a] < prior_rank[(size_t) b];
        return a < b;
    });
    std::vector<std::pair<int32_t, int32_t>> ranked(n);
    for (size_t r = 0; r < n; ++r)
        ranked[r] = {(int32_t) (order[r] / n_expert), (int32_t) (order[r] % n_expert)};
    return ranked;
}

void profile_slot_range(int64_t layer, int64_t slots, int64_t n_layers, int64_t& lo, int64_t& hi) {
    lo = 0;
    hi = 0;
    if (n_layers <= 0 || slots <= 0 || layer < 0 || layer >= n_layers) return;
    const int64_t q = slots / n_layers;
    lo = layer * q;
    hi = (layer == n_layers - 1) ? slots : (layer + 1) * q;
}

// ---------------------------------------------------------------- the routing trace ----------------------------

uint32_t trace_encode(int32_t layer, int32_t expert) { return ((uint32_t) layer << 16) | (uint32_t) expert; }

std::pair<int32_t, int32_t> trace_decode(uint32_t code) {
    return {(int32_t) (code >> 16), (int32_t) (code & 0xFFFF)};
}

bool read_routing_trace(const std::string& path, int64_t n_layers, int64_t n_expert,
                        std::vector<uint32_t>& codes, std::string& err) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) {
        err = "read_routing_trace: cannot open " + path;
        return false;
    }
    std::vector<uint8_t> blob;
    {
        std::fseek(f, 0, SEEK_END);
        const long end = std::ftell(f);
        std::fseek(f, 0, SEEK_SET);
        if (end < 0) {
            std::fclose(f);
            err = "read_routing_trace: cannot size " + path;
            return false;
        }
        blob.resize((size_t) end);
        if (!blob.empty() && std::fread(blob.data(), 1, blob.size(), f) != blob.size()) {
            std::fclose(f);
            err = "read_routing_trace: " + path + " is shorter than it claims";
            return false;
        }
    }
    std::fclose(f);

    const uint8_t* p = blob.data();
    const size_t n = blob.size();
    size_t off = 0;
    while (off + 8 <= n) {
        int32_t rec[2] = {0, 0};
        std::memcpy(rec, p + off, 8);
        const int32_t layer = rec[0], k = rec[1];
        off += 8;
        if (k < 0 || (uint64_t) k > (n - off) / 8) {
            // the reference's struct.unpack_from raises here and the tool dies: a record that runs past the file
            // is a corrupt trace, not a short one, and reading it as if it ended would silently drop the routing
            // a policy is being measured on
            char buf[256];
            std::snprintf(buf, sizeof buf, "read_routing_trace: %s: a record at byte %zu claims k=%d, past the "
                                           "end of the file", path.c_str(), off - 8, k);
            err = buf;
            return false;
        }
        for (int32_t i = 0; i < k; ++i) {
            int32_t e = 0;
            std::memcpy(&e, p + off + (size_t) i * 4, 4);
            if (layer >= 0 && layer < n_layers && e >= 0 && e < n_expert)
                codes.push_back(trace_encode(layer, e));
        }
        off += (size_t) k * 8;   // the ids and the weights
    }
    return true;
}

std::vector<std::pair<std::pair<int32_t, int32_t>, uint64_t>> trace_frequencies(const std::vector<uint32_t>& codes) {
    // the reference keys a defaultdict by pair; the pairs are bounded by the model's own dimensions, so a sorted
    // vector of what was actually routed is both smaller and already in the order the ranking's tie-break needs
    std::vector<std::pair<int32_t, int32_t>> keys;
    keys.reserve(codes.size());
    for (uint32_t code : codes) keys.push_back(trace_decode(code));
    std::sort(keys.begin(), keys.end());
    keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
    std::vector<uint64_t> counts(keys.size(), 0);
    for (uint32_t code : codes) {
        const auto p = trace_decode(code);
        const auto it = std::lower_bound(keys.begin(), keys.end(), p);
        counts[(size_t) (it - keys.begin())] += 1;
    }
    std::vector<std::pair<std::pair<int32_t, int32_t>, uint64_t>> out;
    out.reserve(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) out.push_back({keys[i], counts[i]});
    return out;
}

std::vector<std::pair<int32_t, int32_t>> rank_profile(
    const std::vector<std::pair<int32_t, int32_t>>& base_pairs,
    const std::vector<std::pair<std::pair<int32_t, int32_t>, uint64_t>>& trace_freq, int64_t n_layers,
    int64_t n_expert, bool no_base, bool reorder, int64_t counts[3]) {
    if (counts) counts[0] = counts[1] = counts[2] = 0;
    std::vector<std::pair<int32_t, int32_t>> ranked;
    ranked.reserve((size_t) (n_layers * n_expert));
    std::vector<uint8_t> seen((size_t) (n_layers * n_expert), 0);
    auto take = [&](const std::pair<int32_t, int32_t>& p, int which) {
        if (p.first < 0 || p.first >= n_layers || p.second < 0 || p.second >= n_expert) return;
        const size_t i = (size_t) (p.first * n_expert + p.second);
        if (seen[i]) return;
        seen[i] = 1;
        ranked.push_back(p);
        if (counts) ++counts[which];
    };

    // the traces' pairs, most frequent first; a tie by (layer, expert) ascending, which is the order
    // trace_frequencies returns them in and stable_sort preserves
    std::vector<size_t> idx(trace_freq.size());
    for (size_t i = 0; i < idx.size(); ++i) idx[i] = i;
    std::stable_sort(idx.begin(), idx.end(),
                     [&](size_t a, size_t b) { return trace_freq[a].second > trace_freq[b].second; });
    std::vector<std::pair<int32_t, int32_t>> trace_pairs;
    trace_pairs.reserve(trace_freq.size());
    for (size_t i : idx) trace_pairs.push_back(trace_freq[i].first);

    if (no_base || reorder)
        for (const auto& p : trace_pairs) take(p, 1);
    if (!no_base)
        for (const auto& p : base_pairs) take(p, 0);
    if (!(no_base || reorder))
        for (const auto& p : trace_pairs) take(p, 1);
    for (int64_t e = 0; e < n_expert; ++e)
        for (int64_t l = 0; l < n_layers; ++l) take({(int32_t) l, (int32_t) e}, 2);
    return ranked;
}

std::vector<std::pair<int32_t, int32_t>> rank_profile_mix(
    const std::vector<std::vector<std::pair<std::pair<int32_t, int32_t>, uint64_t>>>& per_trace_freq,
    const std::vector<double>& weights, int64_t n_layers, int64_t n_expert) {
    const std::size_t n = (std::size_t) (n_layers * n_expert);
    std::vector<double> score(n, 0.0);
    std::vector<uint8_t> present(n, 0);
    for (std::size_t t = 0; t < per_trace_freq.size(); ++t) {
        uint64_t total = 0;
        for (const auto& [_, c] : per_trace_freq[t]) total += c;
        if (total == 0) continue;   // an empty trace: the reference says so and moves on
        const double weight = t < weights.size() ? weights[t] : 1.0;
        for (const auto& [pair, c] : per_trace_freq[t]) {
            if (pair.first < 0 || pair.first >= n_layers || pair.second < 0 || pair.second >= n_expert) continue;
            const std::size_t i = (std::size_t) (pair.first * n_expert + pair.second);
            // `weight * c / total`, left to right as the reference writes it: the association decides the last
            // bit, and the last bit decides the order of two pairs whose scores are nearly equal
            score[i] += weight * (double) c / (double) total;
            present[i] = 1;
        }
    }
    std::vector<std::pair<int32_t, int32_t>> ranked;
    ranked.reserve(n);
    std::vector<size_t> idx;
    for (std::size_t i = 0; i < n; ++i)
        if (present[i]) idx.push_back(i);
    // ascending pair order first (the loop above), then a stable sort on the score: a tie by (layer, expert)
    std::stable_sort(idx.begin(), idx.end(),
                     [&](std::size_t a, std::size_t b) { return score[a] > score[b]; });
    std::vector<uint8_t> seen(n, 0);
    for (std::size_t i : idx) {
        seen[i] = 1;
        ranked.push_back({(int32_t) (i / (std::size_t) n_expert), (int32_t) (i % (std::size_t) n_expert)});
    }
    // the fill, in rank_profile's order: the pairs no trace used
    for (int64_t e = 0; e < n_expert; ++e)
        for (int64_t l = 0; l < n_layers; ++l) {
            const std::size_t i = (std::size_t) (l * n_expert + e);
            if (!seen[i]) ranked.push_back({(int32_t) l, (int32_t) e});
        }
    return ranked;
}

} // namespace strata::core
