// src/core/cache_replay.cpp - see include/strata/core/cache_replay.hpp for what this owns and why.
#include "strata/core/cache_replay.hpp"

#include <algorithm>
#include <charconv>
#include <cstdio>
#include <cstring>

namespace strata::core {
namespace {

bool read_int32_file(const std::string& path, std::vector<int32_t>& out, std::string& err) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) {
        err = "cannot open " + path;
        return false;
    }
    std::fseek(f, 0, SEEK_END);
    const long end = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    if (end < 0) {
        std::fclose(f);
        err = "cannot size " + path;
        return false;
    }
    std::vector<uint8_t> blob((size_t) end);
    if (!blob.empty() && std::fread(blob.data(), 1, blob.size(), f) != blob.size()) {
        std::fclose(f);
        err = path + " is shorter than it claims";
        return false;
    }
    std::fclose(f);
    // np.fromfile(dtype=int32): a trailing partial word is not a record, so it is dropped
    out.resize(blob.size() / 4);
    if (!out.empty()) std::memcpy(out.data(), blob.data(), out.size() * 4);
    return true;
}

} // namespace

bool read_replay_trace(const std::string& path, int64_t n_layers, int64_t n_expert,
                       std::vector<ReplayWindow>& windows, int64_t& k_out, std::string& err) {
    std::vector<int32_t> raw;
    if (!read_int32_file(path, raw, err)) return false;
    if (raw.size() < 2) {
        err = "read_replay_trace: " + path + " is too short to hold a record header";
        return false;
    }
    // k comes from the FIRST record's second word, and every record is then assumed to be that wide
    const int32_t k = raw[1];
    if (k < 0 || (uint64_t) k > (raw.size() - 2) / 2) {
        err = "read_replay_trace: " + path + " claims k=" + std::to_string(k) + ", which does not fit the file";
        return false;
    }
    const std::size_t rec = 2 + 2 * (std::size_t) k;
    const std::size_t n_rec = raw.size() / rec;   // the reference truncates to whole records
    k_out = k;

    // the layer groups: a run of records that routed the same layer
    std::vector<ReplayWindow> all;
    ReplayWindow cur;
    bool open = false;
    int32_t last = -1;
    for (std::size_t i = 0; i < n_rec;) {
        std::size_t j = i + 1;
        while (j < n_rec && raw[j * rec] == raw[i * rec]) ++j;
        const int32_t layer = raw[i * rec];
        if (layer < 0 || layer >= n_layers) {
            err = "read_replay_trace: " + path + ": a record routes layer " + std::to_string(layer) +
                  ", outside 0.." + std::to_string(n_layers - 1) + " (the reference indexes with it)";
            return false;
        }
        if (!open || layer <= last) {   // a window is a run of groups whose layer RISES
            if (open) all.push_back(std::move(cur));
            cur = ReplayWindow{};
            open = true;
        }
        ReplayGroup g;
        g.layer = layer;
        g.n_records = (int64_t) (j - i);
        g.ids.reserve((j - i) * (std::size_t) k);
        for (std::size_t r = i; r < j; ++r)
            for (int32_t c = 0; c < k; ++c) {
                const int32_t id = raw[r * rec + 2 + (std::size_t) c];
                if (id < 0 || id >= n_expert) {
                    err = "read_replay_trace: " + path + ": a record routes expert " + std::to_string(id) +
                          ", outside 0.." + std::to_string(n_expert - 1) + " (the reference indexes with it)";
                    return false;
                }
                g.ids.push_back(id);
            }
        cur.groups.push_back(std::move(g));
        last = layer;
        i = j;
    }
    if (open) all.push_back(std::move(cur));

    windows.clear();
    for (ReplayWindow& w : all)
        if ((int64_t) w.groups.size() == n_layers) windows.push_back(std::move(w));
    return true;
}

int64_t replay_adapt_round(std::vector<uint8_t>& res, std::vector<double>& usage, int64_t n_layers,
                           int64_t n_expert, const ReplayPolicy& pol) {
    struct Swap {
        double gain;
        int32_t layer, in, out;
    };
    std::vector<Swap> swaps;
    for (int64_t l = 0; l < n_layers; ++l) {
        const double* u = usage.data() + (std::size_t) (l * n_expert);
        const uint8_t* r = res.data() + (std::size_t) (l * n_expert);
        std::vector<int32_t> cand, vict;
        for (int32_t e = 0; e < (int32_t) n_expert; ++e) {
            if (!r[e]) {
                if (u[e] >= pol.mincand) cand.push_back(e);
            } else {
                vict.push_back(e);
            }
        }
        if (cand.empty() || vict.empty()) continue;
        // STABLE by usage: experts with equal usage keep the ascending expert order, which is what the reference's
        // `np.argsort(..., kind="stable")` gives and what the engine's `std::sort` does NOT
        std::stable_sort(cand.begin(), cand.end(), [&](int32_t a, int32_t b) { return u[a] > u[b]; });
        std::stable_sort(vict.begin(), vict.end(), [&](int32_t a, int32_t b) { return u[a] < u[b]; });
        const std::size_t n = std::min(cand.size(), vict.size());
        for (std::size_t i = 0; i < n; ++i) {
            // the engine stops at the first pair that does not clear the margin, and so does this
            if (u[cand[i]] < u[vict[i]] + pol.margin) break;
            swaps.push_back({u[cand[i]] - u[vict[i]], (int32_t) l, cand[i], vict[i]});
        }
    }
    std::stable_sort(swaps.begin(), swaps.end(), [](const Swap& a, const Swap& b) { return a.gain > b.gain; });
    std::size_t kept = 0;
    for (const Swap& s : swaps) {
        if (s.gain < pol.mingain) continue;
        if ((int64_t) kept >= pol.swaps) break;
        res[(std::size_t) (s.layer * n_expert + s.in)] = 1;
        res[(std::size_t) (s.layer * n_expert + s.out)] = 0;
        ++kept;
    }
    return (int64_t) kept;
}

ReplayStats replay_simulate(const std::vector<ReplayWindow>& windows,
                            const std::vector<std::pair<int32_t, int32_t>>& init, int64_t n_layers,
                            int64_t n_expert, const ReplayPolicy& pol) {
    const std::size_t n = (std::size_t) (n_layers * n_expert);
    std::vector<uint8_t> res(n, 0);
    for (const auto& [l, e] : init)
        if (l >= 0 && l < n_layers && e >= 0 && e < n_expert) res[(std::size_t) (l * n_expert + e)] = 1;
    std::vector<double> usage(n, 0.0);

    ReplayStats st;
    st.windows = (int64_t) windows.size();
    std::vector<int32_t> misses;
    for (std::size_t wi = 0; wi < windows.size(); ++wi) {
        for (const ReplayGroup& g : windows[wi].groups) {
            const std::size_t base = (std::size_t) ((int64_t) g.layer * n_expert);
            misses.clear();
            for (int32_t e : g.ids) {
                ++st.entries;
                if (res[base + (std::size_t) e]) {
                    ++st.hits;
                } else {
                    ++st.miss_ent;
                    misses.push_back(e);
                }
                usage[base + (std::size_t) e] += 1.0;
            }
            // `miss_dist` counts the DISTINCT misses of this group, which is the number of distinct reads the
            // file tier would do - the reference's `len(np.unique(ms))`
            std::sort(misses.begin(), misses.end());
            st.miss_dist += (int64_t) (std::unique(misses.begin(), misses.end()) - misses.begin());
        }
        if (pol.every > 0 && (int64_t) (wi + 1) % pol.every == 0) {
            st.swaps += replay_adapt_round(res, usage, n_layers, n_expert, pol);
            for (double& v : usage) v *= pol.decay;
        }
    }
    st.share = (double) st.hits / (double) std::max<int64_t>(1, st.entries);
    return st;
}

double replay_static_oracle(const std::vector<ReplayWindow>& windows, int64_t n_layers, int64_t n_expert,
                            int64_t slots) {
    const std::size_t n = (std::size_t) (n_layers * n_expert);
    std::vector<int64_t> cnt(n, 0);
    for (const ReplayWindow& w : windows)
        for (const ReplayGroup& g : w.groups) {
            const std::size_t base = (std::size_t) ((int64_t) g.layer * n_expert);
            for (int32_t e : g.ids) cnt[base + (std::size_t) e] += 1;
        }
    // the reference argsorts the FLATTENED array, so a tie keeps the ascending (layer, expert) order: stable over
    // an index-ascending vector is the same thing
    std::vector<std::size_t> idx;
    int64_t total = 0;
    for (std::size_t i = 0; i < n; ++i) {
        total += cnt[i];
        if (cnt[i] > 0) idx.push_back(i);
    }
    std::stable_sort(idx.begin(), idx.end(), [&](std::size_t a, std::size_t b) { return cnt[a] > cnt[b]; });
    int64_t top = 0;
    for (std::size_t i = 0; i < idx.size() && (int64_t) i < slots; ++i) top += cnt[idx[i]];
    return (double) top / (double) std::max<int64_t>(1, total);
}

std::string replay_py_float(double v) {
    char buf[48];
    const auto r = std::to_chars(buf, buf + sizeof buf, v);
    std::string s(buf, r.ptr);
    if (s.find_first_of(".eE") == std::string::npos) s += ".0";
    return s;
}

std::string replay_fixed(double v, int places) {
    char buf[64];
    std::snprintf(buf, sizeof buf, "%.*f", places, v);
    return buf;
}

std::string replay_report(const std::vector<ReplayWindow>& windows,
                          const std::vector<std::pair<int32_t, int32_t>>& init, int64_t slots, int64_t k,
                          int64_t n_layers, int64_t n_expert, const ReplayPolicy& pol) {
    double first_groups = 0.0;
    for (const ReplayWindow& w : windows) first_groups += (double) w.groups.front().n_records;

    ReplayPolicy no_adapt = pol;
    no_adapt.every = 1000000000;   // the reference's `Policy(every=10**9)`: the round never fires
    const ReplayStats base = replay_simulate(windows, init, n_layers, n_expert, no_adapt);
    const ReplayStats cur = replay_simulate(windows, init, n_layers, n_expert, pol);

    std::string o;
    o += "windows " + std::to_string(windows.size()) + " k " + std::to_string(k) + " tokens/window " +
         replay_py_float(first_groups / (double) windows.size()) + "\n";
    o += "static oracle (top-" + std::to_string(slots) + " by this trace's own counts): " +
         replay_fixed(replay_static_oracle(windows, n_layers, n_expert, slots), 4) + "\n";
    o += "profile only, no adaptation: " + replay_fixed(base.share, 4) + "\n";
    o += "policy every " + std::to_string(pol.every) + ", swaps " + std::to_string(pol.swaps) + ", decay " +
         replay_fixed(pol.decay, 2) + ": share " + replay_fixed(cur.share, 4) + ", swaps " +
         std::to_string(cur.swaps) + " (" + replay_fixed((double) cur.swaps / (double) cur.windows, 2) +
         "/window); miss entries " + std::to_string(cur.miss_ent) + ", distinct misses " +
         std::to_string(cur.miss_dist) + " (" +
         replay_fixed(100.0 * (1.0 - (double) cur.miss_dist / (double) std::max<int64_t>(1, cur.miss_ent)), 1) +
         "% fewer reads)\n";
    return o;
}

} // namespace strata::core
