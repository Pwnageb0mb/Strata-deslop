// src/core/eviction_sim.cpp - see include/strata/core/eviction_sim.hpp for what this owns and why.
//
// A policy's hit rate depends only on the ORDER the run routed pairs, so a trace replays through a simulated slot
// pool with no GPU.  Each policy below is the reference's, including its tie-breaks: the counts are the same, the
// victims are the same, and a trace that ties is where a port that "cleaned up" the comparison would diverge.
#include "strata/core/eviction_sim.hpp"

#include <algorithm>
#include <cstdio>
#include <functional>
#include <list>
#include <memory>
#include <queue>
#include <tuple>
#include <unordered_map>
#include <unordered_set>

#include "strata/core/expert_profile.hpp"

namespace strata::core {
namespace {

std::string ljust(const std::string& s, std::size_t n) {
    return s.size() >= n ? s : s + std::string(n - s.size(), ' ');
}
std::string rjust(const std::string& s, std::size_t n) {
    return s.size() >= n ? s : std::string(n - s.size(), ' ') + s;
}

std::string fmt_rate(int64_t hits, int64_t lookups) {
    if (lookups == 0) return "n/a";
    char buf[32];
    std::snprintf(buf, sizeof buf, "%.4f", (double) hits / (double) lookups);
    return buf;
}

/// One policy's slot pool.  `lookup(code, i)` answers hit/miss for the i-th lookup of the trace and admits or
/// evicts by the policy; `hits` and `evictions` accumulate.
struct Sim {
    int64_t n_layers, n_expert, slots;
    bool per_layer;
    int64_t hits = 0, evictions = 0;

    Sim(int64_t nl, int64_t ne, int64_t s, bool pl) : n_layers(nl), n_expert(ne), slots(s), per_layer(pl) {}
    virtual ~Sim() = default;

    int64_t scope(int32_t layer) const { return per_layer ? (int64_t) layer : 0; }

    /// The slots a lookup on `layer` may use.  In a GLOBAL scope that is the whole pool - not the pool divided by
    /// the layer count: `scope()` maps every layer to one pool, and dividing here would starve every layer of a
    /// small pool (4 slots over 48 layers is 0 each, and an eviction policy then evicts from an empty cache).  The
    /// per-layer branch is the same rule the cache uses, so the simulation and the engine cannot disagree about
    /// what a per-layer scope means.
    int64_t capacity(int32_t layer) const {
        if (!per_layer) return slots;
        int64_t lo = 0, hi = 0;
        profile_slot_range(layer, slots, n_layers, lo, hi);
        return hi - lo;
    }

    virtual void prepare(const std::vector<uint32_t>&) {}
    virtual void prefill(const std::vector<std::pair<int32_t, int32_t>>&) = 0;
    virtual bool lookup(uint32_t code, int64_t i) = 0;
};

/// The engine's current admission: the first S distinct pairs stay resident forever.
struct FillOnly : Sim {
    using Sim::Sim;
    std::unordered_set<uint32_t> resident;
    std::unordered_map<int64_t, int64_t> used;

    void prefill(const std::vector<std::pair<int32_t, int32_t>>& pairs) override {
        for (const auto& [l, e] : pairs) {
            const uint32_t code = trace_encode(l, e);
            const int64_t s = scope(l);
            if (resident.find(code) == resident.end() && used[s] < capacity(l)) {
                resident.insert(code);
                ++used[s];
            }
        }
    }

    bool lookup(uint32_t code, int64_t) override {
        const auto [layer, _] = trace_decode(code);
        if (resident.find(code) != resident.end()) {
            ++hits;
            return true;
        }
        const int64_t s = scope(layer);
        if (used[s] < capacity(layer)) {
            resident.insert(code);
            ++used[s];
        }
        return false;
    }
};

/// Evict the least-recently-used resident of the scope.  A list with the oldest at the front is the reference's
/// OrderedDict; `splice` is its move_to_end.
struct Lru : Sim {
    using Sim::Sim;
    std::unordered_map<int64_t, std::list<uint32_t>> od;
    std::unordered_map<int64_t, std::unordered_map<uint32_t, std::list<uint32_t>::iterator>> where;

    void prefill(const std::vector<std::pair<int32_t, int32_t>>& pairs) override {
        // each scope's top pairs in RANKED order, then inserted reversed so rank 0 lands at the MRU end
        std::vector<std::pair<uint32_t, int64_t>> chosen;
        std::unordered_map<int64_t, int64_t> used;
        for (const auto& [l, e] : pairs) {
            const int64_t s = scope(l);
            if (used[s] < capacity(l)) {
                chosen.push_back({trace_encode(l, e), s});
                ++used[s];
            }
        }
        for (auto it = chosen.rbegin(); it != chosen.rend(); ++it) {
            auto& lst = od[it->second];
            lst.push_back(it->first);
            where[it->second][it->first] = std::prev(lst.end());
        }
    }

    bool lookup(uint32_t code, int64_t) override {
        const auto [layer, _] = trace_decode(code);
        const int64_t s = scope(layer);
        auto& lst = od[s];
        auto& w = where[s];
        const auto it = w.find(code);
        if (it != w.end()) {
            lst.splice(lst.end(), lst, it->second);
            it->second = std::prev(lst.end());
            ++hits;
            return true;
        }
        if ((int64_t) lst.size() >= capacity(layer)) {
            w.erase(lst.front());
            lst.pop_front();
            ++evictions;
        }
        lst.push_back(code);
        w[code] = std::prev(lst.end());
        return false;
    }
};

/// Evict the lowest decayed-frequency resident of the scope, oldest on a tie.
///
/// Every lookup adds the current increment to the pair's count; the increment doubles every `decay_window`
/// lookups, which is the table-wide halving of classic LFU-decay without touching the table (halving the table is
/// O(slots) per window; doubling new increments scales every past access by the same 1/2).  Comparisons stay
/// exact: counts are sums of powers of two, below 2^53 for any realistic trace.  Victim selection is a lazy heap.
struct LfuDecay : Sim {
    int64_t decay_window, tick = 0;
    double incr = 1.0;
    std::unordered_map<uint32_t, double> counts;
    std::unordered_map<int64_t, std::priority_queue<std::tuple<double, int64_t, uint32_t>,
                                                    std::vector<std::tuple<double, int64_t, uint32_t>>,
                                                    std::greater<>>> heaps;
    std::unordered_map<int64_t, int64_t> used;

    LfuDecay(int64_t nl, int64_t ne, int64_t s, bool pl, int64_t dw)
        : Sim(nl, ne, s, pl), decay_window(dw) {}

    void prefill(const std::vector<std::pair<int32_t, int32_t>>& pairs) override {
        std::vector<std::pair<uint32_t, int64_t>> chosen;
        std::unordered_map<int64_t, int64_t> u;
        for (const auto& [l, e] : pairs) {
            const int64_t s = scope(l);
            if (u[s] < capacity(l)) {
                chosen.push_back({trace_encode(l, e), s});
                ++u[s];
            }
        }
        for (auto it = chosen.rbegin(); it != chosen.rend(); ++it) {
            const uint32_t code = it->first;
            const int64_t s = it->second;
            if (counts.find(code) == counts.end()) {
                counts[code] = incr;
                heaps[s].push({incr, tick, code});
                ++used[s];
                ++tick;
            }
        }
    }

    bool lookup(uint32_t code, int64_t) override {
        if (tick > 0 && tick % decay_window == 0) incr *= 2.0;
        const int64_t t = tick++;
        const auto [layer, _] = trace_decode(code);
        const int64_t s = scope(layer);
        const auto it = counts.find(code);
        if (it != counts.end()) {              // resident (an evicted pair's count is deleted with it)
            it->second += incr;
            heaps[s].push({it->second, t, code});
            ++hits;
            return true;
        }
        if (used[s] >= capacity(layer)) {
            auto& h = heaps[s];
            while (!h.empty()) {               // drop stale heap entries until the true minimum surfaces
                const auto& top = h.top();
                const auto ci = counts.find(std::get<2>(top));
                if (ci != counts.end() && ci->second == std::get<0>(top)) break;
                h.pop();
            }
            const uint32_t victim = std::get<2>(h.top());
            h.pop();
            counts.erase(victim);
            --used[s];
            ++evictions;
        }
        counts[code] = incr;
        heaps[s].push({incr, t, code});
        ++used[s];
        return false;
    }
};

/// Belady's MIN: evict the resident whose next use is furthest away.  Needs the whole trace up front, which is
/// exactly why it is a bound and not a candidate.
struct Oracle : Sim {
    using Sim::Sim;
    std::vector<uint32_t> nxt;
    std::unordered_map<uint32_t, int64_t> first, cur;
    std::unordered_map<int64_t, std::priority_queue<std::pair<int64_t, uint32_t>,
                                                    std::vector<std::pair<int64_t, uint32_t>>, std::greater<>>>
        heaps;
    std::unordered_map<int64_t, int64_t> used;
    int64_t T = 0;

    void prepare(const std::vector<uint32_t>& codes) override {
        T = (int64_t) codes.size();
        nxt.assign((std::size_t) T, (uint32_t) T);
        for (int64_t i = T - 1; i >= 0; --i) {
            const uint32_t c = codes[(std::size_t) i];
            const auto it = first.find(c);
            nxt[(std::size_t) i] = it == first.end() ? (uint32_t) T : (uint32_t) it->second;
            first[c] = i;                      // and after the loop this map is the FIRST occurrence of each
        }
    }

    void prefill(const std::vector<std::pair<int32_t, int32_t>>& pairs) override {
        for (const auto& [l, e] : pairs) {
            const uint32_t code = trace_encode(l, e);
            const int64_t s = scope(l);
            if (cur.find(code) == cur.end() && used[s] < capacity(l)) {
                const auto it = first.find(code);
                cur[code] = it == first.end() ? T : it->second;
                heaps[s].push({-cur[code], code});
                ++used[s];
            }
        }
    }

    bool lookup(uint32_t code, int64_t i) override {
        const auto [layer, _] = trace_decode(code);
        const int64_t s = scope(layer);
        const int64_t nx = (int64_t) nxt[(std::size_t) i];
        const auto it = cur.find(code);
        if (it != cur.end()) {
            it->second = nx;
            heaps[s].push({-nx, code});
            ++hits;
            return true;
        }
        if (used[s] >= capacity(layer)) {
            auto& h = heaps[s];
            while (!h.empty()) {
                const auto& top = h.top();
                const auto ci = cur.find(top.second);
                if (ci != cur.end() && ci->second == -top.first) break;
                h.pop();
            }
            const uint32_t victim = h.top().second;
            h.pop();
            cur.erase(victim);
            --used[s];
            ++evictions;
        }
        cur[code] = nx;
        heaps[s].push({-nx, code});
        ++used[s];
        return false;
    }
};

std::unique_ptr<Sim> make_sim(const std::string& name, int64_t n_layers, int64_t n_expert, int64_t slots,
                              bool per_layer, int64_t decay_window) {
    if (name == "fill-only") return std::make_unique<FillOnly>(n_layers, n_expert, slots, per_layer);
    if (name == "lru") return std::make_unique<Lru>(n_layers, n_expert, slots, per_layer);
    if (name == "lfu-decay") return std::make_unique<LfuDecay>(n_layers, n_expert, slots, per_layer, decay_window);
    if (name == "oracle") return std::make_unique<Oracle>(n_layers, n_expert, slots, per_layer);
    return nullptr;
}

} // namespace

const std::vector<std::string>& eviction_policy_names() {
    static const std::vector<std::string> names = {"fill-only", "lru", "lfu-decay", "oracle"};
    return names;
}

std::string eviction_config_error(int64_t n_layers, int64_t slots, bool per_layer, const std::string& policy) {
    if (policy == "fill-only") return {};   // never evicts, so a capacity of 0 is simply a cache that stays empty
    int64_t lo = 0, hi = 0;
    // the smallest capacity any layer has: in a global scope it is the whole pool, so only a per-layer scope can
    // starve a layer, and only the division can do it
    profile_slot_range(0, slots, per_layer ? n_layers : 1, lo, hi);
    if (hi - lo > 0) return {};
    char buf[256];
    std::snprintf(buf, sizeof buf,
                  "policy %s cannot be replayed at %lld slot(s) with a per-layer scope: each layer would have "
                  "none, and the reference crashes on that (its Lru pops an empty OrderedDict). Use --slots >= "
                  "%lld, a global scope, or fill-only",
                  policy.c_str(), (long long) slots, (long long) (per_layer ? n_layers : 1));
    return buf;
}

std::vector<EvictionRow> eviction_run(const std::vector<uint32_t>& codes, int64_t n_layers, int64_t n_expert,
                                      const std::vector<int64_t>& slots_list, bool per_layer,
                                      const std::vector<std::string>& policies, int64_t windows,
                                      int64_t decay_window,
                                      const std::vector<std::pair<int32_t, int32_t>>* prefill) {
    const int64_t T = (int64_t) codes.size();
    std::vector<EvictionRow> rows;
    for (int64_t slots : slots_list) {
        for (const std::string& name : policies) {
            std::unique_ptr<Sim> sim = make_sim(name, n_layers, n_expert, slots, per_layer, decay_window);
            if (!sim) continue;
            if (name == "oracle") sim->prepare(codes);
            if (prefill != nullptr) sim->prefill(*prefill);
            std::vector<int64_t> w_hits((std::size_t) windows, 0), w_tot((std::size_t) windows, 0);
            for (int64_t i = 0; i < T; ++i) {
                const int64_t w = T ? i * windows / T : 0;
                ++w_tot[(std::size_t) w];
                if (sim->lookup(codes[(std::size_t) i], i)) ++w_hits[(std::size_t) w];
            }
            rows.push_back({slots, name, sim->hits, T, sim->evictions, std::move(w_hits), std::move(w_tot)});
        }
    }
    return rows;
}

std::string eviction_report(const std::vector<EvictionRow>& rows, int64_t windows) {
    std::string o;
    std::vector<int64_t> seen;
    for (const EvictionRow& r : rows)
        if (std::find(seen.begin(), seen.end(), r.slots) == seen.end()) seen.push_back(r.slots);
    for (int64_t slots : seen) {
        o += "\nslots = " + std::to_string(slots) + "\n";
        o += "  " + ljust("policy", 10) + " " + rjust("hits", 10) + " " + rjust("lookups", 10) + " " +
             rjust("hit-rate", 9) + " " + rjust("evictions", 10) + "\n";
        for (const EvictionRow& r : rows) {
            if (r.slots != slots) continue;
            o += "  " + ljust(r.policy, 10) + " " + rjust(std::to_string(r.hits), 10) + " " +
                 rjust(std::to_string(r.lookups), 10) + " " + rjust(fmt_rate(r.hits, r.lookups), 9) + " " +
                 rjust(std::to_string(r.evictions), 10) + "\n";
        }
        if (windows > 1) {
            o += "  per-window hit-rate (" + std::to_string(windows) + " windows, in trace order):\n";
            o += "  " + ljust("policy", 10);
            for (int64_t i = 0; i < windows; ++i) o += " " + rjust("w" + std::to_string(i + 1), 8);
            o += "\n";
            for (const EvictionRow& r : rows) {
                if (r.slots != slots) continue;
                o += "  " + ljust(r.policy, 10);
                for (std::size_t w = 0; w < r.w_tot.size(); ++w)
                    o += " " + rjust(fmt_rate(r.w_hits[w], r.w_tot[w]), 8);
                o += "\n";
            }
        }
    }
    return o;
}

std::string eviction_csv(const std::vector<EvictionRow>& rows, const std::string& scope) {
    std::string o = "slots,scope,policy,window,lookups,hits,hit_rate,evictions\n";
    for (const EvictionRow& r : rows) {
        o += std::to_string(r.slots) + "," + scope + "," + r.policy + ",all," + std::to_string(r.lookups) + "," +
             std::to_string(r.hits) + "," + fmt_rate(r.hits, r.lookups) + "," + std::to_string(r.evictions) + "\n";
        for (std::size_t w = 0; w < r.w_tot.size(); ++w)
            if (r.w_tot.size() > 1)
                o += std::to_string(r.slots) + "," + scope + "," + r.policy + ",w" + std::to_string(w + 1) + "," +
                     std::to_string(r.w_tot[w]) + "," + std::to_string(r.w_hits[w]) + "," +
                     fmt_rate(r.w_hits[w], r.w_tot[w]) + ",\n";
    }
    return o;
}

} // namespace strata::core
