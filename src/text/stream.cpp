// src/text/stream.cpp - the C ABI of include/strata/text/strata_stream.h: Detokenizer, StopMatcher, the OutputParser
// glue and reasoning_repeat_coverage.  Python references: serve/server.py and serve/frontend.py.
#include "strata/text/strata_stream.h"

#include "output_parser.hpp"
#include "pytext.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <exception>
#include <string>
#include <unordered_map>

using namespace strata::text;
using sv = std::string_view;

namespace {
thread_local std::string g_error;

template <class F>
auto guarded(F&& f, decltype(f()) failed) -> decltype(f()) {
    try {
        g_error.clear();
        return f();
    } catch (const std::exception& e) {
        g_error = e.what();
    } catch (...) {
        g_error = "unknown error";
    }
    return failed;
}
}  // namespace

struct strata_detok {
    std::string held, out;
};
struct strata_stop {
    std::vector<std::string> stops;
    std::string held, out;
    int64_t hit = -1;
};
struct strata_parser {
    OutputParser p;
    std::vector<Event> evs;
    std::string data;
    strata_parser(OutputParser&& q) : p(std::move(q)) {}
};

struct strata_pipe {
    strata_detok* det;
    strata_stop* stop;  // may be null
    strata_parser* par;
    const char* blob;
    const int64_t* offs;
    int64_t vocab;
    strata_parser_status* st;  // the parser wrapper's status, kept current
    char* ext;  // the caller's record buffer
    int64_t cap;
    std::string rec, tail, scratch;
    std::vector<Event> cut;
};

namespace {

void fill_status(const strata_parser* h, strata_parser_status* st) {
    if (!st) return;
    st->state = static_cast<int32_t>(h->p.state);
    st->pending = static_cast<int32_t>(h->p.pending_count());
    st->rescued = h->p.rescued;
    st->refused = h->p.refused;
    st->buf_is_think_end = h->p.buf == "</think>";
    st->reserved = 0;
    st->buf_len = static_cast<int64_t>(h->p.buf.size());
}

void put32(std::string& o, uint32_t v) { o.append(reinterpret_cast<const char*>(&v), 4); }

void encode_event(std::string& o, const Event& e) {
    o += static_cast<char>(e.kind);
    switch (e.kind) {
        case kEvReasoning:
        case kEvContent:
            put32(o, static_cast<uint32_t>(e.text.size()));
            o += e.text;
            break;
        case kEvToolStart:
            put32(o, static_cast<uint32_t>(e.cid));
            put32(o, static_cast<uint32_t>(e.name.size()));
            o += e.name;
            break;
        case kEvToolArgs:
            put32(o, static_cast<uint32_t>(e.cid));
            put32(o, static_cast<uint32_t>(e.text.size()));
            o += e.text;
            break;
        case kEvToolCall:
            put32(o, static_cast<uint32_t>(e.cid));
            o += static_cast<char>(e.same ? 1 : 0);
            put32(o, static_cast<uint32_t>(e.name.size()));
            o += e.name;
            put32(o, static_cast<uint32_t>(e.args.size()));
            o += e.args;
            break;
    }
}

void encode(strata_parser* h) {
    std::string& o = h->data;
    o.clear();
    for (const Event& e : h->evs) encode_event(o, e);
}

// codecs.getincrementaldecoder("utf-8")(errors="replace").decode(bytes): complete characters out, an incomplete
// tail held, each maximal invalid subpart one U+FFFD.
void detok_push(strata_detok* h, sv bytes) {
    std::string data;
    data.swap(h->held);
    data.append(bytes);
    h->out.clear();
    size_t i = 0, n = data.size();
    while (i < n) {
        unsigned char c = static_cast<unsigned char>(data[i]);
        if (c < 0x80) {
            h->out += static_cast<char>(c);
            ++i;
            continue;
        }
        size_t need;
        unsigned char lo = 0x80, hi = 0xBF;
        if (c >= 0xC2 && c <= 0xDF) need = 1;
        else if (c == 0xE0) { need = 2; lo = 0xA0; }
        else if (c == 0xED) { need = 2; hi = 0x9F; }
        else if (c >= 0xE1 && c <= 0xEF) need = 2;
        else if (c == 0xF0) { need = 3; lo = 0x90; }
        else if (c == 0xF4) { need = 3; hi = 0x8F; }
        else if (c >= 0xF1 && c <= 0xF3) need = 3;
        else {
            h->out += "\xEF\xBF\xBD";
            ++i;
            continue;
        }
        size_t k = 1;
        bool bad = false;
        for (; k <= need; ++k) {
            if (i + k >= n) break;
            unsigned char d = static_cast<unsigned char>(data[i + k]);
            // CPython holds a final "ED A0..BF" (an encoded surrogate) as if it were still incomplete, and only
            // rejects it once a third byte arrives; match that, it is what the reference does
            if (c == 0xED && k == 1 && i + 2 == n && d >= 0xA0 && d <= 0xBF) break;
            if (!(d >= (k == 1 ? lo : 0x80) && d <= (k == 1 ? hi : 0xBF))) {
                bad = true;
                break;
            }
        }
        if (bad) {
            h->out += "\xEF\xBF\xBD";
            i += k;
        } else if (k > need) {
            h->out.append(data, i, need + 1);
            i += need + 1;
        } else {
            h->held = data.substr(i);
            return;
        }
    }
}

void stop_push(strata_stop* h, sv text) {
    h->out.clear();
    if (h->hit >= 0) return;
    std::string buf = h->held + std::string(text);
    long first = -1;
    int64_t which = -1;
    for (size_t k = 0; k < h->stops.size(); ++k) {
        size_t i = buf.find(h->stops[k]);
        if (i != std::string::npos && (which < 0 || static_cast<long>(i) < first)) {
            first = static_cast<long>(i);
            which = static_cast<int64_t>(k);
        }
    }
    if (which >= 0) {
        h->held.clear();
        h->hit = which;
        h->out = buf.substr(0, static_cast<size_t>(first));
        return;
    }
    size_t keep = 0;  // the longest end of buf that a stop string starts with
    for (const std::string& s : h->stops) {
        if (s.size() < 2) continue;
        for (size_t k = std::min(s.size() - 1, buf.size()); k > keep; --k) {
            if ((static_cast<unsigned char>(s[k]) & 0xC0) == 0x80) continue;  // not at a character boundary
            if (buf.compare(buf.size() - k, k, s, 0, k) == 0) {
                keep = k;
                break;
            }
        }
    }
    h->held = buf.substr(buf.size() - keep);
    h->out = buf.substr(0, buf.size() - keep);
}

// str.lower(), as a list of code points: simple mappings from the generated table, U+0130 -> "i" + U+0307, and the
// Greek final sigma rule (a capital sigma after a cased letter and not before one is U+03C2).
std::vector<uint32_t> py_lower(const std::vector<uint32_t>& t) {
    std::vector<uint32_t> o;
    o.reserve(t.size() + 4);
    for (size_t i = 0; i < t.size(); ++i) {
        uint32_t c = t[i];
        if (c == 0x3A3) {
            size_t j = i;
            while (j > 0 && is_ignorable(t[j - 1])) --j;
            bool fin = j > 0 && is_cased_ni(t[j - 1]);
            if (fin) {
                size_t k = i + 1;
                while (k < t.size() && is_ignorable(t[k])) ++k;
                fin = k == t.size() || !is_cased_ni(t[k]);
            }
            o.push_back(fin ? 0x3C2 : 0x3C3);
        } else if (c == 0x130) {
            o.push_back(0x69);
            o.push_back(0x307);
        } else {
            o.push_back(lower_cp(c));
        }
    }
    return o;
}

std::vector<uint32_t> cps_of(sv s) {
    std::vector<uint32_t> o;
    o.reserve(s.size());
    for (size_t i = 0; i < s.size();) {
        size_t n;
        o.push_back(next_cp(s, i, &n));
        i += n;
    }
    return o;
}

struct GramHash {
    size_t operator()(const std::array<uint32_t, 12>& g) const {
        uint64_t h = 1469598103934665603ull;
        for (uint32_t v : g) h = (h ^ v) * 1099511628211ull;
        return static_cast<size_t>(h ^ (h >> 29));
    }
};

constexpr size_t LOOP_TAIL_CHARS = 2048, LOOP_TAIL_PERIOD = 64, LOOP_HISTORY_WORDS = 30000;

double repeat_coverage(sv text) {
    // the last LOOP_TAIL_CHARS characters
    size_t from = text.size();
    for (size_t n = 0; n < LOOP_TAIL_CHARS && from > 0; ++n) {
        --from;
        while (from > 0 && (static_cast<unsigned char>(text[from]) & 0xC0) == 0x80) --from;
    }
    std::vector<uint32_t> tail = py_lower(cps_of(text.substr(from)));
    if (tail.size() == LOOP_TAIL_CHARS) {
        for (size_t p = 1; p <= LOOP_TAIL_PERIOD; ++p)
            if (std::equal(tail.begin() + static_cast<long>(p), tail.end(), tail.begin())) return 1.0;
    }
    std::vector<uint32_t> low = py_lower(cps_of(text));
    // re.findall(r"\w+|[^\w\s]") as [begin, end) spans
    std::vector<std::pair<size_t, size_t>> spans;
    for (size_t i = 0; i < low.size();) {
        if (is_word_cp(low[i])) {
            size_t j = i + 1;
            while (j < low.size() && is_word_cp(low[j])) ++j;
            spans.emplace_back(i, j);
            i = j;
        } else {
            if (!is_space_cp(low[i])) spans.emplace_back(i, i + 1);
            ++i;
        }
    }
    if (spans.size() > LOOP_HISTORY_WORDS) spans.erase(spans.begin(), spans.end() - LOOP_HISTORY_WORDS);
    const size_t width = 12, window = 2000, N = spans.size();
    if (N < window) return 0.0;
    std::unordered_map<std::u32string, uint32_t> ids;
    std::vector<uint32_t> w(N);
    for (size_t k = 0; k < N; ++k) {
        std::u32string key(low.begin() + static_cast<long>(spans[k].first), low.begin() + static_cast<long>(spans[k].second));
        w[k] = ids.emplace(std::move(key), static_cast<uint32_t>(ids.size())).first->second;
    }
    size_t ngrams = N - width + 1;
    std::unordered_map<std::array<uint32_t, 12>, int, GramHash> counts;
    counts.reserve(ngrams * 2);
    auto gram = [&](size_t i) {
        std::array<uint32_t, 12> g;
        std::copy(w.begin() + static_cast<long>(i), w.begin() + static_cast<long>(i + width), g.begin());
        return g;
    };
    for (size_t i = 0; i < ngrams; ++i) ++counts[gram(i)];
    size_t start = N - window;
    std::vector<char> covered(N, 0);
    for (size_t i = start; i < ngrams; ++i)
        if (counts[gram(i)] >= 3) std::fill(covered.begin() + static_cast<long>(i), covered.begin() + static_cast<long>(i + width), 1);
    size_t n = static_cast<size_t>(std::count(covered.begin(), covered.end(), 1));
    return static_cast<double>(n) / static_cast<double>(N - start);
}


// ---- the fused step: Detokenizer -> OutputParser -> the server's cut() at the stop strings, in one call
int64_t pipe_flags(const strata_pipe* h) {
    const OutputParser& p = h->par->p;
    int64_t st = static_cast<int64_t>(p.state);
    int64_t f = st;
    if (!p.buf.empty()) f |= STRATA_PIPE_BUF;
    if (p.pending_count()) f |= STRATA_PIPE_PENDING;
    if (!h->det->held.empty()) f |= STRATA_PIPE_DETOK;
    if (h->stop && h->stop->hit >= 0) f |= STRATA_PIPE_HIT;
    if (p.buf == "</think>") f |= STRATA_PIPE_THINK_END;
    if (st == 0 || st == 4) f |= STRATA_PIPE_THINKING;
    return f;
}

// the server's cut(evs); `evs` is h->par->evs, the result goes to h->cut
void pipe_cut(strata_pipe* h) {
    std::vector<Event>& out = h->cut;
    out.clear();
    strata_stop* s = h->stop;
    for (Event& e : h->par->evs) {
        if (!s) {
            out.push_back(std::move(e));
            continue;
        }
        if (s->hit >= 0) break;
        if (e.kind == kEvContent) {
            stop_push(s, e.text);
            if (!s->out.empty()) {
                Event c;
                c.kind = kEvContent;
                c.text = s->out;
                out.push_back(std::move(c));
            }
            continue;
        }
        if (!s->held.empty()) {
            Event c;
            c.kind = kEvContent;
            c.text = std::move(s->held);
            s->held.clear();
            out.push_back(std::move(c));
        }
        out.push_back(std::move(e));
    }
}

int64_t pipe_emit(strata_pipe* h) {
    fill_status(h->par, h->st);
    int64_t f = pipe_flags(h);
    const std::vector<Event>& v = h->cut;
    const char* src;
    size_t n;
    if (v.empty()) return f;
    if (v.size() == 1 && (v[0].kind == kEvReasoning || v[0].kind == kEvContent)) {
        f |= (v[0].kind == kEvReasoning ? STRATA_PIPE_ONE_REASONING : STRATA_PIPE_ONE_CONTENT);
        src = v[0].text.data();
        n = v[0].text.size();
    } else {
        h->rec.clear();
        for (const Event& e : v) encode_event(h->rec, e);
        src = h->rec.data();
        n = h->rec.size();
    }
    if (n == 0) return f;
    if (h->ext && static_cast<int64_t>(n) <= h->cap) {
        std::memcpy(h->ext, src, n);
    } else {
        if (src != h->rec.data()) h->rec.assign(src, n);
        f |= STRATA_PIPE_OVERFLOW;
    }
    return f | (static_cast<int64_t>(n) << 16);
}

}  // namespace

extern "C" {

int strata_stream_abi_version(void) { return 1; }
const char* strata_stream_error(void) { return g_error.c_str(); }

strata_detok* strata_detok_new(void) {
    return guarded([] { return new strata_detok(); }, static_cast<strata_detok*>(nullptr));
}
void strata_detok_free(strata_detok* h) { delete h; }
int64_t strata_detok_push(strata_detok* h, const char* bytes, int64_t len) {
    return guarded(
        [&]() -> int64_t {
            detok_push(h, sv(bytes, static_cast<size_t>(len)));
            return static_cast<int64_t>(h->out.size());
        },
        -1);
}
const char* strata_detok_data(const strata_detok* h) { return h->out.c_str(); }
int strata_detok_pending(const strata_detok* h) { return h->held.empty() ? 0 : 1; }

strata_stop* strata_stop_new(const char* blob, const int64_t* offsets, int64_t n) {
    return guarded(
        [&]() -> strata_stop* {
            auto* h = new strata_stop();
            for (int64_t k = 0; k < n; ++k)
                h->stops.emplace_back(blob + offsets[k], static_cast<size_t>(offsets[k + 1] - offsets[k]));
            return h;
        },
        static_cast<strata_stop*>(nullptr));
}
void strata_stop_free(strata_stop* h) { delete h; }
int64_t strata_stop_push(strata_stop* h, const char* text, int64_t len) {
    return guarded(
        [&]() -> int64_t {
            stop_push(h, sv(text, static_cast<size_t>(len)));
            return static_cast<int64_t>(h->out.size());
        },
        -1);
}
int64_t strata_stop_flush(strata_stop* h) {
    h->out.swap(h->held);
    h->held.clear();
    return static_cast<int64_t>(h->out.size());
}
const char* strata_stop_data(const strata_stop* h) { return h->out.c_str(); }
int64_t strata_stop_hit(const strata_stop* h) { return h->hit; }
int64_t strata_stop_held_len(const strata_stop* h) { return static_cast<int64_t>(h->held.size()); }
const char* strata_stop_held(const strata_stop* h) { return h->held.c_str(); }

strata_parser* strata_parser_new(int thinking, int stream_tools, int recover, const char* tools_json, int64_t len) {
    return guarded(
        [&]() -> strata_parser* {
            bool any = false;
            std::unordered_map<std::string, ToolInfo> tools;
            if (tools_json && len > 0) {
                JVal j;
                if (!json_loads(sv(tools_json, static_cast<size_t>(len)), &j) || j.kind != JVal::Obj)
                    throw std::runtime_error("tools_json is not a JSON object");
                if (const JVal* a = j.get("any")) any = a->kind == JVal::Bool && a->b;
                if (const JVal* t = j.get("tools")) {
                    for (auto& m : t->o) {
                        ToolInfo ti;
                        for (auto& p : m.second.a) ti.string_params.insert(p.s);
                        tools[m.first] = std::move(ti);
                    }
                }
            }
            return new strata_parser(OutputParser(thinking != 0, stream_tools != 0, recover != 0, any, std::move(tools)));
        },
        static_cast<strata_parser*>(nullptr));
}
void strata_parser_free(strata_parser* h) { delete h; }

int64_t strata_parser_feed(strata_parser* h, const char* text, int64_t len, strata_parser_status* st) {
    return guarded(
        [&]() -> int64_t {
            h->evs.clear();
            try {
                h->p.feed(sv(text, static_cast<size_t>(len)), h->evs);
            } catch (...) {
                h->evs.clear();
                h->data.clear();
                fill_status(h, st);
                throw;
            }
            encode(h);
            fill_status(h, st);
            return static_cast<int64_t>(h->data.size());
        },
        -1);
}
int64_t strata_parser_finish(strata_parser* h, int natural_stop, strata_parser_status* st) {
    return guarded(
        [&]() -> int64_t {
            h->evs.clear();
            h->p.finish(natural_stop != 0, h->evs);
            encode(h);
            fill_status(h, st);
            return static_cast<int64_t>(h->data.size());
        },
        -1);
}
void strata_parser_status_get(const strata_parser* h, strata_parser_status* st) { fill_status(h, st); }
const char* strata_parser_data(const strata_parser* h) { return h->data.c_str(); }
const char* strata_parser_buf(const strata_parser* h) { return h->p.buf.c_str(); }


strata_pipe* strata_pipe_new(const char* blob, const int64_t* offsets, int64_t vocab, strata_detok* det,
                             strata_stop* stop, strata_parser* par, strata_parser_status* st, char* buf, int64_t cap) {
    return guarded(
        [&]() -> strata_pipe* {
            auto* h = new strata_pipe();
            h->blob = blob;
            h->offs = offsets;
            h->vocab = vocab;
            h->det = det;
            h->stop = stop;
            h->par = par;
            h->st = st;
            h->ext = buf;
            h->cap = cap;
            return h;
        },
        static_cast<strata_pipe*>(nullptr));
}
void strata_pipe_free(strata_pipe* h) { delete h; }
int64_t strata_pipe_status(const strata_pipe* h) { return pipe_flags(h); }
const char* strata_pipe_data(const strata_pipe* h) { return h->rec.c_str(); }
int64_t strata_pipe_tail(const strata_pipe* h) { return static_cast<int64_t>(h->tail.size()); }
const char* strata_pipe_tail_data(const strata_pipe* h) { return h->tail.c_str(); }

int64_t strata_pipe_step(strata_pipe* h, int32_t token, int track_tail) {
    if (token < 0 || token >= h->vocab) {
        g_error = "token id is outside the vocabulary";
        return -2;
    }
    try {
        detok_push(h->det, sv(h->blob + h->offs[token], static_cast<size_t>(h->offs[token + 1] - h->offs[token])));
        const std::string& piece = h->det->out;
        if (track_tail && !piece.empty()) {
            if (piece.size() >= 8) h->tail.assign(piece, piece.size() - 8, 8);
            else {
                h->tail += piece;
                if (h->tail.size() > 8) h->tail.erase(0, h->tail.size() - 8);
            }
        }
        std::vector<Event>& evs = h->par->evs;
        evs.clear();
        h->par->p.feed(sv(piece), evs);
        pipe_cut(h);
        return pipe_emit(h);
    } catch (const std::exception& e) {
        g_error = e.what();
    } catch (...) {
        g_error = "unknown error";
    }
    h->par->evs.clear();
    fill_status(h->par, h->st);
    return -1;
}

double strata_repeat_coverage(const char* text, int64_t len) {
    return guarded([&]() -> double { return repeat_coverage(sv(text, static_cast<size_t>(len))); }, -1.0);
}

}  // extern "C"
