// src/text/output_parser.cpp - see output_parser.hpp.  A line-by-line port of serve/frontend.py (ToolCall, param_end,
// call_end, function_end, json_tool_call, parse_tool_call, OutputParser); the comments there say why each rule is
// there.  Positions are byte offsets where Python's are code point offsets: every cut falls at an ASCII tag or at
// whitespace, so the two agree.
#include "output_parser.hpp"

#include <algorithm>
#include <tuple>

namespace strata::text {

namespace {

using sv = std::string_view;
constexpr sv THINK_END = "</think>", CALL_START = "<tool_call>", CALL_END = "</tool_call>";
constexpr sv PARAM_END = "</parameter>", FUNC_START = "<function=", FUNC_END = "</function>";
constexpr sv PARAM_START = "<parameter=";
constexpr size_t RCALL_MAX = 32768;

long lfind(sv s, sv n, size_t from = 0) {
    if (from > s.size()) return -1;
    size_t p = s.find(n, from);
    return p == sv::npos ? -1 : static_cast<long>(p);
}
long lfind(sv s, char c, size_t from = 0) {
    if (from > s.size()) return -1;
    size_t p = s.find(c, from);
    return p == sv::npos ? -1 : static_cast<long>(p);
}

// Python: where a parameter value ends (see frontend.param_end).  -1 none yet, -2 candidate awaiting its follower.
long param_end(sv text, bool final_ = false) {
    long at = lfind(text, PARAM_END);
    while (at >= 0) {
        sv after = lstrip(text.substr(static_cast<size_t>(at) + PARAM_END.size()));
        if (starts_with(after, PARAM_START) || starts_with(after, FUNC_END)) return at;
        if (after.empty() || prefix_of(after, PARAM_START) || prefix_of(after, FUNC_END)) return final_ ? at : -2;
        at = lfind(text, PARAM_END, static_cast<size_t>(at) + 1);
    }
    return -1;
}

long call_end(sv text) {
    size_t pos = lstrip_pos(text);
    sv s = text.substr(pos);
    if (!starts_with(s, FUNC_START)) return prefix_of(s, FUNC_START) ? -1 : lfind(text, CALL_END);
    long gt = lfind(text, '>', pos);
    if (gt < 0) return -1;
    pos = static_cast<size_t>(gt) + 1;
    for (;;) {
        sv rest = text.substr(pos);
        s = lstrip(rest);
        pos += rest.size() - s.size();
        if (starts_with(s, PARAM_START)) {
            gt = lfind(text, '>', pos);
            if (gt < 0) return -1;
            long end = param_end(text.substr(static_cast<size_t>(gt) + 1));
            if (end < 0) return -1;
            pos = static_cast<size_t>(gt) + 1 + static_cast<size_t>(end) + PARAM_END.size();
        } else if (starts_with(s, FUNC_END)) {
            rest = text.substr(pos + FUNC_END.size());
            s = lstrip(rest);
            if (starts_with(s, CALL_END)) return static_cast<long>(pos + FUNC_END.size() + rest.size() - s.size());
            return prefix_of(s, CALL_END) ? -1 : lfind(text, CALL_END, pos);
        } else if (s.empty() || prefix_of(s, PARAM_START) || prefix_of(s, FUNC_END)) {
            return -1;
        } else {
            return lfind(text, CALL_END, pos);
        }
    }
}

long function_end(sv text) {
    size_t pos = lstrip_pos(text);
    sv s = text.substr(pos);
    if (!starts_with(s, FUNC_START)) return -1;
    long gt = lfind(text, '>', pos);
    if (gt < 0) return -1;
    pos = static_cast<size_t>(gt) + 1;
    for (;;) {
        sv rest = text.substr(pos);
        s = lstrip(rest);
        pos += rest.size() - s.size();
        if (starts_with(s, PARAM_START)) {
            gt = lfind(text, '>', pos);
            if (gt < 0) return -1;
            long end = param_end(text.substr(static_cast<size_t>(gt) + 1));
            if (end < 0) return -1;
            pos = static_cast<size_t>(gt) + 1 + static_cast<size_t>(end) + PARAM_END.size();
        } else if (starts_with(s, FUNC_END)) {
            return static_cast<long>(pos + FUNC_END.size());
        } else {
            return -1;
        }
    }
}

size_t hold(sv text, std::initializer_list<sv> tags) {
    size_t best = 0;
    for (sv tag : tags)
        for (size_t n = 1; n < tag.size(); ++n)
            if (text.size() >= n && text.substr(text.size() - n) == tag.substr(0, n)) best = std::max(best, n);
    return best;
}

std::string first_cps(sv s, size_t n) {
    size_t i = 0;
    for (size_t k = 0; k < n && i < s.size(); ++k) {
        size_t l;
        next_cp(s, i, &l);
        i += l;
    }
    return std::string(s.substr(0, i));
}

void set_arg(JVal& obj, std::string key, JVal v) {
    for (auto& m : obj.o)
        if (m.first == key) {
            m.second = std::move(v);
            return;
        }
    obj.o.emplace_back(std::move(key), std::move(v));
}

// frontend.parse_tool_call
Call parse_tool_call(sv body_in, const ToolInfo* schema) {
    sv body = strip(body_in);
    long gt0 = lfind(body, '>');
    if (!starts_with(body, FUNC_START) || gt0 < 0) throw ValueErr("malformed tool call: " + first_cps(body, 80));
    Call c;
    c.name = std::string(strip(body.substr(FUNC_START.size(), static_cast<size_t>(gt0) - FUNC_START.size())));
    c.args.kind = JVal::Obj;
    sv rest = body.substr(static_cast<size_t>(gt0) + 1);
    for (;;) {
        long p = lfind(rest, PARAM_START);
        if (p < 0) break;
        rest = rest.substr(static_cast<size_t>(p) + PARAM_START.size());
        long gt = lfind(rest, '>');
        if (gt < 0) throw ValueErr("substring not found");
        std::string pname(rest.substr(0, static_cast<size_t>(gt)));
        rest = rest.substr(static_cast<size_t>(gt) + 1);
        long end = param_end(rest, true);
        sv value = end >= 0 ? rest.substr(0, static_cast<size_t>(end)) : rest;
        rest = end >= 0 ? rest.substr(static_cast<size_t>(end) + PARAM_END.size()) : sv();
        if (!value.empty() && value.front() == '\n') value.remove_prefix(1);
        if (!value.empty() && value.back() == '\n') value.remove_suffix(1);
        JVal v;
        if (schema && schema->string_params.count(pname)) {
            v.kind = JVal::Str;
            v.s = std::string(value);
        } else if (!json_loads(value, &v)) {
            v = JVal();
            v.kind = JVal::Str;
            v.s = std::string(value);
        }
        set_arg(c.args, std::move(pname), std::move(v));
    }
    return c;
}

}  // namespace

OutputParser::OutputParser(bool thinking, bool stream_tools_, bool recover_, bool any_tools,
                           std::unordered_map<std::string, ToolInfo> tools)
    : state(thinking ? PState::Reasoning : PState::Content), schemas(std::move(tools)), has_schemas(any_tools),
      stream_tools(stream_tools_), recover(recover_) {
    reset_scan();
}

const ToolInfo* OutputParser::schema_of(sv name) const {
    auto it = schemas.find(std::string(name));
    return it == schemas.end() ? nullptr : &it->second;
}

Event OutputParser::text_event(EvKind k, sv t) {
    track(t);
    Event e;
    e.kind = k;
    e.text = std::string(t);
    return e;
}

void OutputParser::track(sv text) {
    if (text.find('\n') == sv::npos) {
        line.append(text);
        return;
    }
    size_t start = 0;
    for (;;) {
        size_t nl = text.find('\n', start);
        if (nl == sv::npos) {  // the last part: the line goes on
            line.assign(text.substr(start));
            return;
        }
        line.append(text.substr(start, nl - start));
        sv s = lstrip(line);
        if (!fence.empty()) {
            if (s.size() >= 3 && s[0] == fence[0] && s[1] == fence[0] && s[2] == fence[0]) fence.clear();
        } else if (s.size() >= 3 && (starts_with(s, "```") || starts_with(s, "~~~")) &&
                   s.substr(3).find(std::string(3, s[0])) == sv::npos) {
            fence = std::string(1, s[0]);
        } else if (s.empty()) {
            ticks = 0;
        } else {
            ticks += std::count(line.begin(), line.end(), '`');
        }
        line.clear();
        start = nl + 1;
    }
}

bool OutputParser::opener_ok() const { return fence.empty() && lstrip(line).empty() && ticks % 2 == 0; }

bool OutputParser::in_code() const {
    return !fence.empty() || (ticks + std::count(line.begin(), line.end(), '`')) % 2 == 1;
}

bool OutputParser::ok_at(size_t p) {
    std::string f = fence, l = line;
    int64_t t = ticks;
    track(sv(buf).substr(0, p));
    bool ok = opener_ok();
    fence = std::move(f);
    line = std::move(l);
    ticks = t;
    return ok;
}

std::optional<bool> OutputParser::quoted_think_end(size_t i) const {
    char before = i ? buf[i - 1] : (line.empty() ? '\0' : line.back());
    if (before != '"' && before != '\'' && before != '`') return false;
    if (buf.size() <= i + THINK_END.size()) return std::nullopt;
    char a = buf[i + THINK_END.size()];
    return a != '\r' && a != '\n';
}

void OutputParser::release(bool deliver, std::vector<Event>& out) {
    for (auto& p : pending) {
        if (p.call && deliver) {
            Event e;
            e.kind = kEvToolCall;
            e.name = p.call->name;
            json_dumps(p.call->args, false, true, e.args);
            out.push_back(std::move(e));
            ++rescued;
        } else {
            if (p.call) ++refused;
            if (!p.raw.empty()) out.push_back(text_event(kEvReasoning, p.raw));
        }
    }
    pending.clear();
}

void OutputParser::reset_scan() {
    sp = 0;
    ss = SS::Name;
    scall = -1;
    scall_name.clear();
    sfirst = true;
    sval_started = false;
    sdeclared = nullptr;
}

void OutputParser::scan(sv b, std::vector<Event>& out) {
    auto args = [&](const std::string& s) {
        if (!s.empty()) {
            Event e;
            e.kind = kEvToolArgs;
            e.text = s;
            e.cid = scall;
            out.push_back(std::move(e));
        }
    };
    for (;;) {
        sv rest = b.substr(std::min(sp, b.size()));
        if (ss == SS::Name) {
            long a = lfind(rest, FUNC_START);
            long bb = a >= 0 ? lfind(rest, '>', static_cast<size_t>(a) + FUNC_START.size()) : -1;
            if (bb < 0) return;
            sv name = rest.substr(static_cast<size_t>(a) + FUNC_START.size(),
                                  static_cast<size_t>(bb) - static_cast<size_t>(a) - FUNC_START.size());
            scall = next_cid++;
            scall_name = std::string(name);
            sdeclared = schema_of(name);
            Event e;
            e.kind = kEvToolStart;
            e.cid = scall;
            e.name = scall_name;
            out.push_back(std::move(e));
            args("{");
            sp += static_cast<size_t>(bb) + 1;
            ss = SS::Between;
        } else if (ss == SS::Between) {
            sv stripped = lstrip(rest);
            sp += rest.size() - stripped.size();
            if (starts_with(stripped, PARAM_START)) {
                long gt = lfind(stripped, '>');
                if (gt < 0) return;
                std::string pname(stripped.substr(PARAM_START.size(), static_cast<size_t>(gt) - PARAM_START.size()));
                std::string piece = sfirst ? "" : ",";
                piece += '"';
                json_escape(pname, true, piece);
                piece += "\":";
                args(piece);
                sfirst = false;
                sp += static_cast<size_t>(gt) + 1;
                if (sdeclared && sdeclared->string_params.count(pname)) {
                    args("\"");
                    ss = SS::Str;
                    sval_started = false;
                } else {
                    ss = SS::Raw;
                }
            } else if (starts_with(stripped, FUNC_END)) {
                args("}");
                sp += FUNC_END.size();
                ss = SS::Done;
            } else {
                return;
            }
        } else if (ss == SS::Str) {
            if (!sval_started) {
                if (rest.empty()) return;
                if (rest[0] == '\n') {
                    sp += 1;
                    rest.remove_prefix(1);
                }
                sval_started = true;
            }
            long end = param_end(rest);
            if (end >= 0) {
                sv value = rest.substr(0, static_cast<size_t>(end));
                if (!value.empty() && value.back() == '\n') value.remove_suffix(1);
                std::string piece;
                json_escape(value, true, piece);
                piece += '"';
                args(piece);
                sp += static_cast<size_t>(end) + PARAM_END.size();
                ss = SS::Between;
                continue;
            }
            long safe = end == -2 ? lfind(rest, PARAM_END) : static_cast<long>(rest.size() - hold(rest, {PARAM_END}));
            if (safe > 0 && rest[static_cast<size_t>(safe) - 1] == '\n') --safe;
            if (safe > 0) {
                std::string piece;
                json_escape(rest.substr(0, static_cast<size_t>(safe)), true, piece);
                args(piece);
                sp += static_cast<size_t>(safe);
            }
            return;
        } else if (ss == SS::Raw) {
            long end = param_end(rest);
            if (end < 0) return;
            sv value = rest.substr(0, static_cast<size_t>(end));
            if (!value.empty() && value.front() == '\n') value.remove_prefix(1);
            if (!value.empty() && value.back() == '\n') value.remove_suffix(1);
            JVal v;
            if (!json_loads(value, &v)) {
                v = JVal();
                v.kind = JVal::Str;
                v.s = std::string(value);
            }
            std::string piece;
            json_dumps(v, false, false, piece);
            args(piece);
            sp += static_cast<size_t>(end) + PARAM_END.size();
            ss = SS::Between;
        } else {
            return;
        }
    }
}

void OutputParser::close_scan(sv b, std::vector<Event>& out) {
    if (scall < 0 || ss == SS::Done) return;
    std::string tail;
    if (ss == SS::Str) {
        tail += '"';
    } else if (ss == SS::Raw) {
        sv rest = b.substr(std::min(sp, b.size()));
        size_t a = 0, e = rest.size();
        while (a < e && rest[a] == '\n') ++a;
        while (e > a && rest[e - 1] == '\n') --e;
        tail += '"';
        json_escape(rest.substr(a, e - a), true, tail);
        tail += '"';
    }
    tail += "}";
    Event ev;
    ev.kind = kEvToolArgs;
    ev.text = tail;
    ev.cid = scall;
    out.push_back(std::move(ev));
    ss = SS::Done;
}

OutputParser::Follow OutputParser::follower(sv after) const {
    if (starts_with(after, FUNC_START)) return Follow::Call;
    if (starts_with(after, PARAM_START)) {
        long gt = lfind(after, '>');
        if (gt < 0) return after.find('\n') == sv::npos ? Follow::Wait : Follow::Text;
        return schemas.count(std::string(after.substr(PARAM_START.size(), static_cast<size_t>(gt) - PARAM_START.size())))
                   ? Follow::Param
                   : Follow::Text;
    }
    if (starts_with(after, "{")) return has_schemas ? Follow::Json : Follow::Text;
    if (after.empty() || prefix_of(after, FUNC_START) || prefix_of(after, PARAM_START)) return Follow::Wait;
    return Follow::Text;
}

std::pair<long, bool> OutputParser::bare_opener() {
    long p = lfind(buf, FUNC_START);
    while (p >= 0) {
        if (ok_at(static_cast<size_t>(p))) {
            long gt = lfind(buf, '>', static_cast<size_t>(p));
            size_t ns = static_cast<size_t>(p) + FUNC_START.size();
            sv name = sv(buf).substr(ns, gt >= 0 ? static_cast<size_t>(gt) - ns : sv::npos);
            if (gt < 0 && !has_space(name)) return {p, false};
            if (gt >= 0 && schemas.count(std::string(name))) return {p, true};
        }
        p = lfind(buf, FUNC_START, static_cast<size_t>(p) + 1);
    }
    return {-1, false};
}

void OutputParser::finish_call(sv body, std::vector<Event>& out) {
    sv st = strip(body);
    st = st.size() >= FUNC_START.size() ? st.substr(FUNC_START.size()) : sv();
    long g = lfind(st, '>');
    std::string name(strip(g >= 0 ? st.substr(0, static_cast<size_t>(g)) : st));
    bool was_bare = bare;
    bare = false;
    Call call;
    try {
        call = parse_tool_call(body, schema_of(name));
    } catch (const ValueErr&) {
        if (!recover) throw;
        reset_scan();
        state = PState::Content;
        std::string t = was_bare ? std::string(body) : std::string(CALL_START) + std::string(body) + std::string(CALL_END);
        out.push_back(text_event(kEvContent, t));
        return;
    }
    Event e;
    e.kind = kEvToolCall;
    e.cid = scall;  // reuses the announced call's id, if there was one
    e.name = call.name;
    json_dumps(call.args, false, true, e.args);
    reset_scan();
    state = PState::Content;
    lead = true;
    out.push_back(std::move(e));
}

void OutputParser::feed(sv delta, std::vector<Event>& out) {
    buf.append(delta);
    for (;;) {
        if (state == PState::RCall) {
            sv body = sv(buf).substr(CALL_START.size());
            long end = call_end(body), think = lfind(body, THINK_END);
            if (end >= 0 && (think < 0 || think >= end)) {
                std::optional<Call> call;
                try {
                    sv b = strip(body.substr(0, static_cast<size_t>(end)));
                    b = b.size() >= FUNC_START.size() ? b.substr(FUNC_START.size()) : sv();
                    long g = lfind(b, '>');
                    sv name = strip(g >= 0 ? b.substr(0, static_cast<size_t>(g)) : b);
                    if (const ToolInfo* ti = schema_of(name)) {
                        if (schemas.count(std::string(name))) call = parse_tool_call(body.substr(0, static_cast<size_t>(end)), ti);
                    }
                } catch (const ValueErr&) {
                }
                std::string raw = buf.substr(0, CALL_START.size() + static_cast<size_t>(end) + CALL_END.size());
                if (call || !pending.empty()) {
                    pending.push_back({std::move(raw), std::move(call)});
                } else {
                    out.push_back(text_event(kEvReasoning, raw));
                }
                buf = std::string(body.substr(static_cast<size_t>(end) + CALL_END.size()));
                state = PState::Reasoning;
            } else if (think >= 0) {
                release(false, out);
                out.push_back(text_event(kEvReasoning, sv(buf).substr(0, CALL_START.size() + static_cast<size_t>(think))));
                buf = std::string(body.substr(static_cast<size_t>(think)));
                state = PState::Reasoning;
            } else if (body.size() > RCALL_MAX && [&] {
                           size_t chars = 0;
                           for (char c : body) chars += (static_cast<unsigned char>(c) & 0xC0) != 0x80;
                           return chars > RCALL_MAX;
                       }()) {
                release(false, out);
                out.push_back(text_event(kEvReasoning, buf));
                buf.clear();
                state = PState::Reasoning;
            } else {
                return;
            }
        } else if (state == PState::Reasoning) {
            if (!pending.empty()) {
                sv stripped = lstrip(buf);
                size_t ws = buf.size() - stripped.size();
                if (ws > 0) {
                    pending.back().raw.append(buf, 0, ws);
                    buf.erase(0, ws);
                }
                if (buf.empty()) return;
                if (starts_with(buf, CALL_START)) {
                    state = PState::RCall;
                } else if (starts_with(buf, THINK_END)) {
                    release(true, out);
                } else if (prefix_of(buf, CALL_START) || prefix_of(buf, THINK_END)) {
                    return;
                } else {
                    release(false, out);
                }
                continue;
            }
            long i = lfind(buf, THINK_END);
            long tool = has_schemas ? lfind(buf, CALL_START) : -1;
            if (tool >= 0 && (i < 0 || tool < i)) {
                if (tool) out.push_back(text_event(kEvReasoning, sv(buf).substr(0, static_cast<size_t>(tool))));
                if (opener_ok()) {
                    buf.erase(0, static_cast<size_t>(tool));
                    state = PState::RCall;
                } else {
                    out.push_back(text_event(kEvReasoning, CALL_START));
                    buf.erase(0, static_cast<size_t>(tool) + CALL_START.size());
                }
                continue;
            }
            if (i < 0) {
                size_t keep = has_schemas ? hold(buf, {THINK_END, CALL_START}) : hold(buf, {THINK_END});
                if (buf.size() > keep) {
                    out.push_back(text_event(kEvReasoning, sv(buf).substr(0, buf.size() - keep)));
                    buf.erase(0, buf.size() - keep);
                }
                return;
            }
            auto quoted = quoted_think_end(static_cast<size_t>(i));
            if (!quoted) {
                if (i) {
                    out.push_back(text_event(kEvReasoning, sv(buf).substr(0, static_cast<size_t>(i))));
                    buf.erase(0, static_cast<size_t>(i));
                }
                return;
            }
            if (*quoted) {
                out.push_back(text_event(kEvReasoning, sv(buf).substr(0, static_cast<size_t>(i) + THINK_END.size())));
                buf.erase(0, static_cast<size_t>(i) + THINK_END.size());
                continue;
            }
            if (i) out.push_back(text_event(kEvReasoning, sv(buf).substr(0, static_cast<size_t>(i))));
            buf.erase(0, static_cast<size_t>(i) + THINK_END.size());
            state = PState::Content;
            lead = true;
            fence.clear();
            line.clear();
            ticks = 0;
        } else if (state == PState::Content) {
            if (lead) {
                size_t k = 0;
                while (k < buf.size() && buf[k] == '\n') ++k;
                if (k == buf.size()) {
                    buf.clear();
                    return;
                }
                buf.erase(0, k);
                lead = false;
            }
            long i = lfind(buf, CALL_START);
            long b = -1;
            bool decided = false;
            if (recover && has_schemas) std::tie(b, decided) = bare_opener();
            if (b >= 0 && (i < 0 || b < i)) {
                size_t j = static_cast<size_t>(b);
                while (j > 0 && buf[j - 1] == '\n') --j;
                if (j > 0 && !strip(sv(buf).substr(0, j)).empty())
                    out.push_back(text_event(kEvContent, sv(buf).substr(0, j)));
                if (!decided) {
                    buf.erase(0, j);
                    return;
                }
                buf.erase(0, static_cast<size_t>(b));
                state = PState::Call;
                bare = true;
                continue;
            }
            if (i < 0) {
                size_t j = buf.size() - hold(buf, {CALL_START});
                if (recover && has_schemas) {
                    size_t k = buf.rfind('\n');
                    k = k == std::string::npos ? 0 : k + 1;
                    sv tail = sv(buf).substr(k);
                    if (!tail.empty() && prefix_of(tail, FUNC_START) && ok_at(k)) j = std::min(j, k);
                }
                while (j > 0 && buf[j - 1] == '\n') --j;
                if (j > 0) {
                    out.push_back(text_event(kEvContent, sv(buf).substr(0, j)));
                    buf.erase(0, j);
                }
                return;
            }
            size_t ui = static_cast<size_t>(i);
            {
                std::string f = fence, l = line;
                int64_t t = ticks;
                track(sv(buf).substr(0, ui));
                bool code = in_code();
                fence = std::move(f);
                line = std::move(l);
                ticks = t;
                if (code) {
                    out.push_back(text_event(kEvContent, sv(buf).substr(0, ui + CALL_START.size())));
                    buf.erase(0, ui + CALL_START.size());
                    continue;
                }
            }
            sv after = lstrip(sv(buf).substr(ui + CALL_START.size()));
            Follow kind = recover ? follower(after) : Follow::Call;  // Call here stands for Python's None
            bool kind_none = !recover;
            if (!kind_none && (kind == Follow::Param || kind == Follow::Json)) {
                if (ui && !strip(sv(buf).substr(0, ui)).empty()) {
                    sv t = sv(buf).substr(0, ui);
                    while (!t.empty() && t.back() == '\n') t.remove_suffix(1);
                    out.push_back(text_event(kEvContent, t));
                }
                std::string rest = buf.substr(ui + CALL_START.size());
                if (kind == Follow::Param) {
                    size_t at = rest.size() - after.size();
                    rest = rest.substr(0, at) + std::string(FUNC_START) + std::string(after.substr(PARAM_START.size()));
                }
                buf = std::move(rest);
                state = kind == Follow::Param ? PState::Call : PState::Json;
                continue;
            }
            if (!kind_none && kind == Follow::Wait) {
                after = sv();
            }
            if (!after.empty() && !starts_with(after, FUNC_START) && !prefix_of(after, FUNC_START)) {
                out.push_back(text_event(kEvContent, sv(buf).substr(0, ui + CALL_START.size())));
                buf.erase(0, ui + CALL_START.size());
                continue;
            }
            if (!starts_with(after, FUNC_START)) {
                size_t j = ui;
                while (j > 0 && buf[j - 1] == '\n') --j;
                if (j > 0) {
                    out.push_back(text_event(kEvContent, sv(buf).substr(0, j)));
                    buf.erase(0, j);
                }
                return;
            }
            if (ui && !strip(sv(buf).substr(0, ui)).empty()) {
                sv t = sv(buf).substr(0, ui);
                while (!t.empty() && t.back() == '\n') t.remove_suffix(1);
                out.push_back(text_event(kEvContent, t));
            }
            buf.erase(0, ui + CALL_START.size());
            state = PState::Call;
        } else if (state == PState::Json) {
            long i = lfind(buf, CALL_END);
            if (i < 0) return;
            sv jb = strip(sv(buf).substr(0, static_cast<size_t>(i)));
            JVal obj;
            std::optional<Call> call;
            if (json_loads(jb, &obj) && obj.kind == JVal::Obj) {
                const JVal* nm = obj.get("name");
                if (nm && nm->kind == JVal::Str && schemas.count(nm->s)) {
                    const JVal* a = obj.get("arguments");
                    if (!a) a = obj.get("parameters");
                    JVal args;
                    args.kind = JVal::Obj;
                    bool good = true;
                    if (a) {
                        if (a->kind == JVal::Str) {
                            JVal inner;
                            if (json_loads(a->s, &inner)) args = std::move(inner), good = true;
                            else good = false;
                        } else {
                            args = *a;
                        }
                    }
                    if (good && args.kind == JVal::Null) {
                        args = JVal();
                        args.kind = JVal::Obj;
                    }
                    if (good && args.kind == JVal::Obj) {
                        Call c;
                        c.name = nm->s;
                        c.args = std::move(args);
                        call = std::move(c);
                    }
                }
            }
            if (!call) {
                std::string t = std::string(CALL_START) + buf.substr(0, static_cast<size_t>(i) + CALL_END.size());
                out.push_back(text_event(kEvContent, t));
            } else {
                Event e;
                e.kind = kEvToolCall;
                e.name = call->name;
                json_dumps(call->args, false, true, e.args);
                out.push_back(std::move(e));
            }
            buf.erase(0, static_cast<size_t>(i) + CALL_END.size());
            state = PState::Content;
            lead = call.has_value();
        } else {  // Call
            long i;
            std::optional<size_t> drop;
            if (bare) {
                i = function_end(buf);
                if (i >= 0) {
                    sv rest = sv(buf).substr(static_cast<size_t>(i));
                    sv after = lstrip(rest);
                    if (starts_with(after, CALL_END)) drop = rest.size() - after.size() + CALL_END.size();
                    else if (!after.empty() && !prefix_of(after, CALL_END)) drop = 0;
                }
            } else {
                i = call_end(buf);
                if (i >= 0) drop = CALL_END.size();
                if (recover) {
                    long f = function_end(buf);
                    std::string nxt_s = f >= 0 ? std::string(lstrip(sv(buf).substr(static_cast<size_t>(f)))) : std::string();
                    sv nxt = nxt_s;
                    if (starts_with(nxt, PARAM_START) && follower(nxt) == Follow::Param) {
                        size_t at = buf.size() - nxt.size();
                        buf = buf.substr(0, at) + std::string(FUNC_START) + std::string(nxt.substr(PARAM_START.size()));
                        nxt_s = buf.substr(at);
                        nxt = nxt_s;
                    }
                    if (starts_with(nxt, FUNC_START)) {
                        i = f;
                        drop = 0;
                        batch = true;
                    }
                }
            }
            if (stream_tools) {
                if (i >= 0) {
                    sv body_only = sv(buf).substr(0, static_cast<size_t>(i));  // scan only the body
                    scan(body_only, out);
                    close_scan(body_only, out);
                } else {
                    scan(buf, out);
                }
            }
            if (i < 0 || !drop) return;
            std::string body = buf.substr(0, static_cast<size_t>(i));
            buf.erase(0, static_cast<size_t>(i) + *drop);
            bool was_batch = batch;
            batch = false;
            finish_call(body, out);
            if (was_batch && out.back().kind == kEvToolCall) {
                state = PState::Call;
                lead = false;
            }
        }
    }
}

void OutputParser::finish(bool natural_stop, std::vector<Event>& out) {
    if (!pending.empty()) release(natural_stop && state == PState::Reasoning && buf.empty(), out);
    if (state == PState::Call && bare && function_end(buf) >= 0) {
        size_t i = static_cast<size_t>(function_end(buf));
        std::string body = buf.substr(0, i);
        std::string rest(strip(sv(buf).substr(i)));
        buf.clear();
        finish_call(body, out);
        if (!rest.empty() && !prefix_of(rest, CALL_END)) out.push_back(text_event(kEvContent, rest));
        return;
    }
    if (state == PState::Call && stream_tools && scall >= 0) {
        scan(buf, out);
        if (ss == SS::Done) {
            Event e;
            e.kind = kEvToolCall;
            e.cid = scall;
            e.same = true;
            e.name = scall_name;
            e.args = "{}";
            out.push_back(std::move(e));
        }
        buf.clear();
        reset_scan();
        return;
    }
    if (state == PState::Reasoning && buf == THINK_END) {
        buf.clear();
        state = PState::Content;
    }
    if (!buf.empty()) {
        EvKind k = (state == PState::Reasoning || state == PState::RCall) ? kEvReasoning : kEvContent;
        std::string t = (state == PState::Call || state == PState::Json) && !bare ? std::string(CALL_START) + buf : buf;
        Event e;
        e.kind = k;
        e.text = std::move(t);
        out.push_back(std::move(e));
        buf.clear();
    }
}

}  // namespace strata::text
