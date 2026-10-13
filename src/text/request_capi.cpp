// src/text/request_capi.cpp - libstrata_text: the request translation of serve/frontend.py behind a C ABI
// (include/strata/text/strata_request.h).
//
// openai_to_messages / anthropic_to_messages and the helpers they call, over the JSON model of pytext.hpp (Python's
// json.loads / json.dumps).  Where the reference would raise something other than ValueError (TypeError,
// AttributeError, KeyError) or reads a value in a way not ported (see py_int), the port throws Unsupported and the
// caller runs the Python reference instead, so its behaviour there is the reference's own.
#include "strata/text/strata_request.h"

#include "pytext.hpp"

#include <cmath>
#include <cstring>
#include <exception>
#include <string>
#include <utility>
#include <vector>

namespace {

using strata::text::JVal;
using strata::text::ValueErr;

struct Unsupported {};

thread_local std::string g_error;
thread_local int g_kind = 0;
thread_local std::string g_result;

const JVal kNull;
const std::vector<JVal> kEmptyList;

// --- JVal helpers ---------------------------------------------------------------------------------------------
JVal mk_str(std::string s) {
    JVal v;
    v.kind = JVal::Str;
    v.s = std::move(s);
    return v;
}
JVal mk_bool(bool b) {
    JVal v;
    v.kind = JVal::Bool;
    v.b = b;
    return v;
}
JVal mk_obj() {
    JVal v;
    v.kind = JVal::Obj;
    return v;
}
JVal mk_arr() {
    JVal v;
    v.kind = JVal::Arr;
    return v;
}
void put(JVal& o, const char* key, JVal v) {  // dict[key] = v: replaces in place, else appends
    for (auto& m : o.o)
        if (m.first == key) {
            m.second = std::move(v);
            return;
        }
    o.o.emplace_back(key, std::move(v));
}
// d.get(key): a missing key reads as None
const JVal& get(const JVal& o, const char* key) {
    const JVal* p = o.get(key);
    return p ? *p : kNull;
}
bool has(const JVal& o, const char* key) { return o.get(key) != nullptr; }
bool truthy(const JVal& v) {
    switch (v.kind) {
        case JVal::Null: return false;
        case JVal::Bool: return v.b;
        case JVal::Int: return v.s != "0";
        case JVal::Float: return v.d != 0.0;
        case JVal::Str: return !v.s.empty();
        case JVal::Arr: return !v.a.empty();
        case JVal::Obj: return !v.o.empty();
    }
    return false;
}
bool is_str(const JVal& v, const char* s) { return v.kind == JVal::Str && v.s == s; }
bool is_obj(const JVal& v) { return v.kind == JVal::Obj; }

bool is_image_type(const JVal& t) { return is_str(t, "image_url") || is_str(t, "input_image") || is_str(t, "image"); }
bool is_text_type(const JVal& t) { return t.kind == JVal::Null || is_str(t, "text") || is_str(t, "input_text"); }

// --- Python's repr / str of a JSON value -----------------------------------------------------------------------
bool printable(uint32_t cp) {  // str.isprintable() for one character, as far as it matters in an error message
    if (cp < 0x20 || cp == 0x7F) return false;
    if (cp < 0x7F) return true;
    if (cp <= 0xA0 || cp == 0xAD) return false;
    if (cp >= 0xD800 && cp <= 0xDFFF) return false;
    if (strata::text::is_space_cp(cp)) return false;
    if ((cp >= 0x200B && cp <= 0x200F) || (cp >= 0x2028 && cp <= 0x202E) || (cp >= 0x2060 && cp <= 0x206F)) return false;
    if (cp == 0xFEFF || (cp >= 0xE000 && cp <= 0xF8FF) || (cp >= 0xFFF0 && cp <= 0xFFFF)) return false;
    if (cp >= 0xF0000) return false;
    return true;
}

void repr_str(const std::string& s, std::string& out) {
    char q = (s.find('\'') != std::string::npos && s.find('"') == std::string::npos) ? '"' : '\'';
    out += q;
    static const char* H = "0123456789abcdef";
    for (size_t i = 0; i < s.size();) {
        size_t n;
        uint32_t cp = strata::text::next_cp(s, i, &n);
        i += n;
        if (cp == static_cast<uint32_t>(q) || cp == '\\') {
            out += '\\';
            out += static_cast<char>(cp);
        } else if (cp == '\t') out += "\\t";
        else if (cp == '\n') out += "\\n";
        else if (cp == '\r') out += "\\r";
        else if (printable(cp)) strata::text::append_cp(out, cp);
        else {
            int digits = cp < 0x100 ? 2 : cp < 0x10000 ? 4 : 8;
            out += cp < 0x100 ? "\\x" : cp < 0x10000 ? "\\u" : "\\U";
            for (int sh = (digits - 1) * 4; sh >= 0; sh -= 4) out += H[(cp >> sh) & 15];
        }
    }
    out += q;
}

void float_str(double d, std::string& out) {  // repr(float): json.dumps writes NaN / Infinity where Python writes nan / inf
    if (std::isnan(d)) out += "nan";
    else if (std::isinf(d)) out += d < 0 ? "-inf" : "inf";
    else {
        JVal v;
        v.kind = JVal::Float;
        v.d = d;
        strata::text::json_dumps(v, false, true, out);
    }
}

void py_repr(const JVal& v, std::string& out) {
    switch (v.kind) {
        case JVal::Null: out += "None"; break;
        case JVal::Bool: out += v.b ? "True" : "False"; break;
        case JVal::Int: out += v.s; break;
        case JVal::Float: float_str(v.d, out); break;
        case JVal::Str: repr_str(v.s, out); break;
        case JVal::Arr:
            out += '[';
            for (size_t k = 0; k < v.a.size(); ++k) {
                if (k) out += ", ";
                py_repr(v.a[k], out);
            }
            out += ']';
            break;
        case JVal::Obj:
            out += '{';
            for (size_t k = 0; k < v.o.size(); ++k) {
                if (k) out += ", ";
                repr_str(v.o[k].first, out);
                out += ": ";
                py_repr(v.o[k].second, out);
            }
            out += '}';
            break;
    }
}

std::string py_str(const JVal& v) {
    if (v.kind == JVal::Str) return v.s;
    std::string out;
    py_repr(v, out);
    return out;
}

const char* type_name(const JVal& v) {
    switch (v.kind) {
        case JVal::Null: return "NoneType";
        case JVal::Bool: return "bool";
        case JVal::Int: return "int";
        case JVal::Float: return "float";
        case JVal::Str: return "str";
        case JVal::Arr: return "list";
        case JVal::Obj: return "dict";
    }
    return "object";
}

// --- the reference's helpers ------------------------------------------------------------------------------------
// _text_of(content)
std::string text_of(const JVal& c) {
    switch (c.kind) {
        case JVal::Null: return "";
        case JVal::Str: return c.s;
        case JVal::Obj: return "";  // iterating a dict yields its keys, which are not objects
        case JVal::Arr: {
            std::string out;
            for (const JVal& p : c.a) {
                if (!is_obj(p) || !is_text_type(get(p, "type"))) continue;
                const JVal* t = p.get("text");
                if (!t) continue;
                if (t->kind != JVal::Str) throw Unsupported{};  // "".join of a non-str: TypeError
                out += t->s;
            }
            return out;
        }
        default: throw Unsupported{};  // iterating a number: TypeError
    }
}

bool has_image(const JVal& c) {
    if (c.kind != JVal::Arr) return false;
    for (const JVal& p : c.a)
        if (is_obj(p) && is_image_type(get(p, "type"))) return true;
    return false;
}

// _image_source(part)
JVal image_source(const JVal& part) {
    if (is_str(get(part, "type"), "image")) {
        const JVal& sv = get(part, "source");
        if (truthy(sv) && !is_obj(sv)) throw Unsupported{};  // .get on a str / list / number: AttributeError
        const JVal& src = truthy(sv) ? sv : kNull;
        if (is_str(get(src, "type"), "base64")) {
            const JVal* m = src.get("media_type");
            const JVal* d = src.get("data");
            return mk_str("data:" + (m ? py_str(*m) : std::string("image/png")) + ";base64," + (d ? py_str(*d) : std::string()));
        }
        const JVal& u = get(src, "url");
        if (truthy(u)) return u;
        const JVal& p = get(src, "path");
        if (truthy(p)) return p;
        return mk_str("");
    }
    const JVal* url = &get(part, "image_url");
    if (is_obj(*url)) url = &get(*url, "url");
    return truthy(*url) ? *url : mk_str("");
}

// _check_parts(content, ...): a "refusal" part is read as text; the rest is returned as it is.  `hold` takes the copy.
const JVal& check_parts(const JVal& c, JVal& hold) {
    if (c.kind != JVal::Arr) return c;
    const JVal* out = &c;
    for (size_t j = 0; j < c.a.size(); ++j) {
        const JVal& part = c.a[j];
        if (!is_obj(part) || !is_str(get(part, "type"), "refusal")) continue;
        if (out == &c) {
            hold = c;
            out = &hold;
        }
        JVal t = mk_obj();
        put(t, "type", mk_str("text"));
        const JVal& r = get(part, "refusal");
        put(t, "text", truthy(r) ? r : mk_str(""));
        hold.a[j] = std::move(t);
    }
    return *out;
}

// _parts_of(content)
JVal parts_of(const JVal& c) {
    if (!has_image(c)) return mk_str(text_of(c));
    JVal items = mk_arr();
    for (const JVal& part : c.a) {
        if (!is_obj(part)) continue;
        const JVal& t = get(part, "type");
        if (is_image_type(t)) {
            JVal it = mk_obj();
            put(it, "type", mk_str("image"));
            put(it, "source", image_source(part));
            items.a.push_back(std::move(it));
        } else if (is_text_type(t) && has(part, "text")) {
            JVal it = mk_obj();
            put(it, "type", mk_str("text"));
            put(it, "text", get(part, "text"));
            items.a.push_back(std::move(it));
        }
    }
    return items;
}

// _object_list(value, name): the elements; `hold` takes a double-encoded value that was decoded
const std::vector<JVal>& object_list(const JVal& value, const char* name, JVal& hold) {
    if (value.kind == JVal::Null) return kEmptyList;
    const JVal* v = &value;
    if (value.kind == JVal::Str) {
        if (!strata::text::json_loads(value.s, &hold))
            throw ValueErr(std::string(name) + " must be a list of objects (a string was sent that is not JSON)");
        v = &hold;
    }
    if (v->kind != JVal::Arr) throw ValueErr(std::string(name) + " must be a list of objects");
    for (const JVal& m : v->a)
        if (!is_obj(m)) throw ValueErr(std::string(name) + " must be a list of objects");
    return v->a;
}

// _tool_list(value, wrapper): the tool objects (the function object inside the OpenAI wrapper), validated.
std::vector<const JVal*> tool_list(const JVal& value, bool openai, JVal& hold) {
    std::vector<const JVal*> out;
    if (!truthy(value)) return out;
    const std::string shape = openai ? "{\"type\": \"function\", \"function\": {\"name\": ..., \"parameters\": {...}}}"
                                     : "{\"name\": ..., \"input_schema\": {...}}";
    const std::vector<JVal>* tools;
    try {
        tools = &object_list(value, "tools", hold);
    } catch (const ValueErr&) {
        throw ValueErr("tools must be a list of tool objects, each " + shape);
    }
    const char* key = openai ? "parameters" : "input_schema";
    for (size_t i = 0; i < tools->size(); ++i) {
        const JVal& t = (*tools)[i];
        const JVal* fn = &t;
        if (openai && is_str(get(t, "type"), "function")) fn = t.get("function") ? t.get("function") : &t;
        const JVal* name = is_obj(*fn) ? fn->get("name") : nullptr;
        if (!name || name->kind != JVal::Str || name->s.empty())
            throw ValueErr("tools[" + std::to_string(i) + "] has no name: each tool must be " + shape);
        const JVal& schema = get(*fn, key);
        if (schema.kind != JVal::Null && !is_obj(schema))
            throw ValueErr("tools[" + std::to_string(i) + "] (" + name->s + "): \"" + key +
                           "\" must be an object (the JSON schema of its parameters), not " + type_name(schema));
        out.push_back(fn);
    }
    return out;
}

// tool_arguments(raw)
JVal tool_arguments(const JVal& raw) {
    auto text_args = [&](const JVal& r) {
        JVal o = mk_obj();
        put(o, "arguments", r);
        return o;
    };
    if (raw.kind == JVal::Null || (raw.kind == JVal::Str && strata::text::strip(raw.s).empty())) return mk_obj();
    if (is_obj(raw)) return raw;
    if (raw.kind == JVal::Str) {
        JVal v;
        if (!strata::text::json_loads(raw.s, &v)) return text_args(raw);
        return is_obj(v) ? v : text_args(raw);
    }
    std::string dumped;
    strata::text::json_dumps(raw, false, false, dumped);
    return text_args(mk_str(std::move(dumped)));
}

// effort_kwargs(value)
JVal effort_kwargs(const JVal& value) {
    JVal kw = mk_obj();
    if (value.kind == JVal::Null || (value.kind == JVal::Str && value.s.empty())) return kw;
    if (value.kind == JVal::Bool && !value.b) {
        put(kw, "enable_thinking", mk_bool(false));
        return kw;
    }
    std::string s = py_str(value);
    std::string_view t = strata::text::strip(s);
    std::string key;
    for (size_t i = 0; i < t.size();) {
        size_t n;
        uint32_t cp = strata::text::next_cp(t, i, &n);
        strata::text::append_cp(key, strata::text::lower_cp(cp));
        i += n;
    }
    const char* level = nullptr;
    bool known = true;
    if (key == "none" || key == "off" || key == "minimal" || key == "disabled" || key == "false") level = nullptr;
    else if (key == "low") level = "low";
    else if (key == "medium") level = "medium";
    else if (key == "high" || key == "xhigh" || key == "max" || key == "maximum") level = "xhigh";
    else known = false;
    if (!known) {
        std::string r;
        py_repr(value, r);
        throw ValueErr("unknown reasoning effort " + r + ": use none, low, medium or high");
    }
    if (!level) put(kw, "enable_thinking", mk_bool(false));
    else put(kw, "reasoning_effort", mk_str(level));
    return kw;
}

const char* level_of(bool neg, std::string digits) {  // digits: no sign, no leading zeros ("0" for zero)
    if (digits.size() > 18) return neg ? "low" : "xhigh";
    long long n = std::stoll(digits);
    if (neg) n = -n;
    return n < 2048 ? "low" : n < 8192 ? "medium" : "xhigh";
}

// budget_effort(tokens): int(tokens) as the reference does it; a value int() refuses (ValueError / TypeError) is {}
JVal budget_effort(const JVal& tokens) {
    JVal kw = mk_obj();
    const char* level = nullptr;
    switch (tokens.kind) {
        case JVal::Bool: level = tokens.b ? "low" : "low"; break;
        case JVal::Int: {
            bool neg = !tokens.s.empty() && tokens.s[0] == '-';
            std::string d = neg ? tokens.s.substr(1) : tokens.s;
            level = level_of(neg, d.empty() ? "0" : d);
            break;
        }
        case JVal::Float:
            if (std::isnan(tokens.d)) return kw;       // ValueError
            if (std::isinf(tokens.d)) throw Unsupported{};  // OverflowError, which the reference does not catch
            level = tokens.d < 2048 ? "low" : tokens.d < 8192 ? "medium" : "xhigh";
            break;
        case JVal::Str: {
            for (unsigned char ch : tokens.s)
                if (ch >= 0x80) throw Unsupported{};  // Unicode digits and spaces: int() has its own rules
            if (tokens.s.size() > 4000) throw Unsupported{};
            std::string_view t = strata::text::strip(tokens.s);
            size_t i = 0;
            bool neg = false;
            if (i < t.size() && (t[i] == '+' || t[i] == '-')) neg = t[i++] == '-';
            std::string digits;
            bool ok = i < t.size(), prev_digit = false;
            for (; ok && i < t.size(); ++i) {
                char ch = t[i];
                if (ch >= '0' && ch <= '9') {
                    digits += ch;
                    prev_digit = true;
                } else if (ch == '_' && prev_digit && i + 1 < t.size() && t[i + 1] >= '0' && t[i + 1] <= '9') {
                    prev_digit = false;
                } else {
                    ok = false;
                }
            }
            if (!ok || digits.empty()) return kw;
            size_t nz = digits.find_first_not_of('0');
            digits = nz == std::string::npos ? "0" : digits.substr(nz);
            level = level_of(neg, digits);
            break;
        }
        default: return kw;  // a list / dict: TypeError
    }
    put(kw, "reasoning_effort", mk_str(level));
    return kw;
}

void update(JVal& kw, const JVal& other) {
    for (auto& m : other.o) put(kw, m.first.c_str(), m.second);
}

// _late_system_to_user(messages)
void late_system_to_user(JVal& messages) {
    for (size_t i = 1; i < messages.a.size(); ++i) {
        JVal& m = messages.a[i];
        if (!m.o.empty() && m.o[0].first == "role" && is_str(m.o[0].second, "system")) m.o[0].second = mk_str("user");
    }
}

// pin_billing_stamp(system)
std::string pin_billing_stamp(const std::string& system) {
    static const std::string HEADER = "x-anthropic-billing-header:";
    if (!strata::text::starts_with(system, HEADER)) return system;
    std::vector<uint32_t> o;
    o.reserve(system.size());
    for (size_t i = 0; i < system.size();) {
        size_t n;
        o.push_back(strata::text::next_cp(system, i, &n));
        i += n;
    }
    std::vector<uint32_t> s = o;
    auto find = [&](const char* needle, size_t from) -> long {
        size_t m = std::strlen(needle);
        for (size_t i = from; i + m <= o.size(); ++i) {
            size_t k = 0;
            while (k < m && o[i + k] == static_cast<unsigned char>(needle[k])) ++k;
            if (k == m) return static_cast<long>(i);
        }
        return -1;
    };
    long cch = find("cch=", HEADER.size());
    if (0 <= cch && cch <= 160) {
        long v = cch + 4, end = find(";", static_cast<size_t>(v));
        if (end > v && end - v <= 16)
            for (long k = v; k < end; ++k) s[k] = 'f';
    }
    long cv = find("cc_version=", 0);
    if (0 <= cv && cv <= 160) {
        long v = cv + 11, end = find(";", static_cast<size_t>(cv));
        if (v < end && end - v <= 64) {
            int dots = 0;
            long tail = -1;
            for (long k = v; k < end; ++k)
                if (o[k] == '.' && ++dots == 3) {
                    tail = k + 1;
                    break;
                }
            if (tail >= 0)
                for (long k = tail; k < end; ++k) s[k] = 'f';
        }
    }
    std::string out;
    out.reserve(system.size());
    for (uint32_t cp : s) strata::text::append_cp(out, cp);
    return out;
}

// "".join(items) of the strings in `v`; a non-str element is a TypeError in the reference
std::string join_strs(const std::vector<const JVal*>& v) {
    std::string out;
    for (const JVal* p : v) {
        if (p->kind != JVal::Str) throw Unsupported{};
        out += p->s;
    }
    return out;
}

JVal function_call(const JVal& name, JVal args) {
    JVal f = mk_obj();
    put(f, "name", name);
    put(f, "arguments", std::move(args));
    JVal c = mk_obj();
    put(c, "function", std::move(f));
    return c;
}

JVal result_of(JVal messages, std::vector<const JVal*> tools, bool openai, JVal kwargs) {
    late_system_to_user(messages);
    JVal tl;
    if (tools.empty()) {
        tl = JVal();
    } else {
        tl = mk_arr();
        for (const JVal* fn : tools) {
            if (openai) {
                tl.a.push_back(*fn);
            } else {
                JVal t = mk_obj();
                put(t, "name", get(*fn, "name"));
                const JVal* d = fn->get("description");
                put(t, "description", d ? *d : mk_str(""));
                const JVal* p = fn->get("input_schema");
                put(t, "parameters", p ? *p : mk_obj());
                tl.a.push_back(std::move(t));
            }
        }
    }
    JVal r = mk_obj();
    put(r, "messages", std::move(messages));
    put(r, "tools", std::move(tl));
    put(r, "kwargs", std::move(kwargs));
    return r;
}

// openai_to_messages(req)
JVal openai_to_messages(const JVal& req) {
    JVal messages = mk_arr(), hold_m;
    const std::vector<JVal>& list = object_list(get(req, "messages"), "messages", hold_m);
    for (size_t i = 0; i < list.size(); ++i) {
        const JVal& m = list[i];
        JVal role = get(m, "role");
        if (is_str(role, "developer")) role = mk_str("system");
        bool images = is_str(role, "user") || is_str(role, "tool") || is_str(role, "assistant");
        JVal hold_c;
        const JVal& content = check_parts(get(m, "content"), hold_c);
        JVal out = mk_obj();
        put(out, "role", role);
        put(out, "content", images ? parts_of(content) : mk_str(text_of(content)));
        const JVal& rc = get(m, "reasoning_content");
        if (truthy(rc)) put(out, "reasoning_content", rc);
        const JVal& tc = get(m, "tool_calls");
        if (truthy(tc)) {
            JVal calls = mk_arr(), hold_t;
            for (const JVal& c : object_list(tc, "tool_calls", hold_t)) {
                const JVal* fn = c.get("function") ? c.get("function") : &c;
                if (!is_obj(*fn)) throw ValueErr("tool_calls must be a list of objects (each with a \"function\" object)");
                calls.a.push_back(function_call(get(*fn, "name"), tool_arguments(get(*fn, "arguments"))));
            }
            put(out, "tool_calls", std::move(calls));
        }
        messages.a.push_back(std::move(out));
    }
    JVal hold_tools;
    std::vector<const JVal*> tools = tool_list(get(req, "tools"), true, hold_tools);
    JVal kwargs = mk_obj();
    const JVal& rv = get(req, "reasoning");
    const JVal& reasoning = is_obj(rv) ? rv : kNull;
    const JVal& re = get(req, "reasoning_effort");
    update(kwargs, effort_kwargs(truthy(re) ? re : get(reasoning, "effort")));
    const JVal& ctk = get(req, "chat_template_kwargs");
    if (truthy(ctk)) {
        if (!is_obj(ctk)) throw Unsupported{};  // .items() of a non-dict: AttributeError
        for (auto& kv : ctk.o) {
            if (kv.first == "enable_thinking" && !truthy(kv.second)) {
                kwargs = mk_obj();
                put(kwargs, "enable_thinking", mk_bool(false));
            } else if (kv.first == "reasoning_effort" && !has(kwargs, "enable_thinking")) {
                update(kwargs, effort_kwargs(kv.second));
            }
        }
    }
    return result_of(std::move(messages), std::move(tools), true, std::move(kwargs));
}

// anthropic_to_messages(req, think_unasked)
JVal anthropic_to_messages(const JVal& req, bool think_unasked) {
    JVal messages = mk_arr();
    const JVal& system = get(req, "system");
    if (truthy(system)) {
        JVal hold_s;
        JVal m = mk_obj();
        put(m, "role", mk_str("system"));
        put(m, "content", mk_str(pin_billing_stamp(text_of(check_parts(system, hold_s)))));
        messages.a.push_back(std::move(m));
    }
    JVal hold_m;
    const std::vector<JVal>& list = object_list(get(req, "messages"), "messages", hold_m);
    for (size_t i = 0; i < list.size(); ++i) {
        const JVal& m = list[i];
        JVal hold_c;
        const JVal& content = check_parts(get(m, "content"), hold_c);
        if (content.kind == JVal::Str) {
            const JVal* role = m.get("role");
            if (!role) throw Unsupported{};  // m["role"]: KeyError
            JVal out = mk_obj();
            put(out, "role", *role);
            put(out, "content", content);
            messages.a.push_back(std::move(out));
            continue;
        }
        if (is_str(get(m, "role"), "user") && has_image(content)) {
            bool tool_result = false;
            for (const JVal& b : content.a)
                if (is_obj(b) && is_str(get(b, "type"), "tool_result")) tool_result = true;
            if (!tool_result) {
                JVal out = mk_obj();
                put(out, "role", mk_str("user"));
                put(out, "content", parts_of(content));
                messages.a.push_back(std::move(out));
                continue;
            }
        }
        std::vector<const JVal*> text, reasoning;
        JVal calls = mk_arr(), parts = mk_arr();
        static const JVal kNone;
        const std::vector<JVal>* blocks = &kEmptyList;
        if (content.kind == JVal::Arr) blocks = &content.a;
        else if (truthy(content)) throw Unsupported{};  // iterating a dict's keys / a number
        size_t j = 0;
        for (const JVal& block : *blocks) {
            if (!is_obj(block)) throw Unsupported{};  // block.get: AttributeError
            const JVal& kind = get(block, "type");
            if (is_str(kind, "text")) {
                const JVal* t = block.get("text");
                static const JVal empty = mk_str("");
                text.push_back(t ? t : &empty);
                parts.a.push_back(block);
            } else if (is_image_type(kind)) {
                parts.a.push_back(block);
            } else if (is_str(kind, "thinking")) {
                const JVal* t = block.get("thinking");
                static const JVal empty = mk_str("");
                reasoning.push_back(t ? t : &empty);
            } else if (is_str(kind, "tool_use")) {
                calls.a.push_back(function_call(get(block, "name"), tool_arguments(get(block, "input"))));
            } else if (is_str(kind, "tool_result")) {
                JVal hold_r;
                JVal out = mk_obj();
                put(out, "role", mk_str("tool"));
                put(out, "content", parts_of(check_parts(get(block, "content"), hold_r)));
                messages.a.push_back(std::move(out));
            }
            ++j;
        }
        (void)j;
        if (!text.empty() || !calls.a.empty() || !reasoning.empty() || !parts.a.empty()) {
            const JVal* role = m.get("role");
            if (!role) throw Unsupported{};
            JVal out = mk_obj();
            put(out, "role", *role);
            put(out, "content", has_image(parts) ? parts_of(parts) : mk_str(join_strs(text)));
            if (!reasoning.empty()) put(out, "reasoning_content", mk_str(join_strs(reasoning)));
            if (!calls.a.empty()) put(out, "tool_calls", std::move(calls));
            messages.a.push_back(std::move(out));
        }
    }
    JVal hold_tools;
    std::vector<const JVal*> tools = tool_list(get(req, "tools"), false, hold_tools);
    JVal kwargs = mk_obj();
    const JVal& thinking = get(req, "thinking");
    const JVal& oc = get(req, "output_config");
    const JVal& effort = is_obj(oc) ? get(oc, "effort") : kNull;
    if (is_obj(thinking) && is_str(get(thinking, "type"), "disabled")) {
        put(kwargs, "enable_thinking", mk_bool(false));
    } else if (truthy(effort)) {
        update(kwargs, effort_kwargs(effort));
    } else if (is_obj(thinking) && truthy(get(thinking, "budget_tokens"))) {
        update(kwargs, budget_effort(get(thinking, "budget_tokens")));
    } else if (thinking.kind == JVal::Null && !truthy(get(req, "reasoning_budget_tokens")) && !think_unasked) {
        put(kwargs, "enable_thinking", mk_bool(false));
    }
    return result_of(std::move(messages), std::move(tools), false, std::move(kwargs));
}

}  // namespace

extern "C" {

int strata_request_abi_version(void) { return 1; }

const char* strata_request_error(void) { return g_error.c_str(); }

int strata_request_error_kind(void) { return g_kind; }

int64_t strata_request_convert(int api, const char* req_json, int64_t len, int think_unasked) {
    g_error.clear();
    g_kind = 0;
    try {
        JVal req;
        if (!req_json || len < 0 || !strata::text::json_loads(std::string_view(req_json, static_cast<size_t>(len)), &req)) {
            g_error = "the request is not valid JSON";
            g_kind = 3;
            return -1;
        }
        if (!is_obj(req)) throw Unsupported{};
        JVal r = api == 0 ? openai_to_messages(req) : anthropic_to_messages(req, think_unasked != 0);
        g_result.clear();
        strata::text::json_dumps(r, true, true, g_result);
        return static_cast<int64_t>(g_result.size());
    } catch (const ValueErr& e) {
        g_error = e.what();
        g_kind = 1;
    } catch (const Unsupported&) {
        g_error = "a request shape the native translation leaves to the Python reference";
        g_kind = 2;
    } catch (const std::exception& e) {
        g_error = e.what();
        g_kind = 2;
    } catch (...) {
        g_error = "unknown error";
        g_kind = 2;
    }
    return -1;
}

int64_t strata_request_result(char* buf, int64_t cap) {
    int64_t n = static_cast<int64_t>(g_result.size());
    if (buf && cap > 0) std::memcpy(buf, g_result.data(), static_cast<size_t>(n < cap ? n : cap));
    return n;
}

}  // extern "C"
