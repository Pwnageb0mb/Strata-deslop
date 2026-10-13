// src/text/template_capi.cpp - libstrata_text: chat-template rendering behind a C ABI (include/strata/text/strata_template.h).
//
// The engine is llama.cpp's Jinja (third_party/llama_jinja).  The context is set up the way common/chat.cpp does it
// (jinja::context, jinja::runtime::execute, gather_string_parts) with the variables the Python ChatTemplate passes:
// messages, tools, add_generation_prompt and the request's kwargs.  The JSON arguments are read straight into Jinja
// values by json_reader below.
#include "strata/text/strata_template.h"

#include "jinja/lexer.h"
#include "jinja/parser.h"
#include "jinja/runtime.h"
#include "jinja/value.h"

#include <exception>
#include <unordered_map>
#include <vector>
#include <cstdint>
#include <cmath>
#include <memory>
#include <mutex>
#include <cstring>
#include <string>
#include <stdexcept>

struct strata_template {
    jinja::program prog;
    std::string src;
    std::mutex mu;
};

namespace {

// Builds the Jinja values straight from the JSON text: the values jinja::from_json makes from a parsed nlohmann tree
// (null, bool, int64, double, string, array, object with insertion order and last-key-wins), without the tree and
// without nlohmann's per-character lexer (which was about 40% of a render of a long history).  Strict JSON, as
// nlohmann was: no comments, no trailing commas, no leading zeros, well-formed UTF-8 only, escapes checked.
class json_reader {
public:
    json_reader(const char* s, size_t n) : p_(s), end_(s + n), begin_(s) {}

    jinja::value read_document() {
        jinja::value v = read_value(0);
        skip_ws();
        if (p_ != end_) {
            fail("unexpected data after the value");
        }
        return v;
    }

private:
    const char* p_;
    const char* end_;
    const char* begin_;
    std::unordered_map<std::string, jinja::value> keys_;   // object keys seen so far: shared, immutable key values

    [[noreturn]] void fail(const char* what) const {
        throw std::runtime_error(std::string("parse error at byte ") + std::to_string(p_ - begin_ + 1) + ": " + what);
    }
    void skip_ws() {
        while (p_ < end_ && (*p_ == ' ' || *p_ == '\n' || *p_ == '\t' || *p_ == '\r')) {
            ++p_;
        }
    }
    bool literal(const char* lit, size_t n) {
        if (static_cast<size_t>(end_ - p_) >= n && std::memcmp(p_, lit, n) == 0) {
            p_ += n;
            return true;
        }
        return false;
    }

    jinja::value read_value(int depth) {
        if (depth > 512) {
            fail("nesting too deep");
        }
        skip_ws();
        if (p_ >= end_) {
            fail("unexpected end of input");
        }
        switch (*p_) {
            case '{': return read_object(depth);
            case '[': return read_array(depth);
            case '"': {
                auto str = jinja::mk_val<jinja::value_string>();
                std::string text;
                read_string(text);
                str->val_str.parts.push_back({false, std::move(text)});
                return str;
            }
            case 't':
                if (literal("true", 4)) return jinja::mk_val<jinja::value_bool>(true);
                fail("invalid literal");
            case 'f':
                if (literal("false", 5)) return jinja::mk_val<jinja::value_bool>(false);
                fail("invalid literal");
            case 'n':
                if (literal("null", 4)) return jinja::mk_val<jinja::value_none>();
                fail("invalid literal");
            default:
                return read_number();
        }
    }

    jinja::value read_array(int depth) {
        ++p_;   // [
        auto arr = jinja::mk_val<jinja::value_array>();
        skip_ws();
        if (p_ < end_ && *p_ == ']') {
            ++p_;
            return arr;
        }
        for (;;) {
            arr->push_back(read_value(depth + 1));
            skip_ws();
            if (p_ >= end_) {
                fail("unexpected end of input in array");
            }
            if (*p_ == ',') {
                ++p_;
            } else if (*p_ == ']') {
                ++p_;
                return arr;
            } else {
                fail("expected ',' or ']'");
            }
        }
    }

    jinja::value read_object(int depth) {
        ++p_;   // {
        auto obj = jinja::mk_val<jinja::value_object>();
        skip_ws();
        if (p_ < end_ && *p_ == '}') {
            ++p_;
            return obj;
        }
        std::string key;
        for (;;) {
            skip_ws();
            if (p_ >= end_ || *p_ != '"') {
                fail("expected a string key");
            }
            key.clear();
            read_string(key);
            skip_ws();
            if (p_ >= end_ || *p_ != ':') {
                fail("expected ':'");
            }
            ++p_;
            jinja::value val = read_value(depth + 1);
            auto it = keys_.find(key);
            if (it == keys_.end()) {
                it = keys_.emplace(key, jinja::mk_val<jinja::value_string>(key)).first;
            }
            obj->insert(it->second, val);
            skip_ws();
            if (p_ >= end_) {
                fail("unexpected end of input in object");
            }
            if (*p_ == ',') {
                ++p_;
            } else if (*p_ == '}') {
                ++p_;
                return obj;
            } else {
                fail("expected ',' or '}'");
            }
        }
    }

    static int hex4(const char* q) {
        int v = 0;
        for (int i = 0; i < 4; ++i) {
            const char c = q[i];
            v <<= 4;
            if (c >= '0' && c <= '9') v |= c - '0';
            else if (c >= 'a' && c <= 'f') v |= c - 'a' + 10;
            else if (c >= 'A' && c <= 'F') v |= c - 'A' + 10;
            else return -1;
        }
        return v;
    }
    static void put_utf8(std::string& out, uint32_t cp) {
        if (cp < 0x80) {
            out += static_cast<char>(cp);
        } else if (cp < 0x800) {
            out += static_cast<char>(0xC0 | (cp >> 6));
            out += static_cast<char>(0x80 | (cp & 0x3F));
        } else if (cp < 0x10000) {
            out += static_cast<char>(0xE0 | (cp >> 12));
            out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            out += static_cast<char>(0x80 | (cp & 0x3F));
        } else {
            out += static_cast<char>(0xF0 | (cp >> 18));
            out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
            out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            out += static_cast<char>(0x80 | (cp & 0x3F));
        }
    }
    // length of the well-formed UTF-8 sequence at q (lead byte >= 0x80), or 0 (RFC 3629: no overlongs, no
    // surrogates, nothing above U+10FFFF)
    size_t utf8_len(const char* q) const {
        const unsigned char c = static_cast<unsigned char>(*q);
        const size_t left = static_cast<size_t>(end_ - q);
        auto cont = [&](size_t i, unsigned lo = 0x80, unsigned hi = 0xBF) {
            return i < left && static_cast<unsigned char>(q[i]) >= lo && static_cast<unsigned char>(q[i]) <= hi;
        };
        if (c >= 0xC2 && c <= 0xDF) return cont(1) ? 2 : 0;
        if (c == 0xE0) return cont(1, 0xA0, 0xBF) && cont(2) ? 3 : 0;
        if (c == 0xED) return cont(1, 0x80, 0x9F) && cont(2) ? 3 : 0;
        if (c >= 0xE1 && c <= 0xEF) return cont(1) && cont(2) ? 3 : 0;
        if (c == 0xF0) return cont(1, 0x90, 0xBF) && cont(2) && cont(3) ? 4 : 0;
        if (c >= 0xF1 && c <= 0xF3) return cont(1) && cont(2) && cont(3) ? 4 : 0;
        if (c == 0xF4) return cont(1, 0x80, 0x8F) && cont(2) && cont(3) ? 4 : 0;
        return 0;
    }

    // p_ is at the opening quote; appends the decoded text to out and leaves p_ after the closing quote
    void read_string(std::string& out) {
        ++p_;
        for (;;) {
            // copy the run of plain characters in one go
            const char* run = p_;
            while (p_ < end_) {
                const unsigned char c = static_cast<unsigned char>(*p_);
                if (c == '"' || c == '\\' || c < 0x20) {
                    break;
                }
                if (c >= 0x80) {
                    const size_t n = utf8_len(p_);
                    if (n == 0) {
                        fail("invalid UTF-8 in a string");
                    }
                    p_ += n;
                } else {
                    ++p_;
                }
            }
            out.append(run, static_cast<size_t>(p_ - run));
            if (p_ >= end_) {
                fail("unterminated string");
            }
            const char c = *p_;
            if (c == '"') {
                ++p_;
                return;
            }
            if (c != '\\') {
                fail("control character in a string");
            }
            ++p_;
            if (p_ >= end_) {
                fail("unterminated string");
            }
            switch (*p_++) {
                case '"': out += '"'; break;
                case '\\': out += '\\'; break;
                case '/': out += '/'; break;
                case 'b': out += '\b'; break;
                case 'f': out += '\f'; break;
                case 'n': out += '\n'; break;
                case 'r': out += '\r'; break;
                case 't': out += '\t'; break;
                case 'u': {
                    if (end_ - p_ < 4) fail("bad \\u escape");
                    int cp = hex4(p_);
                    if (cp < 0) fail("bad \\u escape");
                    p_ += 4;
                    if (cp >= 0xD800 && cp <= 0xDBFF) {
                        if (end_ - p_ < 6 || p_[0] != '\\' || p_[1] != 'u') fail("lone surrogate");
                        const int lo = hex4(p_ + 2);
                        if (lo < 0xDC00 || lo > 0xDFFF) fail("lone surrogate");
                        p_ += 6;
                        put_utf8(out, 0x10000 + ((static_cast<uint32_t>(cp) - 0xD800) << 10) + (static_cast<uint32_t>(lo) - 0xDC00));
                    } else if (cp >= 0xDC00 && cp <= 0xDFFF) {
                        fail("lone surrogate");
                    } else {
                        put_utf8(out, static_cast<uint32_t>(cp));
                    }
                    break;
                }
                default:
                    fail("bad escape");
            }
        }
    }

    jinja::value read_number() {
        const char* start = p_;
        bool negative = false;
        if (p_ < end_ && *p_ == '-') {
            negative = true;
            ++p_;
        }
        const char* digits = p_;
        if (p_ >= end_ || *p_ < '0' || *p_ > '9') {
            fail("invalid number");
        }
        if (*p_ == '0') {
            ++p_;
        } else {
            while (p_ < end_ && *p_ >= '0' && *p_ <= '9') ++p_;
        }
        const char* int_end = p_;
        bool is_float = false;
        if (p_ < end_ && *p_ == '.') {
            is_float = true;
            ++p_;
            if (p_ >= end_ || *p_ < '0' || *p_ > '9') fail("invalid number");
            while (p_ < end_ && *p_ >= '0' && *p_ <= '9') ++p_;
        }
        if (p_ < end_ && (*p_ == 'e' || *p_ == 'E')) {
            is_float = true;
            ++p_;
            if (p_ < end_ && (*p_ == '+' || *p_ == '-')) ++p_;
            if (p_ >= end_ || *p_ < '0' || *p_ > '9') fail("invalid number");
            while (p_ < end_ && *p_ >= '0' && *p_ <= '9') ++p_;
        }
        if (!is_float) {
            // an integer: int64, else uint64 (read as int64 like nlohmann's get<int64_t>), else a double
            const size_t nd = static_cast<size_t>(int_end - digits);
            if (nd <= 18) {
                int64_t v = 0;
                for (const char* q = digits; q < int_end; ++q) v = v * 10 + (*q - '0');
                return jinja::mk_val<jinja::value_int>(negative ? -v : v);
            }
            uint64_t u = 0;
            bool overflow = false;
            for (const char* q = digits; q < int_end; ++q) {
                const unsigned d = static_cast<unsigned>(*q - '0');
                if (u > (UINT64_MAX - d) / 10) {
                    overflow = true;
                    break;
                }
                u = u * 10 + d;
            }
            if (!overflow) {
                if (negative) {
                    if (u <= static_cast<uint64_t>(INT64_MAX) + 1) {
                        return jinja::mk_val<jinja::value_int>(static_cast<int64_t>(0 - u));
                    }
                } else {
                    return jinja::mk_val<jinja::value_int>(static_cast<int64_t>(u));
                }
            }
        }
        const std::string tok(start, static_cast<size_t>(p_ - start));
        const double d = std::strtod(tok.c_str(), nullptr);
        if (!std::isfinite(d)) {
            fail("number overflow");     // (as nlohmann: JSON has no infinity)
        }
        return jinja::mk_val<jinja::value_float>(d);
    }
};

jinja::value parse_value(const char* s, const char* what) {
    try {
        return json_reader(s, std::strlen(s)).read_document();
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("invalid ") + what + " JSON: " + e.what());
    }
}

thread_local std::string g_error;
thread_local int g_kind = 0;
thread_local std::string g_result;

void fail(int kind, const std::string& msg) {
    g_kind = kind;
    g_error = msg;
}

}  // namespace

extern "C" {

int strata_template_abi_version(void) { return 1; }

const char* strata_template_error(void) { return g_error.c_str(); }

int strata_template_error_kind(void) { return g_kind; }

strata_template* strata_template_load(const char* utf8, int64_t len) {
    g_error.clear();
    g_kind = 0;
    try {
        if (!utf8 || len < 0) {
            throw std::runtime_error("no template source");
        }
        auto t = std::make_unique<strata_template>();
        t->src.assign(utf8, static_cast<size_t>(len));
        jinja::lexer lexer;
        auto lexed = lexer.tokenize(t->src);
        t->prog = jinja::parse_from_tokens(lexed);
        return t.release();
    } catch (const std::exception& e) {
        fail(2, e.what());
    } catch (...) {
        fail(2, "unknown error loading the template");
    }
    return nullptr;
}

void strata_template_free(strata_template* h) { delete h; }

int64_t strata_template_render(const strata_template* h, const char* messages_json, const char* tools_json,
                               const char* kwargs_json, int add_generation_prompt) {
    g_error.clear();
    g_kind = 0;
    g_result.clear();
    try {
        if (!h || !messages_json) {
            throw std::runtime_error("invalid argument");
        }
        auto* t = const_cast<strata_template*>(h);
        std::lock_guard<std::mutex> lock(t->mu);
        jinja::context ctx(t->src);
        ctx.set_val("messages", parse_value(messages_json, "messages"));
        ctx.set_val("tools", tools_json ? parse_value(tools_json, "tools") : jinja::mk_val<jinja::value_none>());
        ctx.set_val("add_generation_prompt", jinja::mk_val<jinja::value_bool>(add_generation_prompt != 0));
        if (kwargs_json) {
            jinja::value kw = parse_value(kwargs_json, "kwargs");
            if (!jinja::is_val<jinja::value_object>(kw)) {
                throw std::runtime_error("kwargs must be a JSON object");
            }
            for (const auto& kv : kw->as_ordered_object()) {
                ctx.set_val(kv.first, kv.second);
            }
        }
        jinja::runtime runtime(ctx);
        const jinja::value results = runtime.execute(t->prog);
        auto parts = jinja::runtime::gather_string_parts(results);
        g_result = parts->as_string().str();
        return static_cast<int64_t>(g_result.size());
    } catch (const std::exception& e) {
        // raise_exception(msg) throws raised_exception("Jinja Exception: " + msg), which statement::execute wraps once
        // per enclosing statement as "...\nError: <inner>" (with a source excerpt in front).  So the template's own
        // text is whatever follows the first "Error: Jinja Exception: ", to the end.
        static const std::string marker = "Error: Jinja Exception: ";
        const std::string m = e.what();
        const size_t at = m.find(marker);
        if (at != std::string::npos) {
            fail(1, m.substr(at + marker.size()));
        } else if (m.compare(0, 17, "Jinja Exception: ") == 0) {
            fail(1, m.substr(17));
        } else {
            fail(2, m);
        }
    } catch (...) {
        fail(2, "unknown error rendering the template");
    }
    return -1;
}

int64_t strata_template_result(char* buf, int64_t cap) {
    const int64_t n = static_cast<int64_t>(g_result.size());
    if (buf && cap > 0) {
        std::memcpy(buf, g_result.data(), static_cast<size_t>(n < cap ? n : cap));
    }
    return n;
}

}  // extern "C"
