// src/text/pytext.cpp - Python-exact character classes, lower() and json (see pytext.hpp).
#include "pytext.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdlib>

namespace strata::text {

namespace {
#include "text_tables.inc"

bool in_ranges(const Range* t, size_t n, uint32_t cp) {
    size_t lo = 0, hi = n;
    while (lo < hi) {
        size_t mid = (lo + hi) / 2;
        if (cp < t[mid].lo) hi = mid;
        else if (cp > t[mid].hi) lo = mid + 1;
        else return true;
    }
    return false;
}
}  // namespace

bool is_space_cp(uint32_t cp) { return in_ranges(kSpace, kSpaceCount, cp); }
bool is_word_cp(uint32_t cp) { return in_ranges(kWord, kWordCount, cp); }
bool is_cased_ni(uint32_t cp) { return in_ranges(kCasedNI, kCasedNICount, cp); }
bool is_ignorable(uint32_t cp) { return in_ranges(kIgnorable, kIgnorableCount, cp); }
uint32_t lower_cp(uint32_t cp) {
    if (cp < 0x80) return (cp >= 'A' && cp <= 'Z') ? cp + 32 : cp;
    size_t lo = 0, hi = kLowerCount;
    while (lo < hi) {
        size_t mid = (lo + hi) / 2;
        if (cp < kLower[mid].from) hi = mid;
        else if (cp > kLower[mid].from) lo = mid + 1;
        else return kLower[mid].to;
    }
    return cp;
}

// ------------------------------------------------------------------------------------------------ json
namespace {

constexpr int kMaxDepth = 500;
constexpr size_t kMaxIntDigits = 4300;  // Python's int/str conversion limit: a longer int is a ValueError

struct Reader {
    std::string_view s;
    size_t i = 0;

    void ws() {
        while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r')) ++i;
    }
    bool lit(std::string_view w) {
        if (s.substr(i, w.size()) == w) {
            i += w.size();
            return true;
        }
        return false;
    }
    static int hex(char c) {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        return -1;
    }
    bool hex4(size_t at, uint32_t* v) const {
        if (at + 4 > s.size()) return false;
        uint32_t r = 0;
        for (size_t k = 0; k < 4; ++k) {
            int h = hex(s[at + k]);
            if (h < 0) return false;
            r = r * 16 + static_cast<uint32_t>(h);
        }
        *v = r;
        return true;
    }
    bool str(std::string& out) {  // s[i] is the opening quote
        ++i;
        for (;;) {
            if (i >= s.size()) return false;
            unsigned char c = static_cast<unsigned char>(s[i]);
            if (c == '"') {
                ++i;
                return true;
            }
            if (c < 0x20) return false;
            if (c != '\\') {
                out += static_cast<char>(c);
                ++i;
                continue;
            }
            if (++i >= s.size()) return false;
            char e = s[i];
            switch (e) {
                case '"': out += '"'; ++i; break;
                case '\\': out += '\\'; ++i; break;
                case '/': out += '/'; ++i; break;
                case 'b': out += '\b'; ++i; break;
                case 'f': out += '\f'; ++i; break;
                case 'n': out += '\n'; ++i; break;
                case 'r': out += '\r'; ++i; break;
                case 't': out += '\t'; ++i; break;
                case 'u': {
                    uint32_t u;
                    if (!hex4(i + 1, &u)) return false;
                    i += 5;
                    if (u >= 0xD800 && u <= 0xDBFF && s.substr(i, 2) == "\\u") {
                        uint32_t u2;
                        if (hex4(i + 2, &u2) && u2 >= 0xDC00 && u2 <= 0xDFFF) {
                            u = 0x10000 + (((u - 0xD800) << 10) | (u2 - 0xDC00));
                            i += 6;
                        }
                    }
                    append_cp(out, u);
                    break;
                }
                default: return false;
            }
        }
    }
    bool number(JVal& v) {
        size_t st = i;
        bool neg = false;
        if (s[i] == '-') {
            neg = true;
            ++i;
            if (i >= s.size()) return false;
            if (s[i] == 'I') {
                if (!lit("Infinity")) return false;
                v.kind = JVal::Float;
                v.d = -INFINITY;
                return true;
            }
        }
        if (i >= s.size() || s[i] < '0' || s[i] > '9') return false;
        size_t ds = i;
        if (s[i] == '0') ++i;
        else while (i < s.size() && s[i] >= '0' && s[i] <= '9') ++i;
        size_t de = i;
        bool flt = false;
        if (i + 1 < s.size() && s[i] == '.' && s[i + 1] >= '0' && s[i + 1] <= '9') {
            flt = true;
            i += 2;
            while (i < s.size() && s[i] >= '0' && s[i] <= '9') ++i;
        }
        if (i < s.size() && (s[i] == 'e' || s[i] == 'E')) {
            size_t j = i + 1;
            if (j < s.size() && (s[j] == '+' || s[j] == '-')) ++j;
            if (j < s.size() && s[j] >= '0' && s[j] <= '9') {
                while (j < s.size() && s[j] >= '0' && s[j] <= '9') ++j;
                flt = true;
                i = j;
            }
        }
        if (!flt) {
            if (de - ds > kMaxIntDigits) return false;
            v.kind = JVal::Int;
            v.s = (neg && !(de - ds == 1 && s[ds] == '0') ? "-" : "") + std::string(s.substr(ds, de - ds));
            return true;
        }
        v.kind = JVal::Float;
        double d = 0;
        const char* b = s.data() + st;
        auto r = std::from_chars(b, s.data() + i, d);
        if (r.ec != std::errc()) d = std::strtod(std::string(s.substr(st, i - st)).c_str(), nullptr);  // inf / denormal
        v.d = d;
        return true;
    }
    bool value(JVal& v, int depth) {
        if (depth > kMaxDepth || i >= s.size()) return false;
        char c = s[i];
        if (c == '"') {
            v.kind = JVal::Str;
            return str(v.s);
        }
        if (c == '{') {
            ++i;
            v.kind = JVal::Obj;
            ws();
            if (i < s.size() && s[i] == '}') {
                ++i;
                return true;
            }
            for (;;) {
                ws();
                if (i >= s.size() || s[i] != '"') return false;
                std::string key;
                if (!str(key)) return false;
                ws();
                if (i >= s.size() || s[i] != ':') return false;
                ++i;
                ws();
                JVal val;
                if (!value(val, depth + 1)) return false;
                bool dup = false;
                for (auto& m : v.o)
                    if (m.first == key) {
                        m.second = std::move(val);
                        dup = true;
                        break;
                    }
                if (!dup) v.o.emplace_back(std::move(key), std::move(val));
                ws();
                if (i >= s.size()) return false;
                if (s[i] == ',') { ++i; continue; }
                if (s[i] == '}') { ++i; return true; }
                return false;
            }
        }
        if (c == '[') {
            ++i;
            v.kind = JVal::Arr;
            ws();
            if (i < s.size() && s[i] == ']') {
                ++i;
                return true;
            }
            for (;;) {
                ws();
                JVal val;
                if (!value(val, depth + 1)) return false;
                v.a.push_back(std::move(val));
                ws();
                if (i >= s.size()) return false;
                if (s[i] == ',') { ++i; continue; }
                if (s[i] == ']') { ++i; return true; }
                return false;
            }
        }
        if (c == 'n') { v.kind = JVal::Null; return lit("null"); }
        if (c == 't') { v.kind = JVal::Bool; v.b = true; return lit("true"); }
        if (c == 'f') { v.kind = JVal::Bool; v.b = false; return lit("false"); }
        if (c == 'N') { v.kind = JVal::Float; v.d = NAN; return lit("NaN"); }
        if (c == 'I') { v.kind = JVal::Float; v.d = INFINITY; return lit("Infinity"); }
        if (c == '-' || (c >= '0' && c <= '9')) return number(v);
        return false;
    }
};

// repr(float), as Python writes it
void float_repr(double d, std::string& out) {
    if (std::isnan(d)) { out += "NaN"; return; }
    if (std::isinf(d)) { out += d < 0 ? "-Infinity" : "Infinity"; return; }
    char buf[64];
    auto r = std::to_chars(buf, buf + sizeof buf, d, std::chars_format::scientific);
    std::string t(buf, r.ptr);  // [-]d[.ddd]e[+-]XX
    bool neg = t[0] == '-';
    if (neg) t.erase(0, 1);
    size_t e = t.find('e');
    int ex = std::atoi(t.c_str() + e + 1);
    std::string digits;
    for (size_t k = 0; k < e; ++k)
        if (t[k] != '.') digits += t[k];
    if (neg) out += '-';
    int decpt = ex + 1;
    if (decpt > -4 && decpt <= 16) {
        if (decpt <= 0) {
            out += "0.";
            out.append(static_cast<size_t>(-decpt), '0');
            out += digits;
        } else if (static_cast<size_t>(decpt) >= digits.size()) {
            out += digits;
            out.append(static_cast<size_t>(decpt) - digits.size(), '0');
            out += ".0";
        } else {
            out.append(digits, 0, static_cast<size_t>(decpt));
            out += '.';
            out.append(digits, static_cast<size_t>(decpt), std::string::npos);
        }
    } else {
        out += digits[0];
        if (digits.size() > 1) {
            out += '.';
            out.append(digits, 1, std::string::npos);
        }
        out += 'e';
        out += ex < 0 ? '-' : '+';
        int a = std::abs(ex);
        if (a < 10) out += '0';
        out += std::to_string(a);
    }
}

void put_u(std::string& out, uint32_t u) {
    static const char* H = "0123456789abcdef";
    out += "\\u";
    for (int sh = 12; sh >= 0; sh -= 4) out += H[(u >> sh) & 15];
}

}  // namespace

bool json_loads(std::string_view text, JVal* out) {
    Reader r{text};
    r.ws();
    JVal v;
    if (!r.value(v, 0)) return false;
    r.ws();
    if (r.i != text.size()) return false;
    *out = std::move(v);
    return true;
}

void json_escape(std::string_view s, bool ensure_ascii, std::string& out) {
    for (size_t i = 0; i < s.size();) {
        unsigned char c = static_cast<unsigned char>(s[i]);
        if (c < 0x80) {
            ++i;
            switch (c) {
                case '"': out += "\\\""; break;
                case '\\': out += "\\\\"; break;
                case '\n': out += "\\n"; break;
                case '\r': out += "\\r"; break;
                case '\t': out += "\\t"; break;
                case '\b': out += "\\b"; break;
                case '\f': out += "\\f"; break;
                default:
                    if (c < 0x20) put_u(out, c);
                    else out += static_cast<char>(c);
            }
            continue;
        }
        size_t n;
        uint32_t cp = next_cp(s, i, &n);
        if (!ensure_ascii) {
            out.append(s.data() + i, n);
        } else if (cp >= 0x10000) {
            cp -= 0x10000;
            put_u(out, 0xD800 + (cp >> 10));
            put_u(out, 0xDC00 + (cp & 0x3FF));
        } else {
            put_u(out, cp);
        }
        i += n;
    }
}

void json_dumps(const JVal& v, bool ensure_ascii, bool compact, std::string& out) {
    switch (v.kind) {
        case JVal::Null: out += "null"; break;
        case JVal::Bool: out += v.b ? "true" : "false"; break;
        case JVal::Int: out += v.s; break;
        case JVal::Float: float_repr(v.d, out); break;
        case JVal::Str:
            out += '"';
            json_escape(v.s, ensure_ascii, out);
            out += '"';
            break;
        case JVal::Arr:
            out += '[';
            for (size_t k = 0; k < v.a.size(); ++k) {
                if (k) out += compact ? "," : ", ";
                json_dumps(v.a[k], ensure_ascii, compact, out);
            }
            out += ']';
            break;
        case JVal::Obj:
            out += '{';
            for (size_t k = 0; k < v.o.size(); ++k) {
                if (k) out += compact ? "," : ", ";
                out += '"';
                json_escape(v.o[k].first, ensure_ascii, out);
                out += compact ? "\":" : "\": ";
                json_dumps(v.o[k].second, ensure_ascii, compact, out);
            }
            out += '}';
            break;
    }
}

}  // namespace strata::text
