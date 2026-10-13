// src/text/pytext.hpp - Python str / json semantics that the output-side port must reproduce byte for byte.
//
// Internal to libstrata_text (not installed).  Texts are UTF-8 std::string; a Python str slice by code points
// becomes a slice by bytes, which is the same thing as long as the cut falls on a character boundary, and every
// cut the parser makes does (it cuts at ASCII tags and at whitespace).  Lone surrogates, which a Python str may
// hold, travel as their 3-byte (CESU/WTF-8) form and are decoded as one character.
#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace strata::text {

// Python's ValueError: the places where the reference raises one (a malformed tool call).
struct ValueErr : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// --- UTF-8 -------------------------------------------------------------------------------------------------
// The code point starting at s[i] and its byte length.  Lenient: a stray byte is its own length-1 "character".
inline uint32_t next_cp(std::string_view s, size_t i, size_t* len) {
    unsigned char c = static_cast<unsigned char>(s[i]);
    size_t n = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 1;
    if (n == 1 || i + n > s.size()) {
        *len = 1;
        return c;
    }
    uint32_t cp = c & (0xFF >> (n + 1));
    for (size_t k = 1; k < n; ++k) {
        unsigned char d = static_cast<unsigned char>(s[i + k]);
        if ((d & 0xC0) != 0x80) {
            *len = 1;
            return c;
        }
        cp = (cp << 6) | (d & 0x3F);
    }
    *len = n;
    return cp;
}

inline void append_cp(std::string& out, uint32_t cp) {
    if (cp < 0x80) {
        out += static_cast<char>(cp);
    } else if (cp < 0x800) {
        out += static_cast<char>(0xC0 | (cp >> 6));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {  // includes surrogates (surrogatepass)
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

bool is_space_cp(uint32_t cp);  // str.isspace() == re \s            (stream.cpp, from text_tables.inc)
bool is_word_cp(uint32_t cp);   // re \w
uint32_t lower_cp(uint32_t cp); // the single-code-point str.lower() mapping (U+0130 excepted)
bool is_cased_ni(uint32_t cp);
bool is_ignorable(uint32_t cp);

// str.lstrip() / rstrip() / strip() with no argument: Unicode whitespace.
inline size_t lstrip_pos(std::string_view s) {
    size_t i = 0;
    while (i < s.size()) {
        size_t n;
        uint32_t cp = next_cp(s, i, &n);
        if (cp < 0x80 ? !(cp == ' ' || (cp >= 9 && cp <= 13) || (cp >= 0x1C && cp <= 0x1F)) : !is_space_cp(cp)) break;
        i += n;
    }
    return i;
}
inline size_t rstrip_end(std::string_view s) {  // the length after trailing whitespace is removed
    size_t last = 0;
    for (size_t i = 0; i < s.size();) {
        size_t n;
        uint32_t cp = next_cp(s, i, &n);
        i += n;
        if (!is_space_cp(cp)) last = i;
    }
    return last;
}
inline std::string_view lstrip(std::string_view s) { return s.substr(lstrip_pos(s)); }
inline std::string_view strip(std::string_view s) {
    s = lstrip(s);
    return s.substr(0, rstrip_end(s));
}
inline bool has_space(std::string_view s) {  // any(c.isspace() for c in s)
    for (size_t i = 0; i < s.size();) {
        size_t n;
        if (is_space_cp(next_cp(s, i, &n))) return true;
        i += n;
    }
    return false;
}
inline bool starts_with(std::string_view s, std::string_view p) { return s.substr(0, p.size()) == p; }
// p.startswith(s) for a possibly shorter s: s is a prefix of p (empty counts)
inline bool prefix_of(std::string_view s, std::string_view p) { return s.size() <= p.size() && p.substr(0, s.size()) == s; }

// --- JSON, as Python's json module reads and writes it ------------------------------------------------------
struct JVal {
    enum Kind { Null, Bool, Int, Float, Str, Arr, Obj } kind = Null;
    bool b = false;
    double d = 0;
    std::string s;  // Str: the text; Int: the canonical digits
    std::vector<JVal> a;
    std::vector<std::pair<std::string, JVal>> o;  // insertion order, one entry per key
    const JVal* get(std::string_view key) const {
        for (auto& m : o)
            if (m.first == key) return &m.second;
        return nullptr;
    }
};

// json.loads(text); false where it raises ValueError (also past a nesting depth of 500, where Python would
// raise RecursionError - the one place the port is not exact).
bool json_loads(std::string_view text, JVal* out);
// json.dumps(v, ensure_ascii=ascii) with Python's default separators (", " and ": ") or compact ones.
void json_dumps(const JVal& v, bool ensure_ascii, bool compact, std::string& out);
// json.dumps(str) without the quotes
void json_escape(std::string_view s, bool ensure_ascii, std::string& out);

}  // namespace strata::text
