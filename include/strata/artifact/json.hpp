// include/strata/artifact/json.hpp - the smallest JSON reader that can read a safetensors header.
//
// The model-preparation tools read exactly two JSON documents: a safetensors header (tensor name ->
// {dtype, shape, data_offsets}) and model.safetensors.index.json's weight_map (tensor name -> shard file).
// Both are machine-generated, so this is deliberately a reader and not a library: no serialisation, no
// formatting, no DOM mutation, no schema.
//
// LIMIT, stated rather than hidden: a \u escape is decoded for the Basic Multilingual Plane, but a surrogate
// PAIR is not combined into one code point.  The documents this reads carry dtype names, tensor names and
// offsets - ASCII - and no surrogate escape appears in any of them.  If one ever does, this is the line to fix.
#pragma once

#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace strata {
namespace json {

struct Value {
    enum class Kind { Null, Bool, Num, Str, Arr, Obj };

    Kind kind = Kind::Null;
    bool boolean = false;
    double number = 0.0;
    std::string text;                                        // Str
    std::vector<Value> items;                                // Arr
    std::vector<std::pair<std::string, Value>> members;      // Obj, in document order

    bool is_obj() const { return kind == Kind::Obj; }
    bool is_arr() const { return kind == Kind::Arr; }
    bool is_str() const { return kind == Kind::Str; }
    bool is_num() const { return kind == Kind::Num; }

    // linear in the member count (document order is the storage); the headers it reads are small objects
    const Value* find(std::string_view key) const {
        for (const auto& m : members)
            if (m.first == key) return &m.second;
        return nullptr;
    }
};

class Parser {
public:
    explicit Parser(std::string_view text) : s_(text) {}

    Value document() {
        skip_ws();
        Value v = value();
        skip_ws();
        if (pos_ != s_.size()) fail("trailing input");
        return v;
    }

private:
    std::string_view s_;
    std::size_t pos_ = 0;

    [[noreturn]] void fail(const char* what) const {
        throw std::runtime_error(std::string("json: ") + what + " at offset " + std::to_string(pos_));
    }

    void skip_ws() {
        while (pos_ < s_.size()) {
            const char c = s_[pos_];
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r') ++pos_;
            else break;
        }
    }

    bool eat(char c) {
        skip_ws();
        if (pos_ < s_.size() && s_[pos_] == c) {
            ++pos_;
            return true;
        }
        return false;
    }

    void expect(char c) {
        if (!eat(c)) fail("expected a delimiter");
    }

    void literal(std::string_view word) {
        if (s_.compare(pos_, word.size(), word) != 0) fail("bad literal");
        pos_ += word.size();
    }

    Value value() {
        skip_ws();
        if (pos_ >= s_.size()) fail("unexpected end");
        switch (s_[pos_]) {
            case '{': return object();
            case '[': return array();
            case '"': {
                Value v;
                v.kind = Value::Kind::Str;
                v.text = string();
                return v;
            }
            case 't': literal("true"); {
                Value v;
                v.kind = Value::Kind::Bool;
                v.boolean = true;
                return v;
            }
            case 'f': literal("false"); {
                Value v;
                v.kind = Value::Kind::Bool;
                v.boolean = false;
                return v;
            }
            case 'n': literal("null"); return Value{};
            default: return number();
        }
    }

    // a document nested deeper than this is refused rather than recursed into until the stack runs out
    static constexpr int kMaxDepth = 256;
    struct Depth {
        int& d;
        Depth(int& d_, const Parser& p) : d(d_) { if (++d > kMaxDepth) p.fail("nesting too deep"); }
        ~Depth() { --d; }
    };
    int depth_ = 0;

    Value object() {
        Depth guard(depth_, *this);
        Value v;
        v.kind = Value::Kind::Obj;
        expect('{');
        if (eat('}')) return v;
        for (;;) {
            skip_ws();
            if (pos_ >= s_.size() || s_[pos_] != '"') fail("expected a key");
            std::string key = string();
            expect(':');
            v.members.emplace_back(std::move(key), value());
            if (eat(',')) continue;
            expect('}');
            return v;
        }
    }

    Value array() {
        Depth guard(depth_, *this);
        Value v;
        v.kind = Value::Kind::Arr;
        expect('[');
        if (eat(']')) return v;
        for (;;) {
            v.items.push_back(value());
            if (eat(',')) continue;
            expect(']');
            return v;
        }
    }

    static int hex_digit(char c) {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        return -1;
    }

    std::string string() {
        expect('"');
        std::string out;
        for (;;) {
            if (pos_ >= s_.size()) fail("unterminated string");
            const char c = s_[pos_++];
            if (c == '"') return out;
            if (c != '\\') {
                out += c;
                continue;
            }
            if (pos_ >= s_.size()) fail("unterminated escape");
            const char e = s_[pos_++];
            switch (e) {
                case '"': out += '"'; break;
                case '\\': out += '\\'; break;
                case '/': out += '/'; break;
                case 'b': out += '\b'; break;
                case 'f': out += '\f'; break;
                case 'n': out += '\n'; break;
                case 'r': out += '\r'; break;
                case 't': out += '\t'; break;
                case 'u': {
                    unsigned cp = 0;
                    for (int i = 0; i < 4; ++i) {
                        if (pos_ >= s_.size()) fail("short \\u escape");
                        const int d = hex_digit(s_[pos_++]);
                        if (d < 0) fail("bad \\u escape");
                        cp = cp * 16u + (unsigned) d;
                    }
                    utf8_append(out, cp);
                    break;
                }
                default: fail("bad escape");
            }
        }
    }

    static void utf8_append(std::string& out, unsigned cp) {
        if (cp < 0x80) {
            out += (char) cp;
        } else if (cp < 0x800) {
            out += (char) (0xC0 | (cp >> 6));
            out += (char) (0x80 | (cp & 0x3F));
        } else {
            out += (char) (0xE0 | (cp >> 12));
            out += (char) (0x80 | ((cp >> 6) & 0x3F));
            out += (char) (0x80 | (cp & 0x3F));
        }
    }

    Value number() {
        const std::size_t start = pos_;
        if (pos_ < s_.size() && (s_[pos_] == '-' || s_[pos_] == '+')) ++pos_;
        while (pos_ < s_.size()) {
            const char c = s_[pos_];
            if (std::isdigit((unsigned char) c) || c == '.' || c == 'e' || c == 'E' || c == '+' || c == '-') ++pos_;
            else break;
        }
        if (pos_ == start) fail("expected a number");
        Value v;
        v.kind = Value::Kind::Num;
        v.number = std::strtod(std::string(s_.substr(start, pos_ - start)).c_str(), nullptr);
        return v;
    }
};

inline Value parse(std::string_view text) { return Parser(text).document(); }

} // namespace json
} // namespace strata
