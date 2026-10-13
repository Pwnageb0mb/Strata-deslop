// include/strata/tokenizer/utf8.hpp - UTF-8, in one place.
//
// The pre-tokenizer walks a text as codepoints and the tokenizer turns a token's text back into bytes, so both
// need the same decoder.  One implementation, so the two cannot drift apart.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace strata::tokenizer::utf8 {

// A text as codepoints, with the byte offset of each one.  off[cp.size()] is the text's length, so the codepoint
// at index i covers the bytes [off[i], off[i + 1]) - which is how a scanner turns an index range back into a
// substring.  An invalid byte stands alone as its own codepoint rather than being dropped, so the offsets always
// cover the input exactly.
struct Cps {
    std::vector<std::uint32_t> cp;
    std::vector<std::size_t> off;
    std::size_t size() const { return cp.size(); }
};

Cps decode(std::string_view text);

// One codepoint as UTF-8.  A surrogate or a value past U+10FFFF is not encodable and becomes U+FFFD.
void append(std::string& out, std::uint32_t cp);

// Bytes to text, with every invalid byte standing in as U+FFFD (Python's errors="replace").  The byte-level
// tokenizer never produces invalid UTF-8 itself, so this only matters for a vocabulary that does.
std::string decode_replacing(std::string_view bytes);

// True when `bytes` is a whole number of UTF-8 characters: no truncated sequence, no overlong form, no surrogate
// and nothing past U+10FFFF.  This is what `bytes.decode("utf-8")` accepts, and what a vocabulary tool needs to
// tell a token that spells a character from a token that spells PART of one (a BPE piece of a multi-byte
// character is not text, and must not be read as one).
bool is_whole_utf8(std::string_view bytes);

}  // namespace strata::tokenizer::utf8
