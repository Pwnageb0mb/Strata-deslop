// src/tokenizer/pretokenizer.cpp - the qwen35 pre-tokenizer as a scanner.
//
// WHY A SCANNER AND NOT A REGEX LIBRARY.  The reference's pattern leans on \p{L}, \p{M}, \p{N} and \s, and the
// point of the port is to make the same choices the reference makes, not a similar-looking set.  std::regex has
// no Unicode properties, and an ICU dependency would be a whole library for six branches.  The classes come from
// the reference instead, generated into a 9 KB header (tools/gen_unicode_classes.py).
//
// WHY THE ORDER AND THE BACKTRACKING MATTER.  Python's `regex` takes the FIRST alternative that matches, not the
// longest, and a greedy quantifier gives characters back when the rest of the branch fails.  Two consequences
// that a "reasonable-looking" implementation gets wrong, both in the golden file:
//
//   "a  b"  ->  {"a", " ", " b"}    the last space is absorbed into the word by branch 2; the FIRST space cannot
//                                   be, so branch 6 matches it - and branch 6 matches one character short, since
//                                   \s+ must give a character back before (?!\S) can succeed
//   "12ab"  ->  {"1", "2", "ab"}    branch 3 is a single \p{N}, not a run of them
//
// Each branch below is written to make the same choice, and tests/tokenizer/pretokenizer_test.cpp compares the
// whole corpus against the reference's own output.
#include "strata/tokenizer/pretokenizer.hpp"

#include "strata/tokenizer/unicode_classes.hpp"
#include "strata/tokenizer/utf8.hpp"

#include <vector>

namespace strata::tokenizer {
namespace {

using unicode::is_letter;
using unicode::is_mark;
using unicode::is_number;
using unicode::is_space;
using utf8::Cps;

bool is_lm(std::uint32_t cp) { return is_letter(cp) || is_mark(cp); }
bool is_cr(std::uint32_t cp) { return cp == '\r' || cp == '\n'; }

// branch 2's optional leading character: anything that is not CR, LF, a letter or a number.  Whitespace and
// marks qualify, which is why " the" and "a\u00a0b" keep their space as part of the word.
bool may_precede_word(std::uint32_t cp) { return !is_cr(cp) && !is_letter(cp) && !is_number(cp); }

// branch 4's run: not whitespace, not a letter, not a mark, not a number - the punctuation and symbol run.
bool is_punct(std::uint32_t cp) { return !is_space(cp) && !is_letter(cp) && !is_mark(cp) && !is_number(cp); }

std::size_t run_while(const Cps& c, std::size_t i, bool (*pred)(std::uint32_t)) {
    const std::size_t n = c.cp.size();
    while (i < n && pred(c.cp[i])) {
        ++i;
    }
    return i;
}

// branch 1: the English contractions, each a fixed literal.  Longest spelling of a shared prefix first, or
// "'ll" would stop at "'l"'s two-character branch and "we'll" would cut in the middle of itself.
std::size_t contraction(const Cps& c, std::size_t i) {
    const std::size_t n = c.cp.size();
    if (i >= n || c.cp[i] != '\'') {
        return i;
    }
    const auto at = [&](std::size_t k) { return k < n ? c.cp[k] : 0u; };
    const auto lower = [](std::uint32_t x) { return x >= 'A' && x <= 'Z' ? x + 32 : x; };
    const auto is = [&](std::size_t k, char want) { return lower(at(k)) == static_cast<std::uint32_t>(want); };
    if (is(i + 1, 'l') && is(i + 2, 'l')) {
        return i + 3;
    }
    if (is(i + 1, 'r') && is(i + 2, 'e')) {
        return i + 3;
    }
    if (is(i + 1, 'v') && is(i + 2, 'e')) {
        return i + 3;
    }
    for (const char one : {'s', 't', 'm', 'd'}) {
        if (is(i + 1, one)) {
            return i + 2;
        }
    }
    return i;
}

// Which branch matches at codepoint `i`, and where it ends.  Returned as an index; `i` means "none".
std::size_t match_at(const Cps& c, std::size_t i) {
    const std::size_t n = c.cp.size();

    const std::size_t conj = contraction(c, i);
    if (conj > i) {
        return conj;
    }

    // branch 2: one optional non-CR/LF/letter/number character, then at least one letter or mark.  The
    // optional character is preferred (that is what makes " the" one piece), and is given back when nothing
    // letter-like follows it.
    if (i < n && may_precede_word(c.cp[i])) {
        const std::size_t with = run_while(c, i + 1, is_lm);
        if (with > i + 1) {
            return with;
        }
    }
    const std::size_t bare = run_while(c, i, is_lm);
    if (bare > i) {
        return bare;
    }

    // branch 3: a single number character, on its own.
    if (i < n && is_number(c.cp[i])) {
        return i + 1;
    }

    // branch 4: an optional space, a run of punctuation, then any CR/LF run.
    if (i < n && c.cp[i] == ' ') {
        const std::size_t after = run_while(c, i + 1, is_punct);
        if (after > i + 1) {
            return run_while(c, after, is_cr);
        }
    }
    const std::size_t punct = run_while(c, i, is_punct);
    if (punct > i) {
        return run_while(c, punct, is_cr);
    }

    // branch 5: whitespace up to and including its last newline.  (Not the whole whitespace run when
    // something that is not a newline follows the last one: " \n " matches " \n", and the space after it is
    // branch 6's or 7's.)
    if (i < n && is_space(c.cp[i])) {
        const std::size_t end = run_while(c, i, is_space);
        std::size_t last_nl = 0;
        bool any = false;
        for (std::size_t k = i; k < end; ++k) {
            if (is_cr(c.cp[k])) {
                last_nl = k;
                any = true;
            }
        }
        if (any) {
            return last_nl + 1;
        }

        // branch 6: whitespace that runs to the end of the text - one character short of the run otherwise,
        // because \s+ has to give a character back before (?!\S) can hold.
        if (end >= n) {
            return end;
        }
        if (end > i + 1) {
            return end - 1;
        }
    }

    // branch 7: any other run of whitespace.
    const std::size_t ws = run_while(c, i, is_space);
    return ws > i ? ws : i;
}

}  // namespace

std::vector<std::string_view> pretokenize(std::string_view text) {
    const Cps c = utf8::decode(text);
    std::vector<std::string_view> out;
    std::size_t i = 0;
    while (i < c.cp.size()) {
        std::size_t end = match_at(c, i);
        if (end <= i) {
            end = i + 1;                       // no branch matched: never drop a character on the floor
        }
        out.emplace_back(text.data() + c.off[i], c.off[end] - c.off[i]);
        i = end;
    }
    return out;
}

}  // namespace strata::tokenizer
