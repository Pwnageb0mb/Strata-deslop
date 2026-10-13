// include/strata/tokenizer/tokenizer.hpp - byte-level BPE, in C++.
//
// The reference is tools/strata_tokenizer.py: it reads the pack's `tokenizer/` directory (a byte-level BPE
// vocabulary: 256 byte tokens, the merges, the token types) and turns text into ids.  This is the same
// algorithm, checked against the reference's own output by tests/tokenizer/tokenizer_test.cpp.
//
// The pieces the text is cut into come from pretokenize() (the qwen35 pre-tokenizer, also in C++); each piece is
// byte-mapped, merged by LOWEST RANK first, and looked up.  The reference's own docstring names the traps: a
// merge applied in list order rather than rank order gives a different segmentation and a plausible token
// count, and the two classes of literal - type 4 matched always, type 3 only with parse_special - are not
// interchangeable.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace strata::tokenizer {

// One literal from the vocabulary that is matched as itself instead of being run through BPE.  `type` is the
// GGUF token_type: 3 CONTROL (`<|im_start|>`), 4 USER_DEFINED (`<tool_response>`).
struct Special {
    std::string text;
    int id;
    int type;
};

class Tokenizer {
public:
    // Throws std::invalid_argument on a duplicate token, a merge that is not a pair, or a merge naming a token
    // outside the vocabulary - the reference refuses all three at construction rather than producing a
    // plausible wrong id later.
    Tokenizer(std::vector<std::string> tokens, std::vector<std::string> merges,
              std::vector<int> token_types = {});

    // The ids for `text`.  `plain`: (start, end) byte spans that stay ordinary text even where they spell a
    // literal, so a quoted `</think>` in a message is tokenized with the text around it (#537).
    std::vector<int> encode(std::string_view text, bool parse_special = false,
                            const std::vector<std::pair<std::size_t, std::size_t>>& plain = {}) const;

    // The text for `ids`; invalid UTF-8 becomes U+FFFD.
    std::string decode(const std::vector<int>& ids) const;

    // One token's raw bytes.  Throws std::out_of_range outside the vocabulary, like the reference's IndexError.
    std::string token_bytes(int id) const;

    std::size_t size() const { return tokens_.size(); }
    const std::deque<std::string>& tokens() const { return tokens_; }

private:
    std::vector<int> encode_plain(std::string_view text) const;
    std::vector<int> merge_scan(std::vector<int> syms) const;               // pieces up to kScanMax symbols
    std::vector<int> merge_heap(const std::vector<int>& syms) const;        // longer ones, in O(n log n)
    int merged(std::size_t rank) const;

    // A piece with more symbols than this goes to the heap.  The reference's HEAP_MIN is 64, and the two paths
    // must agree - the golden's long cases exist to hold them to it.
    static constexpr std::size_t kScanMax = 64;
    // Pre-tokenizer pieces each thread remembers (the reference's PIECE_CACHE_MAX).
    static constexpr std::size_t kPieceCacheMax = 200000;

    // A deque, not a vector: `ids_` keys are views into these strings, and moving a deque leaves its elements
    // where they are, so a Tokenizer stays usable after it is moved (into a vector, say).  A vector would move
    // its strings - and every short one's buffer lives inside the string object - leaving the keys dangling.
    std::deque<std::string> tokens_;
    std::unordered_map<std::string_view, int> ids_;
    std::unordered_map<std::uint64_t, int> rank_;   // (left id, right id) -> merge rank
    std::vector<int> merge_to_;                     // rank -> the merged token's id, -1 when it is not in the vocabulary
    std::vector<int> byte_id_;                      // 256: the token id a byte maps to, -1 when it is not in the vocabulary
    std::vector<Special> specials_;                 // longest text first
    std::array<bool, 256> special_first_{};         // bytes some literal starts with
    std::uint64_t uid_ = 0;                         // per instance: keys the thread-local piece cache
};

}  // namespace strata::tokenizer
