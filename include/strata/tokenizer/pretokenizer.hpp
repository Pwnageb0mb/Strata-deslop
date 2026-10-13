// include/strata/tokenizer/pretokenizer.hpp - the pre-tokenizer, in C++.
//
// tools/strata_tokenizer.py is the reference: it cuts a prompt into pieces with Python's `regex` module and
// `QWEN35_PATTERN`, and the pieces decide the token ids, so a port has to cut in exactly the same places.
// This is that pattern, as a scanner, with the Unicode classes generated from the reference's own tables
// (include/strata/tokenizer/unicode_classes.hpp).
//
// The pieces are contiguous views into `text`; they cover it exactly, in order, with no gap and no overlap.
#pragma once

#include <cstddef>
#include <string_view>
#include <vector>

namespace strata::tokenizer {

// The qwen35 pre-tokenizer (llama.cpp's LLAMA_VOCAB_PRE_TYPE_QWEN35, transcribed into the reference).
// `text` must be valid UTF-8; an invalid byte is treated as one codepoint so the result still covers it.
std::vector<std::string_view> pretokenize(std::string_view text);

}  // namespace strata::tokenizer
