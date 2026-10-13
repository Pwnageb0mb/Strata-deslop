// include/strata/tokenizer/gguf_vocab.hpp - a Tokenizer built from a model file's vocabulary.
//
// The engine does not tokenize (src/program/generate.cpp takes a prompt as token ids), so the C++ Tokenizer had
// no way to load a real vocabulary: the port's test builds vocabs from a golden.  A tool that works on a MODEL's
// vocabulary - the draft-vocabulary builder is the first - needs the GGUF's tokenizer metadata, and that is this.
//
// It is its own header so `tokenizer.hpp` keeps no dependency on the GGUF reader: the algorithm does not care
// where the vocabulary came from, and a tool that has a vocabulary some other way should not have to link one.
#pragma once

#include <stdexcept>
#include <string>
#include <vector>

#include "strata/artifact/gguf_reader.hpp"
#include "strata/tokenizer/tokenizer.hpp"

namespace strata::tokenizer {

/// The Tokenizer a GGUF describes.  Throws std::invalid_argument when the file carries no tokenizer, is not a
/// byte-level BPE (`tokenizer.ggml.model` != "gpt2"), or declares a pre-tokenizer this port does not implement -
/// the port has the qwen35 pre-tokenizer, and tokenizing another type's text with it would produce plausible
/// wrong ids rather than an error.
Tokenizer tokenizer_from_gguf(const strata::GgufFile& g);

/// The same, opening the file first.
Tokenizer tokenizer_from_gguf(const std::string& path);

} // namespace strata::tokenizer
