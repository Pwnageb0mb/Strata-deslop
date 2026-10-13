// src/tokenizer/gguf_vocab.cpp - see include/strata/tokenizer/gguf_vocab.hpp for what this owns and why.
#include "strata/tokenizer/gguf_vocab.hpp"

namespace strata::tokenizer {

Tokenizer tokenizer_from_gguf(const strata::GgufFile& g) {
    const strata::MetaValue* model = g.get("tokenizer.ggml.model");
    if (model == nullptr) {
        throw std::invalid_argument("the GGUF carries no tokenizer (tokenizer.ggml.model is absent)");
    }
    if (model->s != "gpt2") {
        throw std::invalid_argument("expected a byte-level BPE tokenizer (model 'gpt2'), got '" + model->s + "'");
    }
    // the pre-tokenizer is SHIPPED, not re-derived: this port implements qwen35's, so a model declaring another
    // one is refused rather than tokenized with a pattern that is not its own
    if (const strata::MetaValue* pre = g.get("tokenizer.ggml.pre")) {
        if (pre->s != "qwen35") {
            throw std::invalid_argument("this port has the qwen35 pre-tokenizer, and the GGUF declares '" +
                                        pre->s + "'");
        }
    }
    const strata::MetaValue* tokens = g.get("tokenizer.ggml.tokens");
    const strata::MetaValue* merges = g.get("tokenizer.ggml.merges");
    if (tokens == nullptr || merges == nullptr) {
        throw std::invalid_argument(std::string("the GGUF carries no tokenizer (") +
                                    (tokens == nullptr ? "tokenizer.ggml.tokens" : "tokenizer.ggml.merges") +
                                    " is absent)");
    }
    std::vector<std::string> tok;
    for (const strata::MetaValue& v : g.array_items("tokenizer.ggml.tokens")) tok.push_back(v.s);
    std::vector<std::string> mer;
    for (const strata::MetaValue& v : g.array_items("tokenizer.ggml.merges")) mer.push_back(v.s);
    // token_type is optional; when it is there, the CONSTRUCTOR derives the special literals from it (3 CONTROL,
    // 4 USER_DEFINED), which is exactly what the GGUF's array carries
    std::vector<int> types;
    for (const strata::MetaValue& v : g.array_items("tokenizer.ggml.token_type")) types.push_back((int) v.u);
    return Tokenizer(std::move(tok), std::move(mer), std::move(types));
}

Tokenizer tokenizer_from_gguf(const std::string& path) {
    const strata::GgufFile g(path);
    return tokenizer_from_gguf(g);
}

} // namespace strata::tokenizer
