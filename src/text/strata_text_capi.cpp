// src/text/strata_text_capi.cpp - libstrata_text: the tokenizer behind a C ABI (include/strata/text/strata_text.h).
//
// Reads the pack's tokenizer directory exactly as serve/server.py did for strata_tokenizer.Tokenizer:
//   vocab.json       {token: id}, written in id order; the vector of tokens is indexed by id
//   merges.txt       one "left right" pair per line, split on "\n" (the writer joins with "\n", no final newline;
//                    one empty final line is tolerated, an empty line anywhere else is an error, like the reference)
//   token_type.json  an array of GGUF token types, or null
#include "strata/text/strata_text.h"

#include "strata/artifact/json.hpp"
#include "strata/tokenizer/tokenizer.hpp"

#include <cmath>
#include <cstring>
#include <exception>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

struct strata_text {
    strata::tokenizer::Tokenizer tok;
    std::vector<int> types;
    explicit strata_text(strata::tokenizer::Tokenizer t, std::vector<int> ty) : tok(std::move(t)), types(std::move(ty)) {}
};

namespace {

thread_local std::string g_error;

std::string read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("cannot read " + path);
    }
    return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

// json.hpp does not skip a UTF-8 BOM; Python's read_text(encoding="utf-8") does not either, but be explicit.
std::string_view without_bom(const std::string& s) {
    if (s.size() >= 3 && s.compare(0, 3, "\xEF\xBB\xBF") == 0) {
        return std::string_view(s).substr(3);
    }
    return s;
}


}  // namespace

extern "C" {

int strata_text_abi_version(void) { return 1; }

const char* strata_text_error(void) { return g_error.c_str(); }

strata_text* strata_text_load(const char* dir) {
    g_error.clear();
    try {
        if (dir == nullptr) {
            throw std::invalid_argument("no tokenizer directory given");
        }
        const std::string base = std::string(dir) + "/";

        const std::string vocab_text = read_file(base + "vocab.json");
        const strata::json::Value vocab = strata::json::parse(without_bom(vocab_text));
        if (!vocab.is_obj()) {
            throw std::runtime_error("vocab.json is not an object");
        }
        std::vector<std::string> tokens(vocab.members.size());
        std::vector<bool> seen(tokens.size(), false);
        for (const auto& m : vocab.members) {
            if (!m.second.is_num()) {
                throw std::runtime_error("vocab.json: id of " + m.first + " is not a number");
            }
            const double d = m.second.number;
            if (d < 0 || d >= static_cast<double>(tokens.size()) || d != std::floor(d)) {
                throw std::runtime_error("vocab.json: id of " + m.first + " is outside 0.." +
                                         std::to_string(tokens.size()));
            }
            const std::size_t id = static_cast<std::size_t>(d);
            if (seen[id]) {
                throw std::runtime_error("vocab.json: two tokens have id " + std::to_string(id));
            }
            seen[id] = true;
            tokens[id] = m.first;
        }

        const std::string merge_text = read_file(base + "merges.txt");
        std::string_view mt = without_bom(merge_text);
        std::vector<std::string> merges;
        std::size_t start = 0;
        while (start <= mt.size()) {
            std::size_t nl = mt.find('\n', start);
            const bool last = nl == std::string_view::npos;
            const std::string_view line = mt.substr(start, last ? std::string_view::npos : nl - start);
            if (!(last && line.empty() && !merges.empty())) {      // the empty tail after a final newline
                merges.emplace_back(line);
            }
            if (last) {
                break;
            }
            start = nl + 1;
        }
        if (merges.size() == 1 && merges[0].empty()) {
            merges.clear();                                          // an empty file is no merges
        }

        std::vector<int> types;
        const std::string type_text = read_file(base + "token_type.json");
        const strata::json::Value tv = strata::json::parse(without_bom(type_text));
        if (tv.is_arr()) {
            for (const auto& v : tv.items) {
                types.push_back(v.is_num() ? static_cast<int>(v.number) : 0);
            }
        }

        strata::tokenizer::Tokenizer t(std::move(tokens), std::move(merges), types);
        return new strata_text(std::move(t), std::move(types));
    } catch (const std::exception& e) {
        g_error = e.what();
    } catch (...) {
        g_error = "unknown error loading the tokenizer";
    }
    return nullptr;
}

void strata_text_free(strata_text* h) { delete h; }

int64_t strata_text_vocab_size(const strata_text* h) {
    return h == nullptr ? -1 : static_cast<int64_t>(h->tok.size());
}

int64_t strata_text_encode(const strata_text* h, const char* utf8, int64_t len, int parse_special,
                           const int64_t* spans, int64_t n_spans, int32_t* out, int64_t cap) {
    g_error.clear();
    try {
        if (h == nullptr || (utf8 == nullptr && len > 0) || len < 0 || n_spans < 0 || (n_spans > 0 && spans == nullptr)) {
            throw std::invalid_argument("bad argument to strata_text_encode");
        }
        std::vector<std::pair<std::size_t, std::size_t>> plain;
        plain.reserve(static_cast<std::size_t>(n_spans));
        for (int64_t i = 0; i < n_spans; ++i) {
            const int64_t a = spans[2 * i], b = spans[2 * i + 1];
            plain.emplace_back(a < 0 ? 0 : static_cast<std::size_t>(a), b < 0 ? 0 : static_cast<std::size_t>(b));
        }
        const std::vector<int> ids =
            h->tok.encode(std::string_view(utf8 == nullptr ? "" : utf8, static_cast<std::size_t>(len)),
                          parse_special != 0, plain);
        const int64_t n = static_cast<int64_t>(ids.size());
        if (n <= cap && out != nullptr) {
            for (int64_t i = 0; i < n; ++i) {
                out[i] = static_cast<int32_t>(ids[static_cast<std::size_t>(i)]);
            }
        }
        return n;
    } catch (const std::exception& e) {
        g_error = e.what();
    } catch (...) {
        g_error = "unknown error in strata_text_encode";
    }
    return -1;
}

int64_t strata_text_export_tokens(const strata_text* h, int raw, char* blob, int64_t blob_cap, int64_t* offsets) {
    g_error.clear();
    try {
        if (h == nullptr || offsets == nullptr) {
            throw std::invalid_argument("bad argument to strata_text_export_tokens");
        }
        std::string all;
        const std::size_t n = h->tok.size();
        offsets[0] = 0;
        for (std::size_t i = 0; i < n; ++i) {
            if (raw != 0) {
                try {
                    all += h->tok.token_bytes(static_cast<int>(i));
                } catch (const std::runtime_error&) {
                    // not byte-alphabet text: empty, see the header
                }
            } else {
                all += h->tok.tokens()[i];
            }
            offsets[i + 1] = static_cast<int64_t>(all.size());
        }
        if (blob != nullptr && static_cast<int64_t>(all.size()) <= blob_cap) {
            std::memcpy(blob, all.data(), all.size());
        }
        return static_cast<int64_t>(all.size());
    } catch (const std::exception& e) {
        g_error = e.what();
    } catch (...) {
        g_error = "unknown error in strata_text_export_tokens";
    }
    return -1;
}

int64_t strata_text_control_tokens(const strata_text* h, char* buf, int64_t cap) {
    g_error.clear();
    try {
        if (h == nullptr) {
            throw std::invalid_argument("bad argument to strata_text_control_tokens");
        }
        std::string all;
        for (std::size_t i = 0; i < h->types.size() && i < h->tok.size(); ++i) {
            if (h->types[i] == 3) {
                if (!all.empty()) {
                    all += '\n';
                }
                all += h->tok.tokens()[i];
            }
        }
        if (buf != nullptr && static_cast<int64_t>(all.size()) <= cap) {
            std::memcpy(buf, all.data(), all.size());
        }
        return static_cast<int64_t>(all.size());
    } catch (const std::exception& e) {
        g_error = e.what();
    } catch (...) {
        g_error = "unknown error in strata_text_control_tokens";
    }
    return -1;
}

}  // extern "C"
