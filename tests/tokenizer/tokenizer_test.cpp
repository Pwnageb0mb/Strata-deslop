// tests/tokenizer/tokenizer_test.cpp - the same ids as the reference, on the reference's own answers.
//
// The golden file (tools/gen_tokenizer_golden.py, tests/tokenizer/tokenizer_golden.txt) carries synthetic
// vocabularies and the ids tools/strata_tokenizer.py gives for a corpus over each of them.  This builds the same
// vocabularies in C++, encodes the same text and fails on the first id that differs - because a byte-level BPE
// that is wrong still produces plausible ids, and only a comparison sees it.
//
//    tokenizer_test <tests/tokenizer/tokenizer_golden.txt>
#include "strata/tokenizer/tokenizer.hpp"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <string_view>
#include <vector>

namespace {

std::string unhex(std::string_view hex) {
    static const char* digits = "0123456789abcdef";
    std::string out;
    out.reserve(hex.size() / 2);
    for (std::size_t i = 0; i + 1 < hex.size(); i += 2) {
        const auto hi = std::string(digits).find(hex[i]);
        const auto lo = std::string(digits).find(hex[i + 1]);
        if (hi == std::string::npos || lo == std::string::npos) {
            std::printf("bad hex in the golden file\n");
            std::exit(1);
        }
        out.push_back(static_cast<char>(hi * 16 + lo));
    }
    return out;
}

std::string escaped(std::string_view s) {
    std::string out;
    for (const unsigned char c : s) {
        if (c == '\n') {
            out += "\\n";
        } else if (c == '\r') {
            out += "\\r";
        } else if (c == '\t') {
            out += "\\t";
        } else if (c < 0x20 || c == 0x7F) {
            char buf[8];
            std::snprintf(buf, sizeof buf, "\\x%02X", c);
            out += buf;
        } else {
            out.push_back(static_cast<char>(c));
        }
    }
    return out;
}

std::vector<std::string> fields(const std::string& line) {
    std::vector<std::string> out;
    std::size_t start = 0;
    while (true) {
        const std::size_t tab = line.find('\t', start);
        if (tab == std::string::npos) {
            out.push_back(line.substr(start));
            return out;
        }
        out.push_back(line.substr(start, tab - start));
        start = tab + 1;
    }
}

struct Case {
    int vocab = 0;
    bool parse_special = false;
    std::string text;
    std::vector<int> ids;
    std::string decoded;
};

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::printf("usage: tokenizer_test <tests/tokenizer/tokenizer_golden.txt>\n");
        return 2;
    }
    std::ifstream in(argv[1]);
    if (!in) {
        std::printf("cannot read %s\n", argv[1]);
        return 2;
    }

    std::vector<std::string> lines;
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty() && line[0] != '#') {
            lines.push_back(line);
        }
    }

    std::vector<strata::tokenizer::Tokenizer> vocabs;
    std::vector<Case> cases;
    std::size_t i = 0;
    while (i < lines.size()) {
        const std::vector<std::string> f = fields(lines[i]);
        if (f.size() < 2) {
            std::printf("bad line %zu in the golden file\n", i + 1);
            return 2;
        }
        if (f[0] == "V") {
            const std::size_t ntok = std::stoul(f[2]), nmerge = std::stoul(f[3]);
            std::vector<std::string> tokens, merges;
            std::vector<int> types;
            for (std::size_t k = 0; k < ntok; ++k) {
                const std::vector<std::string> tf = fields(lines[++i]);
                tokens.push_back(unhex(tf[1]));
                types.push_back(tf.size() > 2 ? std::stoi(tf[2]) : 0);
            }
            for (std::size_t k = 0; k < nmerge; ++k) {
                merges.push_back(unhex(fields(lines[++i])[1]));
            }
            vocabs.emplace_back(tokens, merges, types);
            ++i;
            continue;
        }
        if (f[0] == "X") {
            Case c;
            c.vocab = std::stoi(f[1]);
            c.parse_special = f[2] == "1";
            c.text = unhex(f[3]);
            const std::string& list = f[4];
            std::size_t start = 0;
            while (!list.empty() && start <= list.size()) {
                const std::size_t comma = list.find(',', start);
                c.ids.push_back(std::stoi(list.substr(start, comma - start)));
                if (comma == std::string::npos) {
                    break;
                }
                start = comma + 1;
            }
            c.decoded = unhex(f[5]);
            cases.push_back(c);
            ++i;
            continue;
        }
        ++i;
    }

    std::size_t bad = 0;
    for (const Case& c : cases) {
        const strata::tokenizer::Tokenizer& tok = vocabs[static_cast<std::size_t>(c.vocab) - 1];
        std::vector<int> got;
        try {
            got = tok.encode(c.text, c.parse_special);
        } catch (const std::exception& e) {
            std::printf("vocab %d %s: encode threw: %s\n", c.vocab, escaped(c.text).c_str(), e.what());
            ++bad;
            continue;
        }
        if (got != c.ids) {
            std::printf("vocab %d %s (parse_special=%d): %zu ids, reference has %zu\n", c.vocab,
                        escaped(c.text).c_str(), c.parse_special ? 1 : 0, got.size(), c.ids.size());
            const std::size_t n = got.size() < c.ids.size() ? got.size() : c.ids.size();
            for (std::size_t k = 0; k < n; ++k) {
                if (got[k] != c.ids[k]) {
                    std::printf("    first difference at %zu: got %d, reference %d\n", k, got[k], c.ids[k]);
                    break;
                }
            }
            ++bad;
            continue;
        }
        const std::string back = tok.decode(got);
        if (back != c.decoded) {
            std::printf("vocab %d %s: decode gave %s, reference %s\n", c.vocab, escaped(c.text).c_str(),
                        escaped(back).c_str(), escaped(c.decoded).c_str());
            ++bad;
        }
    }

    std::printf("tokenizer: %zu vocabularies, %zu cases, %zu disagreements with the reference\n", vocabs.size(),
                cases.size(), bad);
    return bad == 0 ? 0 : 1;
}
