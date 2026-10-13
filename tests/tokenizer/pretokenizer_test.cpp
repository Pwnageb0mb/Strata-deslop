// tests/tokenizer/pretokenizer_test.cpp - the same pieces as the reference, on the reference's own output.
//
// The golden file (tools/gen_pretok_golden.py, tests/tokenizer/pretok_golden.txt) is what Python's `regex`
// module produces for `QWEN35_PATTERN` over a corpus built to hit every branch.  This runs the C++ scanner over
// the same text and fails on the first piece that differs - the check the reference's own docstring asks for:
// a wrong pre-tokenizer still produces plausible pieces, so only a comparison sees it.
//
//    pretokenizer_test <tests/tokenizer/pretok_golden.txt>
#include "strata/tokenizer/pretokenizer.hpp"

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

std::vector<std::string> split_tabs(const std::string& line) {
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

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::printf("usage: pretokenizer_test <tests/tokenizer/pretok_golden.txt>\n");
        return 2;
    }
    std::ifstream in(argv[1]);
    if (!in) {
        std::printf("cannot read %s\n", argv[1]);
        return 2;
    }

    std::size_t cases = 0, pieces = 0, bad = 0;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        const std::vector<std::string> fields = split_tabs(line);
        const std::size_t want = static_cast<std::size_t>(std::stoul(fields[0]));
        const std::string text = unhex(fields[1]);
        const std::vector<std::string_view> got = strata::tokenizer::pretokenize(text);

        // the pieces must cover the text exactly: no gap, no overlap, in order
        std::string joined;
        for (const std::string_view p : got) {
            joined.append(p.data(), p.size());
        }
        if (joined != text) {
            std::printf("case %zu: the pieces do not rebuild the text: %s\n", cases, escaped(text).c_str());
            ++bad;
        }

        if (got.size() != want || fields.size() != want + 2) {
            std::printf("case %zu: %s -> %zu pieces, reference has %zu\n", cases, escaped(text).c_str(),
                        got.size(), want);
            for (std::size_t k = 0; k < got.size(); ++k) {
                std::printf("    got[%zu]      %s\n", k, escaped(got[k]).c_str());
            }
            for (std::size_t k = 0; k < want; ++k) {
                std::printf("    reference[%zu] %s\n", k, escaped(unhex(fields[k + 2])).c_str());
            }
            ++bad;
        } else {
            for (std::size_t k = 0; k < want; ++k) {
                const std::string expected = unhex(fields[k + 2]);
                if (std::string_view(expected) != got[k]) {
                    std::printf("case %zu: %s piece %zu: got %s, reference %s\n", cases, escaped(text).c_str(),
                                k, escaped(got[k]).c_str(), escaped(expected).c_str());
                    ++bad;
                    break;
                }
            }
        }
        ++cases;
        pieces += got.size();
    }

    std::printf("pretokenizer: %zu cases, %zu pieces, %zu disagreements with the reference\n", cases, pieces, bad);
    return bad == 0 ? 0 : 1;
}
