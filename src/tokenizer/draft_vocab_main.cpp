// src/tokenizer/draft_vocab_main.cpp - `strata-draft-vocab`: the C++ port of tools/draft_vocab.py.
//
// The draft head's token subset: the vocabulary rows the MTP draft layer may propose.  A token outside the subset
// can never be drafted, so a subset built from English and code alone drafts almost nothing in Chinese (#137).
// This adds whole scripts (`--add`), and the most frequent tokens of a text corpus that cover `--coverage` of its
// occurrences (`--corpus`, #597).
//
//    strata-draft-vocab --gguf MODEL.gguf [--base SUBSET.bin] [--add cjk] [--corpus F.txt ...]
//                       [--coverage 0.99] [--out SUBSET.bin] [--stats]
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "strata/tokenizer/gguf_vocab.hpp"
#include "strata/tokenizer/utf8.hpp"

namespace {

// The scripts a subset can be extended with.  A character is in a script when its codepoint is in one of the
// ranges; the ranges are the reference's, and they are what decides which vocabulary rows a name selects.
struct Script {
    const char* name;
    std::vector<std::pair<std::uint32_t, std::uint32_t>> ranges;
};

const std::vector<Script>& scripts() {
    static const std::vector<Script> s = {
        {"han", {{0x3400, 0x4DBF}, {0x4E00, 0x9FFF}, {0xF900, 0xFAFF}, {0x20000, 0x2FA1F}}},
        {"kana", {{0x3040, 0x30FF}, {0x31F0, 0x31FF}, {0xFF66, 0xFF9F}}},
        {"hangul", {{0x1100, 0x11FF}, {0x3130, 0x318F}, {0xAC00, 0xD7AF}}},
        {"cjk_punct", {{0x3000, 0x303F}, {0xFF00, 0xFF65}, {0xFFA0, 0xFFEF}}},
        // Ukrainian, Russian, Bulgarian, Serbian...: the shipped subset held 142 of 18,580 Cyrillic tokens
        {"cyrillic", {{0x0400, 0x04FF}, {0x0500, 0x052F}, {0x1C80, 0x1C8F}, {0x2DE0, 0x2DFF}, {0xA640, 0xA69F}}},
    };
    return s;
}

const std::map<std::string, std::vector<std::string>>& groups() {
    static const std::map<std::string, std::vector<std::string>> g = {
        {"cjk", {"han", "kana", "hangul", "cjk_punct"}},
    };
    return g;
}

/// The scripts a text is written in: the codepoints of its characters, tested against every range.
std::set<std::string> scripts_of(const std::string& text) {
    std::set<std::string> found;
    for (std::uint32_t cp : strata::tokenizer::utf8::decode(text).cp) {
        for (const Script& s : scripts())
            for (const auto& [a, b] : s.ranges)
                if (cp >= a && cp <= b) found.insert(s.name);
    }
    return found;
}

int usage() {
    std::fprintf(stderr,
                 "usage: strata-draft-vocab --gguf MODEL.gguf [--base SUBSET.bin] [--add LIST]\n"
                 "                          [--corpus F.txt ...] [--coverage F] [--out SUBSET.bin] [--stats]\n"
                 "\n"
                 "  --gguf P      the model's (first) GGUF file: its vocabulary\n"
                 "  --base P      the subset to extend (int32 ids)\n"
                 "  --add L       comma list: han, kana, hangul, cjk_punct, cjk, cyrillic\n"
                 "  --corpus F..  UTF-8 text files in the language to add (#597)\n"
                 "  --coverage F  with --corpus: the share of its token occurrences the added tokens cover\n"
                 "                (default 0.99)\n"
                 "  --out P       where to write the new subset\n"
                 "  --stats       print what the base (and the new) subset holds\n");
    return 2;
}

std::vector<std::string> split(const std::string& s, char sep) {
    std::vector<std::string> out;
    std::size_t at = 0;
    while (true) {
        const std::size_t next = s.find(sep, at);
        out.push_back(s.substr(at, next == std::string::npos ? std::string::npos : next - at));
        if (next == std::string::npos) return out;
        at = next + 1;
    }
}

std::string trim(const std::string& s) {
    const std::size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return {};
    const std::size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

std::string read_file(const std::string& path, bool& ok) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) {
        ok = false;
        return {};
    }
    std::fseek(f, 0, SEEK_END);
    const long end = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    std::string s;
    if (end > 0) {
        s.resize((std::size_t) end);
        if (std::fread(&s[0], 1, s.size(), f) != s.size()) s.clear();
    }
    std::fclose(f);
    ok = !s.empty() || end == 0;
    return s;
}

/// The most frequent ids (a tie: the lower id first) that together cover `coverage` of the occurrences.
std::vector<int> covering(const std::map<int, long long>& counts, double coverage) {
    long long total = 0;
    for (const auto& [_, c] : counts) total += c;
    std::vector<std::pair<int, long long>> by_freq(counts.begin(), counts.end());
    // the reference sorts by (-count, id): descending count, and the lower id first on a tie
    std::stable_sort(by_freq.begin(), by_freq.end(),
                     [](const auto& a, const auto& b) { return a.second > b.second; });
    std::vector<int> picked;
    long long covered = 0;
    for (const auto& [id, c] : by_freq) {
        if ((double) covered >= coverage * (double) total) break;
        picked.push_back(id);
        covered += c;
    }
    return picked;
}

/// The share of the occurrences whose token is not in `ids`.
double outside(const std::map<int, long long>& counts, const std::set<int>& ids) {
    long long total = 0, missing = 0;
    for (const auto& [id, c] : counts) {
        total += c;
        if (ids.find(id) == ids.end()) missing += c;
    }
    return total == 0 ? 0.0 : (double) missing / (double) total;
}

/// Python's `f"{x:g}"`, which is what the reference prints `--coverage` with.
std::string py_g(double v) {
    char buf[64];
    std::snprintf(buf, sizeof buf, "%g", v);
    return buf;
}

} // namespace

int main(int argc, char** argv) {
    std::string gguf, base_path, add, out;
    std::vector<std::string> corpus;
    double coverage = 0.99;
    bool stats = false;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char* what) -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "strata-draft-vocab: %s needs a value\n", what);
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "-h" || a == "--help") return usage();
        else if (a == "--gguf") gguf = next("--gguf");
        else if (a == "--base") base_path = next("--base");
        else if (a == "--add") add = next("--add");
        else if (a == "--corpus") { while (i + 1 < argc && argv[i + 1][0] != '-') corpus.push_back(argv[++i]); }
        else if (a == "--coverage") coverage = std::atof(next("--coverage"));
        else if (a == "--out") out = next("--out");
        else if (a == "--stats") stats = true;
        else {
            std::fprintf(stderr, "strata-draft-vocab: unknown option %s\n", a.c_str());
            return usage();
        }
    }
    if (gguf.empty()) return usage();

    strata::tokenizer::Tokenizer tok = [&] {
        try {
            return strata::tokenizer::tokenizer_from_gguf(gguf);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "strata-draft-vocab: %s\n", e.what());
            std::exit(1);
        }
    }();
    const int n = (int) tok.size();

    // each token's script set: a token whose bytes are not whole UTF-8 characters is not text, so it belongs to no
    // script (the reference's token_text returns None for it, and it must not be read as one)
    std::vector<std::set<std::string>> kinds((std::size_t) n);
    for (int i = 0; i < n; ++i) {
        const std::string bytes = tok.token_bytes(i);
        if (strata::tokenizer::utf8::is_whole_utf8(bytes)) kinds[(std::size_t) i] = scripts_of(bytes);
    }

    std::vector<int> base;
    if (!base_path.empty()) {
        bool ok = false;
        const std::string blob = read_file(base_path, ok);
        if (!ok || blob.size() % 4 != 0) {
            std::fprintf(stderr, "strata-draft-vocab: cannot read %s as int32 ids\n", base_path.c_str());
            return 1;
        }
        for (std::size_t i = 0; i + 3 < blob.size(); i += 4) {
            int32_t v = 0;
            std::memcpy(&v, blob.data() + i, 4);
            if (v < 0 || v >= n) {
                std::fprintf(stderr, "%s has ids outside the vocabulary (%d)\n", base_path.c_str(), n);
                return 1;
            }
            base.push_back(v);
        }
    }

    std::set<std::string> want;
    for (const std::string& x : split(add, ',')) {
        const std::string name = trim(x);
        if (name.empty()) continue;
        const auto g = groups().find(name);
        const std::vector<std::string> names = g == groups().end() ? std::vector<std::string>{name} : g->second;
        for (const std::string& nm : names) {
            bool known = false;
            for (const Script& s : scripts()) known = known || nm == s.name;
            if (!known) {
                std::string have;
                for (const Script& s : scripts()) have += std::string(have.empty() ? "" : ", ") + s.name;
                for (const auto& [k, _] : groups()) have += ", " + k;
                std::fprintf(stderr, "--add %s: expected one of %s\n", name.c_str(), have.c_str());
                return 1;
            }
            want.insert(nm);
        }
    }
    if (!(coverage > 0.0 && coverage <= 1.0)) {
        std::fprintf(stderr, "--coverage: expected a share in (0, 1]\n");
        return 1;
    }

    std::map<int, long long> counts;
    for (const std::string& path : corpus) {
        bool ok = false;
        const std::string text = read_file(path, ok);
        if (!ok) {
            std::fprintf(stderr, "strata-draft-vocab: cannot read %s\n", path.c_str());
            return 1;
        }
        for (int id : tok.encode(text)) counts[id] += 1;
    }
    std::set<int> from_corpus;
    if (!counts.empty())
        for (int id : covering(counts, coverage)) from_corpus.insert(id);

    const std::set<int> have(base.begin(), base.end());
    std::vector<int> ids = base;
    for (int i = 0; i < n; ++i)
        if (have.find(i) == have.end() && ((!kinds[(std::size_t) i].empty() &&
                                            std::any_of(want.begin(), want.end(), [&](const std::string& w) {
                                                return kinds[(std::size_t) i].count(w) != 0;
                                            })) ||
                                           from_corpus.count(i) != 0))
            ids.push_back(i);

    // the whole vocabulary's per-script counts, once: the reference recomputes them per call, which is the same
    // numbers
    std::map<std::string, int> full;
    for (const Script& s : scripts()) {
        int c = 0;
        for (const auto& k : kinds)
            if (k.count(s.name) != 0) ++c;
        full[s.name] = c;
    }
    auto print_stats = [&](const char* label, const std::vector<int>& sel) {
        std::string line = std::string(label) + ": " + std::to_string(sel.size()) + " ids; ";
        bool first = true;
        for (const Script& s : scripts()) {
            int c = 0;
            for (int i : sel)
                if (kinds[(std::size_t) i].count(s.name) != 0) ++c;
            if (!first) line += ", ";
            first = false;
            line += std::string(s.name) + " " + std::to_string(c) + "/" + std::to_string(full[s.name]);
        }
        std::printf("%s\n", line.c_str());
    };

    if (stats || out.empty()) {
        if (!base.empty()) print_stats("base", base);
        if (ids.size() > base.size()) print_stats("new", ids);
    }
    if (!counts.empty()) {
        long long total = 0;
        for (const auto& [_, c] : counts) total += c;
        std::size_t not_in_base = 0;
        for (int i : from_corpus)
            if (have.find(i) == have.end()) ++not_in_base;
        std::printf("corpus: %lld token occurrences, %zu distinct; %zu tokens cover %s, %zu of them not in the base\n",
                    total, counts.size(), from_corpus.size(), py_g(coverage).c_str(), not_in_base);
        std::printf("corpus occurrences outside the base: %.2f%%, outside the new subset: %.2f%%\n",
                    100.0 * outside(counts, have), 100.0 * outside(counts, std::set<int>(ids.begin(), ids.end())));
    }
    if (!out.empty()) {
        std::string blob;
        blob.resize(ids.size() * 4);
        for (std::size_t i = 0; i < ids.size(); ++i) {
            const int32_t v = (int32_t) ids[i];
            std::memcpy(&blob[i * 4], &v, 4);
        }
        std::FILE* f = std::fopen(out.c_str(), "wb");
        if (f == nullptr || (!blob.empty() && std::fwrite(blob.data(), 1, blob.size(), f) != blob.size())) {
            if (f != nullptr) std::fclose(f);
            std::fprintf(stderr, "strata-draft-vocab: cannot write %s\n", out.c_str());
            return 1;
        }
        std::fclose(f);
        std::printf("wrote %s: %zu ids (%zu added)\n", out.c_str(), ids.size(), ids.size() - base.size());
    }
    return 0;
}
