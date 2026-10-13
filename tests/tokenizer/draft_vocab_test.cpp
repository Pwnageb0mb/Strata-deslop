// tests/tokenizer/draft_vocab_test.cpp - the draft-vocabulary builder, against the reference's own output.
//
// tools/gen_draft_vocab_golden.py writes a GGUF carrying a byte-level BPE vocabulary (with Han, Kana, Hangul,
// CJK-punctuation and Cyrillic tokens, a CONTROL literal, and a token whose bytes are a PREFIX of a multi-byte
// character), plus a base subset and a corpus, then runs tools/draft_vocab.py over them for four configurations
// and records its stdout and the subset it wrote.
//
// This runs the C++ tool over the SAME fixtures and compares both.  It drives the built binary rather than the
// library, because the printed report IS the acceptance: a second copy of the logic inside the test would be a
// second thing to get wrong.
//
//    draft_vocab_test <tests/tokenizer/draft_vocab_golden.txt> <path to strata-draft-vocab>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace {

int failures = 0;

std::string read_text(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return {};
    return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

void check(const std::string& what, const std::string& got, const std::string& want) {
    if (got == want) return;
    std::fprintf(stderr, "FAIL %s\n  got  %s\n  want %s\n", what.c_str(), got.c_str(), want.c_str());
    ++failures;
}

std::string shell_quote(const std::string& s) {
    std::string o = "'";
    for (char c : s) {
        if (c == '\'') o += "'\\''";
        else o += c;
    }
    return o + "'";
}

/// Runs `cmd` and returns its stdout.  `ok` is false when the tool exited non-zero.
std::string run(const std::string& cmd, bool& ok) {
    std::string full = cmd + " 2>/dev/null";
    std::FILE* p = ::popen(full.c_str(), "r");
    if (p == nullptr) {
        ok = false;
        return {};
    }
    std::string out;
    char buf[4096];
    std::size_t n = 0;
    while ((n = std::fread(buf, 1, sizeof buf, p)) > 0) out.append(buf, n);
    ok = ::pclose(p) == 0;
    return out;
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: draft_vocab_test <golden> <strata-draft-vocab>\n");
        return 2;
    }
    const std::string golden_path = argv[1];
    const std::string exe = argv[2];
    const std::string dir = std::filesystem::path(golden_path).parent_path().string();
    const std::string all = read_text(golden_path);
    if (all.empty()) {
        std::fprintf(stderr, "cannot read %s\n", golden_path.c_str());
        return 2;
    }

    std::string gguf, base, corpus;
    int64_t want_vocab = 0, want_base_ids = 0;
    std::vector<std::string> cfg_order;
    std::map<std::string, std::string> args_by_cfg, stdout_by_cfg, subset_by_cfg;

    std::istringstream ss(all);
    std::string line;
    while (std::getline(ss, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ls(line);
        std::string what, name, rest;
        ls >> what;
        if (what == "gguf") ls >> gguf;
        else if (what == "base") ls >> base;
        else if (what == "corpus") ls >> corpus;
        else if (what == "vocab") ls >> want_vocab;
        else if (what == "base_ids") ls >> want_base_ids;
        else if (what == "cfg") {
            ls >> name;
            std::getline(ls, rest);
            if (!rest.empty() && rest[0] == ' ') rest.erase(0, 1);
            const std::size_t at = rest.find("args ");
            args_by_cfg[name] = at == std::string::npos ? "" : rest.substr(at + 5);
            cfg_order.push_back(name);
        } else if (what == "stdout") {
            ls >> name;
            std::getline(ls, rest);
            if (!rest.empty() && rest[0] == ' ') rest.erase(0, 1);
            stdout_by_cfg[name] += rest + "\n";
        } else if (what == "subset") {
            ls >> name >> rest;   // bytes N sha256 H ids N
            std::getline(ls, rest);
            subset_by_cfg[name] = rest;
        } else {
            std::fprintf(stderr, "unknown golden line: %s\n", line.c_str());
            ++failures;
        }
    }
    if (gguf.empty() || cfg_order.empty()) {
        std::fprintf(stderr, "the golden is not the shape this test expects\n");
        return 1;
    }

    const std::string gguf_path = dir + "/" + gguf;
    const std::string base_path = dir + "/" + base;
    const std::string tmp_out = dir + "/.draft-out.bin";

    for (const std::string& name : cfg_order) {
        const std::string& args = args_by_cfg[name];
        const bool writes = subset_by_cfg.find(name) != subset_by_cfg.end();
        // the args are relative to the fixture directory; a bare filename is a corpus file there
        std::string fixed;
        {
            std::istringstream as(args);
            std::string tok;
            while (as >> tok) {
                std::string v = tok;
                if (tok == "corpus.txt") v = dir + "/" + tok;
                if (!fixed.empty()) fixed += " ";
                fixed += v;
            }
        }
        std::string cmd = shell_quote(exe) + " --gguf " + shell_quote(gguf_path) + " --base " +
                          shell_quote(base_path);
        if (!fixed.empty()) cmd += " " + fixed;
        if (writes) cmd += " --out " + shell_quote(tmp_out);
        bool ok = false;
        std::string got = run(cmd, ok);
        if (!ok) {
            std::fprintf(stderr, "FAIL %s: the tool exited non-zero\n  cmd: %s\n", name.c_str(), cmd.c_str());
            ++failures;
            continue;
        }
        // the tool prints the paths it was given; the golden records them as placeholders
        for (const auto& [placeholder, path] : {std::pair<std::string, std::string>{"<out>", tmp_out},
                                                {"<gguf>", gguf_path},
                                                {"<base>", base_path}}) {
            std::size_t at = 0;
            while ((at = got.find(path, at)) != std::string::npos) {
                got = got.substr(0, at) + placeholder + got.substr(at + path.size());
                at += placeholder.size();
            }
        }
        check("stdout " + name, got, stdout_by_cfg[name]);
        if (writes) {
            const std::string want = read_text(dir + "/draft-" + name + ".bin");
            const std::string have = read_text(tmp_out);
            if (have != want) {
                std::fprintf(stderr, "FAIL subset %s differs from the reference (%zu vs %zu bytes)\n", name.c_str(),
                             have.size(), want.size());
                ++failures;
            }
            std::remove(tmp_out.c_str());
        }
    }

    if (failures) {
        std::fprintf(stderr, "draft_vocab_test: FAILED (%d)\n", failures);
        return 1;
    }
    std::printf("draft_vocab_test: all %zu configurations match the reference's own output and subsets\n",
                cfg_order.size());
    return 0;
}
