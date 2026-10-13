// tests/artifact/mtp_pack_test.cpp - the MTP draft head's GGUF, against tools/mtp_pack.py's own, for all three arms.
//
// tools/gen_mtp_pack_golden.py writes a committable checkpoint and runs the reference over it in q2_0, q4_0 and
// q8_0, recording what each produced: the stdout, the output's size and digest, its first bytes, and the sidecar
// report.  This runs the built tool over the same checkpoint and compares all of it.
//
// The report's per-tensor `seconds` is wall-clock and cannot be compared; both sides normalize it and nothing
// else (see normalize_seconds).
//
//    mtp_pack_test <golden> <strata-mtp-pack>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "strata/artifact/sha256.hpp"

namespace {

int failures = 0;

std::string read_text(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return {};
    return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

std::string hex_of(const unsigned char* p, std::size_t n) {
    static const char* digits = "0123456789abcdef";
    std::string s;
    for (std::size_t i = 0; i < n; ++i) {
        s += digits[p[i] >> 4];
        s += digits[p[i] & 0xF];
    }
    return s;
}

std::string shell_quote(const std::string& s) {
    std::string o = "'";
    for (char c : s) {
        if (c == '\'') o += "'\\''";
        else o += c;
    }
    return o + "'";
}

void check(const std::string& what, const std::string& got, const std::string& want) {
    if (got == want) return;
    std::fprintf(stderr, "FAIL %s\n  got  %s\n  want %s\n", what.c_str(), got.c_str(), want.c_str());
    ++failures;
}

/// The report's `seconds` is wall-clock; replace each value with a constant so the rest can be compared.
std::string normalize_seconds(const std::string& s) {
    static const std::string key = "\"seconds\": ";
    std::string out;
    std::size_t at = 0;
    for (;;) {
        const std::size_t p = s.find(key, at);
        if (p == std::string::npos) {
            out += s.substr(at);
            return out;
        }
        out += s.substr(at, p + key.size() - at);
        std::size_t e = p + key.size();
        while (e < s.size() && s[e] != ',' && s[e] != '\n') ++e;
        out += "0.0";
        at = e;
    }
}

struct Arm {
    std::string name, file_line, stdout_text, head_hex, report;
};

/// Run the tool for one arm and compare everything the golden recorded.
void run_arm(const Arm& a, const std::string& exe, const std::string& src, const std::string& dir) {
    const std::string out = dir + "/.mtp-pack-" + a.name + ".gguf";
    const std::string rep = out + ".report.json";
    std::error_code ec;
    std::filesystem::remove(out, ec);
    std::filesystem::remove(rep, ec);

    const std::string cmd = shell_quote(exe) + " --src " + shell_quote(src) + " --experts " + a.name +
                            " --out " + shell_quote(out) + " 2>&1";
    std::FILE* pipe = ::popen(cmd.c_str(), "r");
    if (pipe == nullptr) {
        std::fprintf(stderr, "FAIL cannot run the tool for %s\n", a.name.c_str());
        ++failures;
        return;
    }
    std::string got;
    char buf[4096];
    std::size_t n = 0;
    while ((n = std::fread(buf, 1, sizeof buf, pipe)) > 0) got.append(buf, n);
    if (::pclose(pipe) != 0) {
        std::fprintf(stderr, "FAIL the tool exited non-zero for %s:\n%s\n", a.name.c_str(), got.c_str());
        ++failures;
        return;
    }
    {
        std::size_t pos = 0;
        while ((pos = got.find(out, pos)) != std::string::npos) {
            got = got.substr(0, pos) + "<out>" + got.substr(pos + out.size());
            pos += 5;
        }
    }
    check(a.name + " stdout", got, a.stdout_text);

    const std::string blob = read_text(out);
    if (blob.empty()) {
        std::fprintf(stderr, "FAIL the tool wrote nothing for %s\n", a.name.c_str());
        ++failures;
        return;
    }
    strata::Sha256 h;
    h.update(blob.data(), blob.size());
    const std::array<unsigned char, 32> d = h.finish();
    {
        const std::size_t b = a.file_line.find("bytes ");
        const std::size_t q = a.file_line.find("sha256 ");
        check(a.name + " size", std::to_string(blob.size()), a.file_line.substr(b + 6, q - 1 - (b + 6)));
        check(a.name + " sha256", hex_of(d.data(), d.size()), a.file_line.substr(q + 7));
    }
    if (!a.head_hex.empty()) {
        const std::size_t m = std::min<std::size_t>(80, blob.size());
        check(a.name + " the first bytes", hex_of((const unsigned char*) blob.data(), m), a.head_hex.substr(0, m * 2));
    }
    check(a.name + " report", normalize_seconds(read_text(rep)), a.report);

    std::filesystem::remove(out, ec);
    std::filesystem::remove(rep, ec);
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: mtp_pack_test <golden> <strata-mtp-pack>\n");
        return 2;
    }
    const std::string golden_path = argv[1], exe = argv[2];
    const std::string dir = std::filesystem::path(golden_path).parent_path().string();
    const std::string all = read_text(golden_path);
    if (all.empty()) {
        std::fprintf(stderr, "cannot read %s\n", golden_path.c_str());
        return 2;
    }

    std::string fixture;
    std::vector<Arm> arms;
    {
        std::size_t p = 0;
        bool in_report = false;
        while (p < all.size()) {
            const std::size_t nl = all.find('\n', p);
            const std::string line = all.substr(p, nl == std::string::npos ? std::string::npos : nl - p);
            p = nl == std::string::npos ? all.size() : nl + 1;
            if (in_report) {
                if (line == "report_end") {
                    in_report = false;
                    if (!arms.back().report.empty()) arms.back().report.pop_back();
                } else {
                    arms.back().report += line + "\n";
                }
                continue;
            }
            if (line.rfind("fixture ", 0) == 0) fixture = line.substr(8);
            else if (line.rfind("## ", 0) == 0) arms.push_back(Arm{line.substr(3)});
            else if (arms.empty()) continue;
            else if (line.rfind("file ", 0) == 0) arms.back().file_line = line.substr(5);
            else if (line.rfind("stdout ", 0) == 0) arms.back().stdout_text += line.substr(7) + "\n";
            else if (line.rfind("head_hex ", 0) == 0) arms.back().head_hex = line.substr(9);
            else if (line == "report_begin") in_report = true;
        }
    }
    if (fixture.empty() || arms.empty()) {
        std::fprintf(stderr, "the golden carries no fixture or no arms\n");
        return 1;
    }

    const std::string src = dir + "/" + fixture;
    for (const Arm& a : arms) run_arm(a, exe, src, dir);

    if (failures) {
        std::fprintf(stderr, "mtp_pack_test: FAILED (%d)\n", failures);
        return 1;
    }
    std::printf("mtp_pack_test: all %zu arms match tools/mtp_pack.py's own GGUF and report\n", arms.size());
    return 0;
}
