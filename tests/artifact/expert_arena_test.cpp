// tests/artifact/expert_arena_test.cpp - the expert arena's layout against the reference's own experts.bin.
//
// tools/gen_expert_arena_golden.py writes a synthetic GGUF whose three role tensors are Q2_0 and runs
// tools/pack_layer.py's own build() over it with the geometry scaled to a fixture-sized model.  This builds the
// same blobs with pack_expert_blob and compares every byte, then inverts them with unpack_expert_role and
// compares against the reference's rebuild_role.
//
// Both halves matter, and for different reasons: the forward comparison is against bytes the reference PRODUCED,
// and the inverse is against a statement the reference wrote separately from its layout ("if it and pack_layer
// were the same code the arena would be unverified by construction").
//
//    expert_arena_test <tests/artifact/expert_arena_golden.txt>
#include "strata/artifact/expert_arena.hpp"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace {

int failures = 0;

std::vector<uint8_t> unhex(const std::string& h) {
    static const char* digits = "0123456789abcdef";
    std::vector<uint8_t> out;
    if (h.size() % 2) {
        std::fprintf(stderr, "bad hex (odd length)\n");
        ++failures;
        return out;
    }
    for (std::size_t i = 0; i + 1 < h.size(); i += 2) {
        const std::size_t hi = std::string(digits).find(h[i]);
        const std::size_t lo = std::string(digits).find(h[i + 1]);
        if (hi == std::string::npos || lo == std::string::npos) {
            std::fprintf(stderr, "bad hex\n");
            ++failures;
            return out;
        }
        out.push_back((uint8_t)((hi << 4) | lo));
    }
    return out;
}

std::string hex_of(const uint8_t* p, std::size_t n) {
    static const char* digits = "0123456789abcdef";
    std::string s;
    s.reserve(n * 2);
    for (std::size_t i = 0; i < n; ++i) {
        s += digits[p[i] >> 4];
        s += digits[p[i] & 0xF];
    }
    return s;
}

void check(const std::string& what, const std::string& got, const std::string& want) {
    if (got != want) {
        std::fprintf(stderr, "FAIL %s (got %zu chars, want %zu)\n", what.c_str(), got.size(), want.size());
        for (std::size_t i = 0; i < got.size() && i < want.size(); ++i)
            if (got[i] != want[i]) {
                std::fprintf(stderr, "  first difference at byte %zu\n", i / 2);
                break;
            }
        ++failures;
    }
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: expert_arena_test <tests/artifact/expert_arena_golden.txt>\n");
        return 2;
    }
    std::ifstream in(argv[1]);
    if (!in) {
        std::fprintf(stderr, "cannot read %s\n", argv[1]);
        return 2;
    }

    strata::ExpertGeometry g;
    std::map<std::string, std::string> offsets, roles, inverse;
    std::string blob_hex;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        std::string what;
        ss >> what;
        if (what == "geom") {
            std::string kv;
            while (ss >> kv) {
                const std::size_t eq = kv.find('=');
                if (eq == std::string::npos) continue;
                const std::string k = kv.substr(0, eq);
                const uint64_t v = std::strtoull(kv.c_str() + eq + 1, nullptr, 10);
                if (k == "H") g.H = v;
                else if (k == "FF") g.FF = v;
                else if (k == "NE") g.NE = v;
                else if (k == "QK") g.QK = v;
                else if (k == "block") g.block_bytes = v;
            }
        } else if (what == "offsets") {
            std::string kv;
            while (ss >> kv) {
                const std::size_t eq = kv.find('=');
                if (eq != std::string::npos) offsets[kv.substr(0, eq)] = kv.substr(eq + 1);
            }
        } else if (what == "role" || what == "inverse") {
            std::string role, hex;
            ss >> role >> hex;
            (what == "role" ? roles : inverse)[role] = hex;
        } else if (what == "blob") {
            ss >> blob_hex;
        } else {
            std::fprintf(stderr, "unknown golden line: %s\n", line.c_str());
            ++failures;
        }
    }

    // the geometry's derived sizes and plane offsets, against the reference's own
    const strata::ExpertArenaOffsets off = g.offsets();
    auto want_off = [&off](const std::string& k) -> uint64_t {
        if (k == "gu_codes") return off.gate_up_codes;
        if (k == "down_codes") return off.down_codes;
        if (k == "gu_scales") return off.gate_up_scales;
        return off.down_scales;
    };
    for (const auto& kv : offsets)
        check("offset " + kv.first, std::to_string(want_off(kv.first)), kv.second);

    const std::vector<uint8_t> gate = unhex(roles["gate"]);
    const std::vector<uint8_t> up = unhex(roles["up"]);
    const std::vector<uint8_t> down = unhex(roles["down"]);
    const std::vector<uint8_t> want_blob = unhex(blob_hex);
    const uint64_t rb = g.role_bytes(), bb = g.blob_bytes();
    if (gate.size() != g.NE * rb || up.size() != g.NE * rb || down.size() != g.NE * rb ||
        want_blob.size() != g.NE * bb) {
        std::fprintf(stderr, "FAIL the golden's sizes do not match the geometry\n");
        return 1;
    }

    // ---- the forward layout: every expert's blob, against the bytes the reference produced
    std::vector<uint8_t> blob((std::size_t)(g.NE * bb), 0);
    for (uint64_t e = 0; e < g.NE; ++e)
        strata::pack_expert_blob(gate.data() + e * rb, up.data() + e * rb, down.data() + e * rb, g,
                                 blob.data() + e * bb);
    check("experts.bin", hex_of(blob.data(), blob.size()), blob_hex);

    // ---- the inverse, against the reference's own rebuild_role
    for (int r = 0; r < 3; ++r) {
        const char* name = r == 0 ? "gate" : (r == 1 ? "up" : "down");
        std::vector<uint8_t> out((std::size_t)rb, 0);
        strata::unpack_expert_role(blob.data(), g, (strata::ExpertRole)r, out.data());
        check(std::string("inverse ") + name, hex_of(out.data(), out.size()), inverse[name]);
        // ...and the inverse of the FIRST expert's blob is that expert's source run, which is the property the
        // whole relayout rests on: nothing was computed, only moved.
        const std::vector<uint8_t>& src = r == 0 ? gate : (r == 1 ? up : down);
        check(std::string("round trip ") + name, hex_of(out.data(), out.size()),
              hex_of(src.data(), (std::size_t)rb));
    }

    // ---- and the layout is not the identity: if the blob equalled the source runs concatenated, every check
    // above would pass for a "layout" that did nothing.
    {
        std::vector<uint8_t> flat;
        flat.insert(flat.end(), gate.begin(), gate.end());
        flat.insert(flat.end(), up.begin(), up.end());
        flat.insert(flat.end(), down.begin(), down.end());
        if (blob.size() == flat.size() && blob == flat) {
            std::fprintf(stderr, "FAIL the arena layout did not move any bytes\n");
            ++failures;
        }
    }

    // ---- the REAL model's geometry, which the fixture does not exercise: if these constants drift, the arena
    // the engine reads is the wrong size, and the fixture-sized run would not notice.
    {
        const strata::ExpertGeometry real;
        const strata::ExpertArenaOffsets ro = real.offsets();
        if (real.blob_bytes() != 1382400 || real.role_bytes() != 460800 || ro.gate_up_codes != 0 ||
            ro.down_codes != 819200 || ro.gate_up_scales != 1228800 || ro.down_scales != 1331200) {
            std::fprintf(stderr, "FAIL the real model's geometry is wrong: blob %llu, offsets %llu/%llu/%llu/%llu\n",
                         (unsigned long long)real.blob_bytes(), (unsigned long long)ro.gate_up_codes,
                         (unsigned long long)ro.down_codes, (unsigned long long)ro.gate_up_scales,
                         (unsigned long long)ro.down_scales);
            ++failures;
        }
    }

    if (failures) {
        std::fprintf(stderr, "expert_arena_test: FAILED (%d)\n", failures);
        return 1;
    }
    std::printf("expert_arena_test: %llu experts' blobs and the inverse match the reference (H=%llu FF=%llu NE=%llu, "
                "%llu B a blob), and the real model's 1,382,400 B geometry is intact\n",
                (unsigned long long)g.NE, (unsigned long long)g.H, (unsigned long long)g.FF,
                (unsigned long long)g.NE, (unsigned long long)bb);
    return 0;
}
