// tests/artifact/pack_manifest_test.cpp - the pack's manifest.json against the reference builder's own.
//
// tools/gen_pack_build_golden.py runs tools/strata_pack.py's build() over a synthetic model (geometry scaled to a
// fixture) and records its manifest.json.  This rebuilds the same entries - accumulating each segment's base the
// way the build does, so the plane offsets land where the reference put them - assembles the manifest and
// compares the whole file.
//
// The only machine-specific part is the two source paths, which are blanked on both sides.  Everything else has to
// match: the key order, the indentation, the entry fields, the codebook table, the expert table and the file sizes.
//
//    pack_manifest_test <tests/artifact/pack_build_golden.txt>
#include "strata/artifact/pack_manifest.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
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

/// Blank every path-valued field on both sides: the source paths are the one machine-specific part of a manifest.
std::string without_paths(std::string s) {
    for (const char* key : {"\"shard1\": \"", "\"shard2\": \"", "\"path\": \""}) {
        std::size_t p = 0;
        while ((p = s.find(key, p)) != std::string::npos) {
            const std::size_t a = p + std::strlen(key);
            const std::size_t b = s.find('"', a);
            if (b == std::string::npos) break;
            s = s.substr(0, a) + "<p>" + s.substr(b);
            p = a + 3;
        }
    }
    return s;
}

std::string first_diff(const std::string& got, const std::string& want) {
    std::size_t gl = 0, wl = 0;
    int n = 1;
    while (gl < got.size() || wl < want.size()) {
        const std::size_t ge = got.find('\n', gl);
        const std::size_t we = want.find('\n', wl);
        const std::string g = got.substr(gl, (ge == std::string::npos ? got.size() : ge) - gl);
        const std::string w = want.substr(wl, (we == std::string::npos ? want.size() : we) - wl);
        if (g != w) return "line " + std::to_string(n) + ":\n  got  " + g + "\n  want " + w;
        if (ge == std::string::npos && we == std::string::npos) break;
        gl = ge == std::string::npos ? got.size() : ge + 1;
        wl = we == std::string::npos ? want.size() : we + 1;
        ++n;
    }
    return {};
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: pack_manifest_test <tests/artifact/pack_build_golden.txt>\n");
        return 2;
    }
    std::ifstream in(argv[1], std::ios::binary);
    if (!in) {
        std::fprintf(stderr, "cannot read %s\n", argv[1]);
        return 2;
    }
    std::string all((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    const std::string marker = "# manifest-begin\n";
    const std::size_t at = all.find(marker);
    if (at == std::string::npos) {
        std::fprintf(stderr, "the golden carries no manifest\n");
        return 1;
    }
    const std::string want = all.substr(at + marker.size());

    strata::PackManifest man;
    std::map<std::string, uint64_t> seg_base;   // one per file: the build accumulates within a segment
    std::istringstream ss(all.substr(0, at));
    std::string line;
    while (std::getline(ss, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ls(line);
        std::string what;
        ls >> what;
        if (what == "geom") {
            std::string kv;
            while (ls >> kv) {
                const std::size_t eq = kv.find('=');
                if (eq == std::string::npos) continue;
                const std::string k = kv.substr(0, eq);
                const uint64_t v = std::strtoull(kv.c_str() + eq + 1, nullptr, 10);
                if (k == "H") man.geometry.H = v;
                else if (k == "FF") man.geometry.FF = v;
                else if (k == "NE") man.geometry.NE = v;
                else if (k == "QK") man.geometry.QK = v;
                else if (k == "block") man.geometry.block_bytes = v;
            }
            man.n_experts_per_layer = man.geometry.NE;
        } else if (what == "align") {
            ls >> man.align;
        } else if (what == "sha1") {
            ls >> man.shard1_sha256;
        } else if (what == "offsets") {
            // the geometry's plane offsets, against the reference's own
            const strata::ExpertArenaOffsets off = man.geometry.offsets();
            std::string kv;
            while (ls >> kv) {
                const std::size_t eq = kv.find('=');
                if (eq == std::string::npos) continue;
                const std::string k = kv.substr(0, eq);
                const uint64_t v = std::strtoull(kv.c_str() + eq + 1, nullptr, 10);
                const uint64_t want = k == "gu_codes"   ? off.gate_up_codes
                                      : k == "down_codes" ? off.down_codes
                                      : k == "gu_scales"  ? off.gate_up_scales
                                                          : off.down_scales;
                if (want != v) {
                    std::fprintf(stderr, "FAIL offset %s is %llu, the reference says %llu\n", k.c_str(),
                                 (unsigned long long)want, (unsigned long long)v);
                    ++failures;
                }
            }
        } else if (what == "n_layers") {
            ls >> man.n_layers;
        } else if (what == "tensor") {
            std::string name, type, file, raw_hex;
            uint64_t ne0 = 0, ne1 = 0, source_offset = 0;
            if (!(ls >> name >> type >> file >> ne0 >> ne1 >> source_offset >> raw_hex)) {
                std::fprintf(stderr, "malformed tensor line: %s\n", line.c_str());
                ++failures;
                continue;
            }
            const std::vector<uint8_t> raw = unhex(raw_hex);
            strata::TensorInfo meta;
            meta.name = name;
            meta.shape.push_back(ne0);
            meta.shape.push_back(ne1);
            meta.offset = source_offset;

            strata::PackIndexEntry e;
            e.name = name;
            e.file = file;
            e.source_type = type;
            std::vector<uint8_t> blob;
            const uint64_t base = seg_base[file];
            const std::string err = strata::pack_tensor_entry(name, meta, type, raw.data(), raw.size(), base,
                                                              blob, e.tensor);
            if (!err.empty()) {
                std::fprintf(stderr, "FAIL %s: tensor_entry refused: %s\n", name.c_str(), err.c_str());
                ++failures;
                continue;
            }
            seg_base[file] = base + blob.size();
            man.tensors.push_back(std::move(e));
        } else if (what == "expert") {
            strata::PackManifestExpertLayer L;
            ls >> L.layer >> L.role_offsets[0] >> L.role_offsets[1] >> L.role_offsets[2] >> L.offset;
            man.expert_layers.push_back(L);
        } else if (what == "file") {
            // the reference's whole pack, for pack_build_test; not needed to emit the manifest
        } else {
            std::fprintf(stderr, "unknown golden line: %s\n", line.c_str());
            ++failures;
        }
    }
    if (man.tensors.empty() || man.expert_layers.empty()) {
        std::fprintf(stderr, "the golden carries no tensors or no expert layers\n");
        return 1;
    }
    man.shard1 = "shard1.gguf";
    man.shard2 = "shard2.gguf";
    // The source hash comes from the golden: the C++ side of this test has no shard file to hash, and the hash
    // itself is what sha256_test covers (the FIPS vectors and a streamed million bytes).
    man.files_experts = man.n_layers * man.geometry.NE * man.geometry.blob_bytes();
    man.files_dense = seg_base["dense.bin"];
    man.files_embd = seg_base["embd.bin"];

    std::string err;
    const std::string got = strata::pack_manifest_json(man, err);
    if (!err.empty()) {
        std::fprintf(stderr, "FAIL the manifest was refused: %s\n", err.c_str());
        return 1;
    }
    if (without_paths(got) != without_paths(want)) {
        std::fprintf(stderr, "manifest.json differs from the reference:\n%s\n",
                     first_diff(without_paths(got), without_paths(want)).c_str());
        ++failures;
    }

    // The pack markers: any one of them in a directory means it already holds a pack, and a build must refuse it
    // rather than overwrite whatever was there.
    {
        char dir[] = "pack-occupied-XXXXXX";   // beside the test's working directory, not in the system temp dir
        if (mkdtemp(dir) == nullptr) {
            std::fprintf(stderr, "FAIL could not make a temporary directory\n");
            ++failures;
        } else {
            const std::string d(dir);
            if (!strata::pack_occupied(d).empty()) {
                std::fprintf(stderr, "FAIL an empty directory was reported as holding a pack\n");
                ++failures;
            }
            for (const char* name : {"manifest.json", "index.txt", "native_experts.txt"}) {
                std::ofstream(d + "/" + name).put('x');
                const std::string got_marker = strata::pack_occupied(d);
                if (got_marker != name) {
                    std::fprintf(stderr, "FAIL %s was not reported as a pack marker (got '%s')\n", name,
                                 got_marker.c_str());
                    ++failures;
                }
                std::remove((d + "/" + name).c_str());
            }
            std::remove(d.c_str());
        }
    }

    if (failures) {
        std::fprintf(stderr, "pack_manifest_test: FAILED (%d)\n", failures);
        return 1;
    }
    std::printf("pack_manifest_test: the manifest for %zu tensors and %zu expert layer(s) matches the reference, "
                "and every pack marker is recognised\n", man.tensors.size(), man.expert_layers.size());
    return 0;
}
