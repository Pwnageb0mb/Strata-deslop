"""tools/gen_pack_build_golden.py - the reference pack builder's own manifest.json, for a fixture-sized model.

`tools/strata_pack.py`'s `build()` assembles a pack: the expert arena, the two tensor segments, the manifest that
describes them, and the hashes of the source.  This runs it over a synthetic model with the geometry scaled to a
fixture, and records what it wrote - so a C++ build can be compared against the reference's output rather than
against a reading of its source.

    .venv/bin/python tools/gen_pack_build_golden.py           # rewrite
    .venv/bin/python tools/gen_pack_build_golden.py --check    # fail if the reference's output changed

The golden carries what the C++ needs to rebuild the same manifest: the geometry, the align, the tensors in the
GGUF's own order (with their type, file, shape and raw bytes) and each layer's role offsets.  Then
`manifest-begin` and the reference's manifest.json, with the temporary directory in every path replaced by
`<tmp>` - the one part of it that is machine-specific.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import pathlib
import shutil
import sys
import tempfile

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "ref"))

import gguf_reader as G  # noqa: E402
import pack_index as PI  # noqa: E402
import pack_layer as PL  # noqa: E402
import strata_pack as SP  # noqa: E402
from gguf_writer import GGUFWriter  # noqa: E402

OUT = ROOT / "tests" / "artifact" / "pack_build_golden.txt"

H, FF, NE = 256, 64, 2
# Q2_0 planes have to be whole 64-byte blocks for the manifest's contiguity check, so the dense Q2_0 tensors get
# 64 groups (4,096 elements): 1,024 code bytes and 128 scale bytes.
Q2_0_ELEMS = 4096
P_ELEMS = 8
EMBD = "token_embd.weight"


def scale_geometry() -> None:
    """Point the reference's own layout at the fixture's numbers.  Its code is not touched."""
    PL.H, PL.FF, PL.NE = H, FF, NE
    PL.QK, PL.BLOCK_BYTES = 64, 18
    PL.ROW_BYTES_GU = PL.H * 2 // 8
    PL.ROW_BYTES_D = PL.FF * 2 // 8
    PL.SCALES_GU = PL.H // PL.QK
    PL.SCALES_D = PL.FF // PL.QK
    PL.ROLE_BYTES = PL.H * PL.FF * PL.BLOCK_BYTES // PL.QK
    PL.BLOB_BYTES = 3 * PL.ROLE_BYTES
    PL.OFF = PL.blob_offsets()


def values(n: int, seed: float) -> np.ndarray:
    i = np.arange(n, dtype=np.float64)
    return (np.sin(i * 0.017 + seed) * 0.7 + np.cos(i * 0.0031 + seed) * 0.3).astype(np.float32)


def build() -> str:
    scale_geometry()
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="strata-pack-build-"))
    try:
        gguf = tmp / "fixture-00001-of-00002.gguf"   # shard 1 of 2, with shard 2 absent: the build records where
                                                     # shard 2 would be and skips the PLE-table lookup
        w = GGUFWriter()
        w.add("general.architecture", "qwen4exp")
        w.add("general.alignment", 32)
        # the expert arena: [in, out, experts], one contiguous run per expert
        w.add_q2_0("blk.0.ffn_gate_exps.weight", values(NE * FF * H, 0.0).reshape(NE * FF, H), [H, FF, NE])
        w.add_q2_0("blk.0.ffn_up_exps.weight", values(NE * FF * H, 1.0).reshape(NE * FF, H), [H, FF, NE])
        w.add_q2_0("blk.0.ffn_down_exps.weight", values(NE * H * FF, 2.0).reshape(NE * H, FF), [FF, H, NE])
        # the dense tensors, one of each kind the engine form has: S2, P32, P16, and the embedding
        w.add_q2_0("blk.0.attn_qkv.weight", values(Q2_0_ELEMS, 3.0).reshape(1, Q2_0_ELEMS), [Q2_0_ELEMS, 1])
        w.add_f32("blk.0.attn_q_norm.weight", values(P_ELEMS, 4.0).reshape(P_ELEMS, 1))
        w.add_f32("blk.0.ffn_gate_inp.weight", values(P_ELEMS, 5.0).reshape(P_ELEMS, 1))
        w.add_f32("blk.0.ssm_alpha.weight", values(P_ELEMS, 6.0).reshape(P_ELEMS, 1))
        w.add_q2_0(EMBD, values(Q2_0_ELEMS, 7.0).reshape(1, Q2_0_ELEMS), [Q2_0_ELEMS, 1])
        w.write(gguf)

        out_dir = tmp / "pack"
        argv = sys.argv
        sys.argv = ["strata_pack.py", "build", "--gguf", str(gguf), "--out", str(out_dir)]
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                rc = SP.build(gguf, out_dir, 1, False)
        finally:
            sys.argv = argv
        if rc != 0:
            raise SystemExit("the reference refused the fixture (exit %d)" % rc)
        # the engine's index.txt is a SECOND step in the reference (setup.py runs pack_index.py after the build),
        # so run it here too: the C++ build produces both from the same entries.
        argv = sys.argv
        sys.argv = ["pack_index.py", "--pack", str(out_dir)]
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                rc = PI.main()
        finally:
            sys.argv = argv
        if rc != 0:
            raise SystemExit("the reference's pack_index refused the fixture (exit %d)" % rc)
        manifest = (out_dir / "manifest.json").read_text(encoding="utf-8").replace(str(tmp), "<tmp>")
        g = G.GGUFFile(gguf)
        with open(gguf, "rb") as fh:
            src = fh.read()
        lines = ["# pack_build golden v1 - tools/strata_pack.py's own manifest.json (gen_pack_build_golden.py)",
                 "geom H=%d FF=%d NE=%d QK=%d block=%d" % (PL.H, PL.FF, PL.NE, PL.QK, PL.BLOCK_BYTES),
                 "offsets gu_codes=%d down_codes=%d gu_scales=%d down_scales=%d"
                 % (PL.OFF["gate_up_codes"], PL.OFF["down_codes"], PL.OFF["gate_up_scales"],
                    PL.OFF["down_scales"]),
                 "align %d" % SP.ALIGN,
                 "n_layers 1"]
        for t in g.tensors:
            if SP.is_expert(t.name):
                continue
            raw = src[g.data_start + t.offset: g.data_start + t.offset + t.expected_bytes()]
            shape = list(t.shape) + [1] * (2 - len(t.shape))
            lines.append("tensor %s %s %s %d %d %d %s" % (t.name, t.type_name,
                                                          "embd.bin" if t.name == EMBD else "dense.bin",
                                                          shape[0], shape[1], t.offset, raw.hex()))
        by_name = {t.name: t for t in g.tensors}
        lines.append("expert 0 %d %d %d %d" % (
            by_name["blk.0.ffn_gate_exps.weight"].offset, by_name["blk.0.ffn_up_exps.weight"].offset,
            by_name["blk.0.ffn_down_exps.weight"].offset, 0))
        lines.append("sha1 %s" % json.loads(manifest)["source"]["shard1_sha256"])
        # the reference's whole pack, so a C++ build can be compared file for file - and the model it was built
        # from, so the C++ test has something to build.  The GGUF is committed beside the golden (it is ~30 KB).
        model_dir = OUT.parent
        model_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(gguf, model_dir / "model.gguf")
        for fname in ("dense.bin", "embd.bin", "experts.bin", "index.txt"):
            lines.append("file %s %s" % (fname, (out_dir / fname).read_bytes().hex()))
        lines.append("# manifest-begin")
        return "\n".join(lines) + "\n" + manifest
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="record the reference pack builder's manifest")
    ap.add_argument("--check", action="store_true", help="do not write; fail when the golden is out of date")
    a = ap.parse_args()
    text = build()
    if a.check:
        if OUT.read_text(encoding="utf-8") != text:
            print("the reference's output changed: rerun tools/gen_pack_build_golden.py", file=sys.stderr)
            return 1
        print("the pack_build golden is current")
        return 0
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)} (H={H} FF={FF} NE={NE})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
