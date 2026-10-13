"""tools/gen_expert_arena_golden.py - the expert arena's layout, from the reference that defines it.

`tools/pack_layer.py` lays out one expert blob (1,382,400 bytes for the real model): the Q2_0 codes and their
fp16 scales in SEPARATE PLANES, with the gate and up ROWS interleaved so one pass over the activation serves
both.  `strata_pack.py`'s `rebuild_role` is the inverse, written as a separate statement on purpose - "if it and
`pack_layer` were the same code the arena would be unverified by construction".

    .venv/bin/python tools/gen_expert_arena_golden.py           # rewrite
    .venv/bin/python tools/gen_expert_arena_golden.py --check    # fail if the reference's output changed

The geometry is SCALED, not reimplemented: a synthetic GGUF is written with the three role tensors as Q2_0 and the
reference's own `build()` runs over it with its module constants set to a fixture-sized model.  Its layout code is
untouched; only the model's numbers change.

    geom H=.. FF=.. NE=.. QK=.. block=.. row_gu=.. row_d=.. scales_gu=.. scales_d=.. role=.. blob=..
    offsets gu_codes=.. down_codes=.. gu_scales=.. down_scales=..
    role <gate|up|down> <hex>       the three source tensors, per expert, as the GGUF stores them
    blob <hex>                      the reference's experts.bin
    inverse <gate|up|down> <hex>    the reference's rebuild_role of expert 0 - which must equal its source run
"""
from __future__ import annotations

import argparse
import pathlib
import shutil
import sys
import tempfile

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "ref"))

import gguf_reader as G  # noqa: E402
import pack_layer as PL  # noqa: E402
import strata_pack as SP  # noqa: E402
from gguf_writer import GGUFWriter  # noqa: E402

OUT = ROOT / "tests" / "artifact" / "expert_arena_golden.txt"

# A fixture-sized model.  H*FF must be a multiple of QK, and H and FF of QK, for the planes to be whole blocks.
H, FF, NE = 256, 64, 2


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


def role_values(rows: int, in_features: int, seed: int) -> np.ndarray:
    """A deterministic (rows, in_features) block of weights.  Rows are expert-major: the expert index is the LAST
    axis of the GGUF shape, so one expert is a contiguous run."""
    n = rows * in_features
    i = np.arange(n, dtype=np.float64)
    return ((np.sin(i * 0.017 + seed) * 0.7 + np.cos(i * 0.0031 + seed) * 0.3).astype(np.float32)).reshape(
        rows, in_features)


def build() -> str:
    scale_geometry()
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="strata-expert-arena-"))
    try:
        gguf = tmp / "fixture-00001-of-00001.gguf"
        w = GGUFWriter()
        w.add("general.architecture", "qwen4exp")
        w.add("general.alignment", 32)
        # gate/up: [in=H, out=FF, experts=NE], one row per (expert, out) of H elements = SCALES_GU blocks
        w.add_q2_0("blk.0.ffn_gate_exps.weight", role_values(NE * FF, H, 0.0), [H, FF, NE])
        w.add_q2_0("blk.0.ffn_up_exps.weight", role_values(NE * FF, H, 1.0), [H, FF, NE])
        # down: [in=FF, out=H, experts=NE], one row per (expert, out) of FF elements = SCALES_D blocks
        w.add_q2_0("blk.0.ffn_down_exps.weight", role_values(NE * H, FF, 2.0), [FF, H, NE])
        w.write(gguf)

        PL.SHARD1 = gguf
        experts = tmp / "experts.bin"
        man = PL.build(1, experts)
        blob = experts.read_bytes()
        assert len(blob) == NE * PL.BLOB_BYTES, (len(blob), NE, PL.BLOB_BYTES)

        # the three role tensors as stored, so the C++ test builds from the same bytes
        g = G.GGUFFile(gguf)
        by_name = {t.name: t for t in g.tensors}
        with open(gguf, "rb") as fh:
            src = fh.read()
        role_bytes = {}
        for role in ("gate", "up", "down"):
            t = by_name["blk.0.ffn_%s_exps.weight" % role]
            n = t.expected_bytes()
            role_bytes[role] = src[g.data_start + t.offset: g.data_start + t.offset + n]
            assert len(role_bytes[role]) == NE * PL.ROLE_BYTES, (role, len(role_bytes[role]))

        # the reference's INVERSE, on the first expert's blob, must give that expert's source run back
        one = blob[:PL.BLOB_BYTES]
        inv = {}
        for role in ("gate", "up", "down"):
            got = SP.rebuild_role(one, role)
            want = role_bytes[role][:PL.ROLE_BYTES]
            assert got == want, role
            inv[role] = got
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    lines = ["# expert_arena golden v1 - tools/pack_layer.py's own layout (gen_expert_arena_golden.py)",
             "geom H=%d FF=%d NE=%d QK=%d block=%d row_gu=%d row_d=%d scales_gu=%d scales_d=%d role=%d blob=%d"
             % (PL.H, PL.FF, PL.NE, PL.QK, PL.BLOCK_BYTES, PL.ROW_BYTES_GU, PL.ROW_BYTES_D, PL.SCALES_GU,
                PL.SCALES_D, PL.ROLE_BYTES, PL.BLOB_BYTES),
             "offsets gu_codes=%d down_codes=%d gu_scales=%d down_scales=%d"
             % (PL.OFF["gate_up_codes"], PL.OFF["down_codes"], PL.OFF["gate_up_scales"], PL.OFF["down_scales"])]
    for role in ("gate", "up", "down"):
        lines.append("role %s %s" % (role, role_bytes[role].hex()))
    lines.append("blob %s" % blob.hex())
    for role in ("gate", "up", "down"):
        lines.append("inverse %s %s" % (role, inv[role].hex()))
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="record the reference's expert arena layout")
    ap.add_argument("--check", action="store_true", help="do not write; fail when the golden is out of date")
    a = ap.parse_args()
    text = build()
    if a.check:
        if OUT.read_text(encoding="utf-8") != text:
            print("the reference's output changed: rerun tools/gen_expert_arena_golden.py", file=sys.stderr)
            return 1
        print("the expert_arena golden is current")
        return 0
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)} (H={H} FF={FF} NE={NE})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
