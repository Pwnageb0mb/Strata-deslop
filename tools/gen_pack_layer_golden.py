"""tools/gen_pack_layer_golden.py - the pack-layout arena TOOL, from the reference that defines it.

`tools/pack_layer.py`'s layout (the blob relayout) is already ported and held by `expert_arena_test`.  What is
NOT yet ported is the tool around it: read a model's GGUF, cut every layer's experts into blobs, write the arena
and the sidecar manifest.  That is what this golden is for, and it is a different acceptance from
`expert_arena_golden.txt` - this one drives the reference's `build()` over a COMMITTED fixture GGUF and records
the two files it writes, so the C++ tool is compared against the reference's own bytes rather than against the
layout function that both would share.

    .venv/bin/python tools/gen_pack_layer_golden.py           # rewrite the fixture and the golden
    .venv/bin/python tools/gen_pack_layer_golden.py --check    # fail if the reference's output changed

The geometry is SCALED, not reimplemented: a synthetic GGUF is written with each layer's three role tensors as
Q2_0 and the reference's own `build()` runs over it with its module constants set to a fixture-sized model.  Its
layout code is untouched; only the model's numbers change.
"""
from __future__ import annotations

import argparse
import hashlib
import pathlib
import shutil
import sys
import tempfile

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "ref"))

import pack_layer as PL  # noqa: E402
from gguf_writer import GGUFWriter  # noqa: E402

FIXTURE = ROOT / "tests" / "artifact" / "pack_layer_fixture.gguf"
GOLDEN = ROOT / "tests" / "artifact" / "pack_layer_golden.txt"

# A fixture-sized model.  H*FF must be a multiple of QK, and H and FF of QK, for the planes to be whole blocks.
H, FF, NE = 256, 64, 2
N_LAYERS = 2   # more than one, so the tool's per-layer loop and its running offset are both exercised


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


def role_values(rows: int, in_features: int, seed: float) -> np.ndarray:
    """A deterministic (rows, in_features) block of weights.  The expert index is the LAST axis of the GGUF
    shape, so one expert is a contiguous run."""
    n = rows * in_features
    i = np.arange(n, dtype=np.float64)
    return ((np.sin(i * 0.017 + seed) * 0.7 + np.cos(i * 0.0031 + seed) * 0.3).astype(np.float32)).reshape(
        rows, in_features)


def write_fixture() -> None:
    """The model the tool reads.  Committed, because the tool takes a path."""
    w = GGUFWriter()
    w.add("general.architecture", "qwen4exp")
    w.add("general.alignment", 32)
    for layer in range(N_LAYERS):
        w.add_q2_0("blk.%d.ffn_gate_exps.weight" % layer, role_values(NE * FF, H, layer + 0.0), [H, FF, NE])
        w.add_q2_0("blk.%d.ffn_up_exps.weight" % layer, role_values(NE * FF, H, layer + 1.0), [H, FF, NE])
        w.add_q2_0("blk.%d.ffn_down_exps.weight" % layer, role_values(NE * H, FF, layer + 2.0), [FF, H, NE])
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    w.write(FIXTURE)


def golden_text() -> str:
    scale_geometry()
    write_fixture()
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="strata-pack-layer-"))
    try:
        PL.SHARD1 = FIXTURE
        out = tmp / "layer.bin"
        # build() prints the per-layer lines; capture them by running it through a redirect
        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            man = PL.build(N_LAYERS, out)
        stdout = buf.getvalue()
        blob = out.read_bytes()
        manifest = out.with_suffix(".json").read_text(encoding="utf-8")
        assert len(blob) == man["total_bytes"] == N_LAYERS * NE * PL.BLOB_BYTES, len(blob)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    lines = ["# pack_layer golden v1 - tools/pack_layer.py's own build() (gen_pack_layer_golden.py)",
             "fixture pack_layer_fixture.gguf",
             "geom H=%d FF=%d NE=%d QK=%d block=%d role=%d blob=%d"
             % (PL.H, PL.FF, PL.NE, PL.QK, PL.BLOCK_BYTES, PL.ROLE_BYTES, PL.BLOB_BYTES),
             "layers %d" % N_LAYERS]
    for line in stdout.rstrip("\n").split("\n"):
        lines.append("stdout " + line)
    lines.append("bin bytes %d sha256 %s" % (len(blob), hashlib.sha256(blob).hexdigest()))
    lines.append("manifest_begin")
    lines.append(manifest)
    lines.append("manifest_end")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="record the reference's pack-layout arena tool output")
    ap.add_argument("--check", action="store_true", help="do not write; fail when the golden is out of date")
    a = ap.parse_args()
    text = golden_text()
    if a.check:
        if GOLDEN.read_text(encoding="utf-8") != text:
            print("the reference's output changed: rerun tools/gen_pack_layer_golden.py", file=sys.stderr)
            return 1
        print("the pack_layer golden is current")
        return 0
    GOLDEN.write_text(text, encoding="utf-8")
    print("wrote %s (H=%d FF=%d NE=%d, %d layers)" % (GOLDEN.relative_to(ROOT), H, FF, NE, N_LAYERS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
