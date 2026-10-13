"""tools/gen_mtp_rt_golden.py - the reference's own MTP runtime files, over a committable fixture.

`tools/mtp_rt.py` writes the MTP draft layer's runtime files from a packed MTP GGUF: `experts.bin` (the blob
layout), `dense.bin` (every other tensor, Q8_0 / BF16 / F32) and `dense.txt` (the index).  This builds a fixture
GGUF carrying those tensors, runs the reference over it and records what it wrote.

THE GEOMETRY IS THE PRODUCTION ONE.  `mtp_rt`'s `blob_of` asserts `gu.shape == (NE, 2*FF, 720)`, so H and FF cannot
be scaled without patching the reference's source - and they should not be: the blob arithmetic is the thing being
ported.  Only `NE` is scaled, to 1, which changes the LOOP COUNT and not one byte of the arithmetic.  The fixture
is 1.38 MB as a result, which is what it costs to exercise the layout at its real width.

    .venv/bin/python tools/gen_mtp_rt_golden.py           # rewrite
    .venv/bin/python tools/gen_mtp_rt_golden.py --check    # fail if the reference's output changed

The binary outputs are recorded as digests and `dense.txt` as text: the C++ test computes the same digests over
its own files, so the comparison is the bytes without committing them.
"""
from __future__ import annotations

import argparse
import hashlib
import pathlib
import tempfile
import shutil
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import mtp_rt as M  # noqa: E402
from gguf_writer import GGUFWriter  # noqa: E402

OUT = ROOT / "tests" / "mtp" / "mtp_rt_golden.txt"
DIR = ROOT / "tests" / "mtp"
SCRATCH = pathlib.Path(tempfile.gettempdir()) / "strata-mtp-rt-out"

NE = 1                      # the only scaled constant: the blob arithmetic below runs at H=2560, FF=640


def bf16_of(floats: np.ndarray) -> np.ndarray:
    """The stored BF16 pattern of a float: the top 16 bits, round-half-even on the discarded half.

    A fixture built from RANDOM uint16 patterns is a trap for the Q8 path: read as a float those are huge values,
    infinities and NaNs, and the quantizer's output for them is not something a port can be held to.
    """
    u = np.ascontiguousarray(floats, dtype="<f4").view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype("<u2")


def build_fixture(path: pathlib.Path) -> None:
    rng = np.random.default_rng(11)
    w = GGUFWriter()
    w.add("general.architecture", "qwen4exp")
    # the experts, as the fused Q2_0 tensors a packed MTP GGUF carries.  add_q2_0 takes (rows, in_features) and
    # quantizes; gguf-py hands the data back shaped (NE, rows, bytes_per_row), which is what mtp_rt reads.
    w.add_q2_0("mtp.layers.0.mlp.experts.gate_up_proj",
               rng.standard_normal((NE * 2 * M.FF, M.H)).astype(np.float32), [M.H, 2 * M.FF, NE])
    w.add_q2_0("mtp.layers.0.mlp.experts.down_proj",
               rng.standard_normal((NE * M.H, M.FF)).astype(np.float32), [M.FF, M.H, NE])
    # a projection in mtp_rt's Q8 set: BF16 in, quantized to Q8_0 on the way out (width a multiple of 32)
    w.add_bf16("mtp.layers.0.self_attn.q_proj.weight",
               bf16_of(rng.standard_normal((32, 64)).astype(np.float32)))
    # a hyper-connection weight: BF16 in, BF16 out
    w.add_bf16("mtp.layers.0.hc_gate.weight", bf16_of(rng.standard_normal((8, 16)).astype(np.float32)))
    # a norm: F32 in, and the reference applies the Gemma "+1"
    w.add_f32("mtp.layers.0.attn_norm.weight", rng.standard_normal(64).astype(np.float32))
    w.write(path)


def q8_0_golden() -> list[str]:
    """The reference's own q8_0 over a deterministic array: the one piece of arithmetic the port adds.

    The values are exact in float32 ((k*37 % 101 - 50) / 16: sixteenths), so both sides generate the same input
    without depending on a libm's sin, and the quantizer's rounding is exercised on exact halves.
    """
    x = ((np.arange(256, dtype=np.int64) * 37 % 101) - 50).astype(np.float32) / np.float32(16.0)
    raw = M.q8_0(x)
    blocks = len(x) // 32
    return [f"q8_0 values {len(x)} blocks {blocks} bytes {len(raw)}",
            f"q8_0_sha256 {hashlib.sha256(raw).hexdigest()}",
            "q8_0_first " + raw[:34].hex()]


def build() -> str:
    DIR.mkdir(parents=True, exist_ok=True)
    fixture = DIR / "mtp-fixture.gguf"
    build_fixture(fixture)
    if SCRATCH.exists():
        shutil.rmtree(SCRATCH)

    saved_ne, saved_argv = M.NE, sys.argv
    M.NE = NE
    sys.argv = ["mtp_rt.py", "--gguf", str(fixture), "--out", str(SCRATCH)]
    import contextlib
    import io
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            rc = M.main()
    finally:
        M.NE, sys.argv = saved_ne, saved_argv
    if rc != 0:
        raise SystemExit(f"mtp_rt refused the fixture (exit {rc})")

    lines = ["# mtp_rt golden: tools/mtp_rt.py's own runtime files (gen_mtp_rt_golden.py)",
             "# the fixture keeps the PRODUCTION geometry (H, FF) and scales only NE, to 1",
             f"H {M.H}", f"FF {M.FF}", f"NE {NE}", f"blob {M.BLOB}",
             f"fixture mtp-fixture.gguf bytes {fixture.stat().st_size}",
             f"sha256 {hashlib.sha256(fixture.read_bytes()).hexdigest()}"]
    for name in ("experts.bin", "dense.bin"):
        blob = (SCRATCH / name).read_bytes()
        lines.append(f"file {name} bytes {len(blob)} sha256 {hashlib.sha256(blob).hexdigest()}")
    lines.append(f"dense_txt_sha256 {hashlib.sha256((SCRATCH / 'dense.txt').read_bytes()).hexdigest()}")
    for line in buf.getvalue().rstrip("\n").split("\n"):
        lines.append(f"stdout {line.replace(str(SCRATCH), '<out>')}")
    # the q8_0 section BEFORE the dense.txt block: a reader takes everything after `# dense.txt-begin` as the
    # file, so anything emitted after it would be read as part of dense.txt
    lines += q8_0_golden()
    lines.append("# dense.txt-begin")
    lines.append((SCRATCH / "dense.txt").read_text(encoding="utf-8").rstrip("\n"))
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    text = build()
    if a.check:
        if not OUT.exists() or OUT.read_text(encoding="utf-8") != text:
            raise SystemExit("the mtp_rt golden is out of date - rerun without --check")
        print("the mtp_rt golden is current")
        return 0
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)} (fixture {fixture_size()} bytes)")
    return 0


def fixture_size() -> int:
    p = DIR / "mtp-fixture.gguf"
    return p.stat().st_size if p.exists() else 0


if __name__ == "__main__":
    raise SystemExit(main())
