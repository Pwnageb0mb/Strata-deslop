"""tools/gen_mtp_rt_blob_golden.py - tools/mtp_rt.py's own expert blob, at the PRODUCTION geometry.

`mtp_rt.blob_of` is a second, independent implementation of the relayout `expert_arena::pack_expert_blob` does
(the MTP draft layer, plan v0.3 P6).  The arena's own golden runs at a SCALED geometry (H=256/FF=64/NE=2); this
runs the reference at H=2560/FF=640/NE=512 - the geometry the MTP layer actually uses - over synthetic Q2_0 blocks,
so the port is checked against the reference's bytes where it matters and not only where a fixture fits.

The inputs are a FORMULA, not bytes in the golden: both sides generate the same blocks, so the golden is the
reference's digest of its own output rather than 1.4 MB of hex.

    .venv/bin/python tools/gen_mtp_rt_blob_golden.py           # rewrite
    .venv/bin/python tools/gen_mtp_rt_blob_golden.py --check    # fail if the reference's output changed
"""
from __future__ import annotations

import argparse
import hashlib
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import mtp_rt as M  # noqa: E402

OUT = ROOT / "tests" / "artifact" / "mtp_rt_blob_golden.txt"
GU_BYTES = 2 * M.FF * (M.H // 64) * 18        # gate rows then up rows, each block 18 bytes
DN_BYTES = M.H * (M.FF // 64) * 18


def pattern(n: int) -> bytes:
    """The synthetic Q2_0 blocks, identical on both sides: byte k is (k * 37 + 11) & 0xFF."""
    return bytes(((k * 37 + 11) & 0xFF) for k in range(n))


def build() -> str:
    gu = np.frombuffer(pattern(GU_BYTES), dtype=np.uint8).reshape(2 * M.FF, M.H // 64, 18)
    dn = np.frombuffer(pattern(DN_BYTES), dtype=np.uint8).reshape(M.H, M.FF // 64, 18)
    blob = M.blob_of(gu, dn)
    assert len(blob) == M.BLOB, (len(blob), M.BLOB)
    return "\n".join([
        "# mtp_rt blob golden: tools/mtp_rt.py's blob_of at the production geometry (gen_mtp_rt_blob_golden.py)",
        f"H {M.H}", f"FF {M.FF}", f"NE {M.NE}",
        "pattern (k * 37 + 11) & 0xFF",
        f"gu_bytes {GU_BYTES}", f"dn_bytes {DN_BYTES}", f"blob_bytes {len(blob)}",
        f"sha256 {hashlib.sha256(blob).hexdigest()}",
        "",
    ])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    text = build()
    if a.check:
        if not OUT.exists() or OUT.read_text(encoding="utf-8") != text:
            raise SystemExit("the mtp_rt blob golden is out of date - rerun without --check")
        print("the mtp_rt blob golden is current")
        return 0
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)} (blob {M.BLOB} bytes at H={M.H} FF={M.FF} NE={M.NE})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
