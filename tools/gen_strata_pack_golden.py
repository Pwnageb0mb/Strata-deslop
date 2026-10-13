"""tools/gen_strata_pack_golden.py - the reference's own answers for the canonical pack's code packing.

`tools/strata_pack.py` is the canonical Q2_0 pack builder, and it is the tool the tree designates as the C++
`strata-pack` (its own docstring: "the C++ tools/strata-pack is a port of it, and if the two disagree the C++ is
wrong").  This records the answers of the pure functions at its base - the code packing every S2/S4/S8 form goes
through, and the scale-precision choice - so a C++ port can be compared against them without a model.

    .venv/bin/python tools/gen_strata_pack_golden.py           # rewrite
    .venv/bin/python tools/gen_strata_pack_golden.py --check    # fail if the reference's answers changed

The packing is Q2_0's own convention (4 codes per byte, LSB-first) kept for every form so the kernel has one
unpacking loop.  A port that gets the bit order wrong still produces plausible weights, which is why this is
recorded rather than reasoned about.
"""
from __future__ import annotations

import argparse
import hashlib
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "ref"))

import strata_pack as SP  # noqa: E402

OUT = ROOT / "tests" / "artifact" / "strata_pack_golden.txt"

# (bits, how many codes).  bits=2 packs four per byte, 4 packs two, 8 is a pass-through; the counts are multiples
# of the codes-per-byte so every case is whole.
CASES = [(2, 4), (2, 8), (2, 256), (4, 2), (4, 8), (4, 256), (8, 1), (8, 5), (8, 256)]


def codes_for(bits: int, n: int) -> np.ndarray:
    """A deterministic pattern that uses the whole code range, not just 0 and 1."""
    hi = (1 << bits) - 1
    return np.array([(i * 7 + i // 3) % (hi + 1) for i in range(n)], dtype=np.uint8)


def scales_fp16_exact() -> np.ndarray:
    return np.array([1.0, 0.5, -0.25, 2.0, 0.0, -0.0, 1024.0, 0.0078125], dtype=np.float32)


def scales_not_fp16() -> np.ndarray:
    return np.array([0.1, 1.0 / 3.0, 3.14159, 1e-8], dtype=np.float32)


def build() -> str:
    lines = ["# strata_pack golden v1 - tools/strata_pack.py's own answers (gen_strata_pack_golden.py)",
             "# pack <bits> <n> <codes_hex> <packed_hex>",
             "# scale <fp16?> <scales_f32_hex> <packed_hex>"]
    for bits, n in CASES:
        codes = codes_for(bits, n)
        packed = SP.pack_codes(codes, bits)
        back = SP.unpack_codes(packed, bits, n)
        assert np.array_equal(back, codes), (bits, n)   # the reference round-trips its own packing
        lines.append("pack %d %d %s %s" % (bits, n, codes.tobytes().hex(), packed.hex()))
    for scales in (scales_fp16_exact(), scales_not_fp16()):
        packed, is16 = SP.choose_scale_dtype(scales)
        lines.append("scale %d %s %s" % (1 if is16 else 0, scales.tobytes().hex(), packed.hex()))
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="record the reference's canonical code packing")
    ap.add_argument("--check", action="store_true", help="do not write; fail when the golden is out of date")
    a = ap.parse_args()
    text = build()
    if a.check:
        if OUT.read_text(encoding="utf-8") != text:
            print("the reference's answers changed: rerun tools/gen_strata_pack_golden.py", file=sys.stderr)
            return 1
        print("the strata_pack golden is current")
        return 0
    OUT.write_text(text, encoding="utf-8")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    print(f"wrote {OUT.relative_to(ROOT)} ({len(text.splitlines())} lines, sha256 {digest})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
