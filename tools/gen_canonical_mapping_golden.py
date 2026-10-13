"""tools/gen_canonical_mapping_golden.py - the reference's own answers for the canonical source-type mappings.

`tools/canonical_xcheck.py` holds MAPPINGS, which its docstring calls the specification of record: each source
type's encode into the canonical planes and the decode transcribed expression-by-expression from
`ggml-quants.c`.  This records both, for the mapping pairs the C++ port has reached, so the port can be compared
against the reference without a model.

    .venv/bin/python tools/gen_canonical_mapping_golden.py           # rewrite
    .venv/bin/python tools/gen_canonical_mapping_golden.py --check    # fail if the reference's answers changed

One line per tensor's worth of whole blocks:

    case <type> <raw_hex> <codes|-> <scales|-> <values_hex>

`codes` and `scales` are the S forms' planes (a `-` where the form has none: the P forms carry a single value
plane); `values` is what `from_*` decodes.  The traps this catches are the ones that still produce plausible
weights: Q8_0's code plane is UNSIGNED (the source's int8 stored +128), Q4_0's low nibbles fill the FIRST half of
a block and its high nibbles the SECOND, Q2_0's codes must be ONE 2-bit code per element rather than the 16 packed
bytes, and IQ4_NL's code is an INDEX into a non-affine table, not a value.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "ref"))

import canonical_xcheck as CX  # noqa: E402

OUT = ROOT / "tests" / "artifact" / "canonical_mapping_golden.txt"

# The types the port has reached.  Add to this as the C++ mappings land - the reference has all thirteen.
TYPES = ["Q2_0", "Q8_0", "Q4_0", "Q4_K", "Q5_K", "Q3_K", "Q5_0", "Q6_K", "IQ4_NL", "IQ4_XS",
         "F32", "BF16", "F16"]
SCALE = [0.5, -0.25, 1.0]


def _f16_scale(b: int) -> bytes:
    return np.float16(SCALE[b % 3]).tobytes()


def _packed_k_scales(b: int) -> bytes:
    """The 12 bytes of packed 6-bit scales and mins a K-quant block carries.  Any bytes will do: they are
    masked by get_scale_min_k4, which is the thing under test."""
    return bytes(((i * 19 + b * 5) % 256) for i in range(12))


def _f16_dmin(b: int) -> bytes:
    """A K-quant block's `dmin` - the fp16 the offsets are `-(dmin*m)` of.  Deliberately a value with a full
    mantissa, so the product is NOT fp16-exact for a nonzero m: that is the case every real tensor is in
    (`offsets_fp16` is false for all 54 tensors that have an offset plane), and a fixture with fp16-exact offsets
    trips a check in pack_index.py that only real packs avoid."""
    return np.float16([0.1, 0.3, 0.7][b % 3]).tobytes()


def q2_0_tensor(blocks: int = 3) -> bytes:
    """Q2_0: 18 bytes a block - an fp16 scale, then 16 bytes of four packed 2-bit codes each."""
    out = bytearray()
    for b in range(blocks):
        out += _f16_scale(b)
        out += bytes(((i * 41 + b * 13) % 256) for i in range(16))
    return bytes(out)


def q8_0_tensor(blocks: int = 3) -> bytes:
    """Q8_0: 34 bytes a block - an fp16 scale, then 32 int8 codes.  The codes sweep the whole range, -128
    included, because that is the value the unsigned code plane turns into 0."""
    out = bytearray()
    for b in range(blocks):
        out += _f16_scale(b)
        codes = np.array([((i * 37 + b * 11) % 256) - 128 for i in range(32)], dtype=np.int16).astype(np.int8)
        out += codes.tobytes()
    return bytes(out)


def q4_0_tensor(blocks: int = 3) -> bytes:
    """Q4_0 and IQ4_NL: 18 bytes a block - an fp16 scale, then 16 nibble bytes, low nibble the first half."""
    out = bytearray()
    for b in range(blocks):
        out += _f16_scale(b)
        out += bytes(((i * 23 + b * 7) % 256) for i in range(16))
    return bytes(out)


def f32_tensor(n: int = 8) -> bytes:
    return np.linspace(-2.0, 2.0, n, dtype=np.float32).tobytes()


def q4_k_tensor(blocks: int = 2) -> bytes:
    """Q4_K: 144 bytes a block - fp16 d, fp16 dmin, 12 packed 6-bit scale bytes, then 128 nibble bytes."""
    out = bytearray()
    for b in range(blocks):
        out += _f16_scale(b)
        out += _f16_dmin(b)
        out += _packed_k_scales(b)
        out += bytes(((i * 29 + b * 17) % 256) for i in range(128))
    return bytes(out)


def q5_k_tensor(blocks: int = 2) -> bytes:
    """Q5_K: 176 bytes a block - fp16 d, fp16 dmin, 12 packed scale bytes, 32 high-bit bytes, 128 nibble
    bytes.  `qh` does not advance inside the source's loop: the same 32 bytes are re-read by all four
    64-element steps, selected by a mask that walks 1,4,16,64 (and 2,8,32,128 for the high nibbles)."""
    out = bytearray()
    for b in range(blocks):
        out += _f16_scale(b)
        out += _f16_dmin(b)
        out += _packed_k_scales(b)
        out += bytes(((i * 31 + b * 11) % 256) for i in range(32))
        out += bytes(((i * 29 + b * 17) % 256) for i in range(128))
    return bytes(out)


def bf16_tensor(n: int = 8) -> bytes:
    """bf16 is the top half of an f32, so these are exact by construction."""
    return (np.linspace(-2.0, 2.0, n, dtype=np.float32).view(np.uint32) >> 16).astype("<u2").tobytes()


def f16_tensor(n: int = 8) -> bytes:
    return np.linspace(-2.0, 2.0, n, dtype=np.float32).astype(np.float16).tobytes()


def q3_k_tensor(blocks: int = 2) -> bytes:
    """Q3_K: 110 bytes a block - 32 hmask bytes, 64 qs bytes, 12 packed scale bytes, then fp16 d LAST."""
    out = bytearray()
    for b in range(blocks):
        out += bytes(((i * 43 + b * 7) % 256) for i in range(32))
        out += bytes(((i * 37 + b * 19) % 256) for i in range(64))
        out += _packed_k_scales(b)
        out += _f16_scale(b)
    return bytes(out)


def q5_0_tensor(blocks: int = 2) -> bytes:
    """Q5_0: 22 bytes a block - an fp16 scale, 4 bytes of high bits read as ONE little-endian uint32, then
    16 nibble bytes."""
    out = bytearray()
    for b in range(blocks):
        out += _f16_scale(b)
        out += bytes(((i * 53 + b * 23) % 256) for i in range(4))
        out += bytes(((i * 23 + b * 7) % 256) for i in range(16))
    return bytes(out)


def q6_k_tensor(blocks: int = 2) -> bytes:
    """Q6_K: 210 bytes a block - 128 ql bytes, 64 qh bytes, 16 SIGNED scale bytes, then fp16 d LAST."""
    out = bytearray()
    for b in range(blocks):
        out += bytes(((i * 29 + b * 17) % 256) for i in range(128))
        out += bytes(((i * 31 + b * 13) % 256) for i in range(64))
        # signed scales: the low byte of each is the magnitude, so leave the high bit clear for some and set
        # for others, to exercise a negative scale
        out += bytes((((i * 11 + b * 5) % 120) | (0x80 if i % 3 == 0 else 0)) for i in range(16))
        out += _f16_scale(b)
    return bytes(out)


def iq4_xs_tensor(blocks: int = 2) -> bytes:
    """IQ4_XS: 136 bytes a block - fp16 d, a uint16 scales_h, 4 bytes of scales_l, then 128 nibble bytes."""
    out = bytearray()
    for b in range(blocks):
        out += _f16_scale(b)
        out += bytes(((i * 61 + b * 29) % 256) for i in range(2))
        out += bytes(((i * 47 + b * 11) % 256) for i in range(4))
        out += bytes(((i * 23 + b * 7) % 256) for i in range(128))
    return bytes(out)


RAW = {"Q2_0": q2_0_tensor, "Q8_0": q8_0_tensor, "Q4_0": q4_0_tensor, "Q4_K": q4_k_tensor,
       "Q5_K": q5_k_tensor, "Q3_K": q3_k_tensor, "Q5_0": q5_0_tensor, "Q6_K": q6_k_tensor,
       "IQ4_NL": q4_0_tensor, "IQ4_XS": iq4_xs_tensor, "F32": f32_tensor, "BF16": bf16_tensor,
       "F16": f16_tensor}


def build() -> str:
    lines = ["# canonical_mapping golden v1 - tools/canonical_xcheck.py's own answers "
             "(gen_canonical_mapping_golden.py)",
             "# case <type> <raw_hex> <codes|-> <scales|-> <offsets|-> <values_hex>"]
    for tname in TYPES:
        raw = RAW[tname]()
        m = CX.MAPPINGS[tname]
        parts = m.enc(raw)
        assert 2 <= len(parts) <= 3, tname
        values = m.dec(parts)
        n = len(raw) // m.block_bytes
        assert len(values) == n * m.block_elems, (tname, len(values), n, m.block_elems)
        offsets_hex = "-"
        if m.form in ("P16", "P32"):
            # one value plane and nothing else: the `ones` the reference returns in the scales slot is a
            # tuple-shape artifact, not something the pack stores
            assert np.array_equal(parts[1], np.ones(len(parts[1]), dtype=np.float32)), tname
            codes_hex = scales_hex = "-"
        else:
            codes = np.asarray(parts[0], dtype=np.uint8)
            scales = np.asarray(parts[1], dtype=np.float32)
            assert codes.size == n * m.block_elems, (tname, codes.size, n, m.block_elems)
            # one scale per GROUP: groups per block is block_elems/group_elems, which is 1 where the two are
            # equal (Q8_0, Q2_0) and 8 for Q4_K's 256-element blocks of 32-element groups
            n_groups = n * (m.block_elems // m.group_elems)
            assert scales.size == n_groups, (tname, scales.size, n_groups)
            codes_hex = codes.tobytes().hex()
            scales_hex = scales.astype("<f4").tobytes().hex()
            if len(parts) == 3:
                offsets = np.asarray(parts[2], dtype=np.float32)
                assert offsets.size == n_groups, (tname, offsets.size, n_groups)
                offsets_hex = offsets.astype("<f4").tobytes().hex()
        lines.append("case %s %s %s %s %s %s" % (
            tname, raw.hex(), codes_hex, scales_hex, offsets_hex,
            np.asarray(values, dtype="<f4").tobytes().hex()))
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="record the reference's canonical source-type mappings")
    ap.add_argument("--check", action="store_true", help="do not write; fail when the golden is out of date")
    a = ap.parse_args()
    text = build()
    if a.check:
        if OUT.read_text(encoding="utf-8") != text:
            print("the reference's answers changed: rerun tools/gen_canonical_mapping_golden.py", file=sys.stderr)
            return 1
        print("the canonical_mapping golden is current")
        return 0
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)} ({len(TYPES)} types, {len(text.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
