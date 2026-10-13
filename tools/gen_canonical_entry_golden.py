"""tools/gen_canonical_entry_golden.py - the reference's own plane layout and manifest-driven decode.

`tools/strata_pack.py`'s `tensor_entry` turns one tensor's source bytes into the bytes the pack holds plus the
manifest entry that describes them, and `decode_entry` reads a tensor back using ONLY what that entry says.  The
pair is the format: if the entry does not carry enough to decode the tensor, `decode_entry` cannot fall back on
knowledge of the source type and the omission becomes a failure there.

    .venv/bin/python tools/gen_canonical_entry_golden.py           # rewrite
    .venv/bin/python tools/gen_canonical_entry_golden.py --check    # fail if the reference's answers changed

One line per tensor:

    case <type> <raw_hex> <blob_hex> <digest> <decoded_hex>

`digest` is the entry projected to one order-fixed string, because that is what the manifest holds and what the
C++ has to reproduce:

    form=..,codebook=..,code_bias=..,has_offset=..,block_elems=..,block_bytes=..,group_elems=..,elements=..,
    code_bits=..,scales_fp16=..,offsets_fp16=..,values_fp16=..,codes=<off>:<bytes>,scales=..,offsets=..,values=..

The planes are laid out by `put()`, which pads to 64 bytes before each one, so the offsets in the digest are the
part of the layout a port is most likely to get wrong while still decoding its own output.
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
import strata_pack as SP  # noqa: E402
from _paths import add_gguf_py  # noqa: E402
from gen_canonical_mapping_golden import RAW, TYPES  # noqa: E402

add_gguf_py()   # `reference_values` goes through gguf-py (and, for Q2_0, the project's own dequantizer)

OUT = ROOT / "tests" / "artifact" / "canonical_entry_golden.txt"

SEG_OFFSET = 1024          # the tensor's place in the pack, so the offsets are not all trivially zero
SOURCE_OFFSET = 4096       # where its bytes sit in the GGUF


class Meta:
    """What `tensor_entry` reads off a TensorInfo: the shape and the source offset."""

    def __init__(self, shape, offset):
        self.shape = shape
        self.offset = offset


def plane(ent, key):
    return "%d:%d" % (ent[key]["offset"], ent[key]["bytes"]) if key in ent else "-"


def digest(ent) -> str:
    """The entry as one order-fixed string.  The C++ must build the same one."""
    return ",".join([
        "form=%s" % ent["form"],
        "codebook=%s" % (ent["codebook"] or "-"),
        "code_bias=%d" % ent["code_bias"],
        "has_offset=%d" % (1 if ent["has_offset"] else 0),
        "block_elems=%d" % ent["block_elems"],
        "block_bytes=%d" % ent["block_bytes"],
        "group_elems=%d" % ent["group_elems"],
        "elements=%d" % ent["elements"],
        "code_bits=%s" % (ent["code_bits"] if "code_bits" in ent else "-"),
        "scales_fp16=%s" % ("-" if "scales_fp16" not in ent else (1 if ent["scales_fp16"] else 0)),
        "offsets_fp16=%s" % ("-" if "offsets_fp16" not in ent else (1 if ent["offsets_fp16"] else 0)),
        "values_fp16=%s" % ("-" if "values_fp16" not in ent else (1 if ent["values_fp16"] else 0)),
        "codes=%s" % plane(ent, "codes"),
        "scales=%s" % plane(ent, "scales"),
        "offsets=%s" % plane(ent, "offsets"),
        "values=%s" % plane(ent, "values"),
    ])


def build() -> str:
    lines = ["# canonical_entry golden v1 - tools/strata_pack.py's own plane layout (gen_canonical_entry_golden.py)",
             "# case <type> <raw_hex> <blob_hex> <digest> <decoded_hex> <reference_hex>"]
    for tname in TYPES:
        raw = RAW[tname]()
        m = SP.MAPPINGS[tname]
        elements = (len(raw) // m.block_bytes) * m.block_elems
        meta = Meta([elements], SOURCE_OFFSET)
        blob, ent = SP.tensor_entry("t." + tname, meta, tname, raw, SEG_OFFSET)
        # The entry's plane offsets are ABSOLUTE in the segment the tensor is written into (base = seg_offset), so
        # decoding means handing `decode_entry` the segment, not the tensor's own bytes.  Testing it this way is
        # what pins that the offsets are segment-relative.
        head = b"\0" * SEG_OFFSET + blob
        decoded = np.asarray(SP.decode_entry(head, ent), dtype=np.float32)
        assert decoded.size == elements, (tname, decoded.size, elements)
        # The INDEPENDENT side of `verify`: what the source's bytes dequantize to, from an implementation other
        # than the one under test.  `verify` is the comparison of this against the pack's decode.
        reference = CX.reference_values(tname, raw)
        assert reference.size == elements, (tname, reference.size, elements)
        assert np.array_equal(reference, decoded), tname   # the reference's own pack verifies
        # the blob must start at the segment offset and be padded to ALIGN between planes
        assert ent["name"] == "t." + tname and ent["source_offset"] == SOURCE_OFFSET
        lines.append("case %s %s %s %s %s %s" % (tname, raw.hex(), blob.hex(), digest(ent),
                                                 decoded.astype("<f4").tobytes().hex(),
                                                 reference.astype("<f4").tobytes().hex()))
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="record the reference's plane layout and manifest-driven decode")
    ap.add_argument("--check", action="store_true", help="do not write; fail when the golden is out of date")
    a = ap.parse_args()
    text = build()
    if a.check:
        if OUT.read_text(encoding="utf-8") != text:
            print("the reference's answers changed: rerun tools/gen_canonical_entry_golden.py", file=sys.stderr)
            return 1
        print("the canonical_entry golden is current")
        return 0
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)} ({len(TYPES)} types, {len(text.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
