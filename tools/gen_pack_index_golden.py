"""tools/gen_pack_index_golden.py - the engine's index.txt, as tools/pack_index.py writes it.

`pack_index.py` flattens a pack's `manifest.json` into the one line per tensor that the engine's loader reads,
deciding there (not in the engine) which activation a weight wants and how the engine's form differs from the
pack's.  This builds a synthetic manifest from the fixture tensors, runs the reference tool over it, and records
both the entries and the index it produced.

    .venv/bin/python tools/gen_pack_index_golden.py           # rewrite
    .venv/bin/python tools/gen_pack_index_golden.py --check    # fail if the reference's output changed

The golden is two parts: one `entry` line per tensor (its source type, the file it lands in, and its raw bytes -
enough for a C++ test to rebuild the same manifest entry), then `index-begin` and the reference's index.txt
verbatim.

WHY THE C++ PORT TAKES ENTRIES AND NOT A MANIFEST.  `pack_index.py`'s own docstring gives the reason the engine
has no JSON parser: it would be a new, unaudited component whose failure mode is a wrong byte offset.  A C++
`strata-pack` that writes the manifest itself already holds these entries in memory, so it needs no parser
either - and the `--pack <dir>` re-derivation mode is the one part of the Python tool that does not port.
"""
from __future__ import annotations

import argparse
import io
import contextlib
import json
import pathlib
import shutil
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "ref"))

import pack_index as PI  # noqa: E402
import strata_pack as SP  # noqa: E402
from gen_canonical_mapping_golden import RAW, TYPES  # noqa: E402

OUT = ROOT / "tests" / "artifact" / "pack_index_golden.txt"

# Which file each fixture tensor lands in.  `token_embd.weight` is the one the real build sends to embd.bin, so
# the fixture sends one tensor there too - FILE_ID has to be exercised, not just dense.bin.
FILE_OF = {"F32": "embd.bin"}


class Meta:
    def __init__(self, shape, offset):
        self.shape = shape
        self.offset = offset


def build() -> str:
    tensors = {}
    entry_lines = []
    for tname in TYPES:
        m = SP.MAPPINGS[tname]
        if m.form in ("P16", "P32"):
            raw = RAW[tname](8)
            elements = 8
        else:
            # The planes must be CONTIGUOUS, and `put()` pads each one to 64 bytes, so a plane whose size is not
            # a multiple of 64 pushes the next one out and the reference refuses the manifest.  Real tensors are
            # large enough that every plane is a multiple of 64; this sizes the fixture so it is too:
            # 64 groups per tensor makes codes_bytes = 64*group_elems*code_bits/8 and scales_bytes = 128, both
            # multiples of 64.
            blocks = 64 * m.group_elems // m.block_elems
            raw = RAW[tname](blocks)
            elements = blocks * m.block_elems
        # A 2-D shape with ne0 the contiguous axis: the engine's plane sizes are ne1 * groups, so a shape of
        # [elements] would derive zero-length scale planes and fail the tool's own consistency check.
        meta = Meta([elements, 1], 4096)
        blob, ent = SP.tensor_entry("t." + tname, meta, tname, raw, 0)
        ent["file"] = FILE_OF.get(tname, "dense.bin")
        tensors["t." + tname] = ent
        entry_lines.append("entry %s %s %s" % (tname, ent["file"], raw.hex()))

    tmp = pathlib.Path(tempfile.mkdtemp(prefix="strata-pack-index-"))
    try:
        (tmp / "manifest.json").write_text(json.dumps({"align": 64, "tensors": tensors}), encoding="utf-8")
        argv = sys.argv
        sys.argv = ["pack_index.py", "--pack", str(tmp)]
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                rc = PI.main()
        finally:
            sys.argv = argv
        if rc != 0:
            raise SystemExit("the reference refused the fixture manifest (exit %d)" % rc)
        index = (tmp / "index.txt").read_text(encoding="utf-8")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    lines = ["# pack_index golden v1 - tools/pack_index.py's own index.txt (gen_pack_index_golden.py)",
             "# entry <type> <file> <raw_hex>",
             *entry_lines,
             "# index-begin"]
    return "\n".join(lines) + "\n" + index


def main() -> int:
    ap = argparse.ArgumentParser(description="record the reference's engine index")
    ap.add_argument("--check", action="store_true", help="do not write; fail when the golden is out of date")
    a = ap.parse_args()
    text = build()
    if a.check:
        if OUT.read_text(encoding="utf-8") != text:
            print("the reference's output changed: rerun tools/gen_pack_index_golden.py", file=sys.stderr)
            return 1
        print("the pack_index golden is current")
        return 0
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)} ({len(TYPES)} tensors)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
