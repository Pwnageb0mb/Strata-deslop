"""tools/gen_mtp_pack_golden.py - the acceptance for tools/mtp_pack.py.

`tools/mtp_pack.py` packs the MTP draft head's `mtp.*` tensors into a Strata GGUF: the routed experts quantized
(q2_0 / q4_0 / q8_0) and everything else carried through as BF16 (F32 for 1-D norms).  It builds the file with
llama.cpp's gguf-py, so the C++ port is a transcription of that container layout AND of the three quantizers.

It writes a checkpoint small enough to commit, runs the reference over it in all three expert formats, and
records what it produced: the stdout, the output's size and digest, its first bytes, and the sidecar report.

    .venv/bin/python tools/gen_mtp_pack_golden.py           # write the fixture and the golden
    .venv/bin/python tools/gen_mtp_pack_golden.py --check    # is the committed golden current?

The one field that cannot be compared is the report's per-tensor `seconds`, which is wall-clock.  Both sides
normalize it (see `_normalize`), and nothing else is normalized.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import shutil
import struct
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "artifact" / "mtp_pack_fixture"
GOLDEN = ROOT / "tests" / "artifact" / "mtp_pack_golden.txt"

# Small, but big enough that every mechanism fires: gate_up's inner axis is two Q2_0 blocks AND four Q4_0/Q8_0
# blocks, three experts so the per-expert loop runs more than once, and values with a spread so the q2_0 scale
# search actually has something to choose between.
GATE_UP = [3, 4, 128]
DOWN = [3, 8, 64]
NORM1D = [128]
BF2D = [4, 8]
ARMS = ("q2_0", "q4_0", "q8_0")


def bf16_file(n: int, seed: int) -> bytes:
    """`n` bf16 values with a mantissa and a spread, written as their stored patterns."""
    out = bytearray()
    for i in range(n):
        v = ((i * 37 + seed * 11) % 211 - 105) * 0.015625 + (seed - 2) * 0.75
        bits = struct.unpack("<I", struct.pack("<f", v))[0]
        out += struct.pack("<H", (bits >> 16) & 0xFFFF)
    return bytes(out)


def build_fixture() -> None:
    shutil.rmtree(FIXTURE, ignore_errors=True)
    FIXTURE.mkdir(parents=True, exist_ok=True)
    tensors = [
        ("mtp.layers.0.input_layernorm.weight", BF2D, 1),
        ("mtp.layers.0.mlp.experts.down_proj", DOWN, 2),
        ("mtp.layers.0.mlp.experts.gate_up_proj", GATE_UP, 3),
        ("mtp.layers.0.self_attn.q_norm.weight", NORM1D, 4),
    ]
    manifest = []
    for name, shape, seed in tensors:
        n = 1
        for d in shape:
            n *= d
        data = bf16_file(n, seed)
        (FIXTURE / (name.replace(".", "_") + ".bf16")).write_bytes(data)
        manifest.append({"name": name, "file": name.replace(".", "_") + ".bf16", "shape": shape,
                         "sha256": hashlib.sha256(data).hexdigest()})
    (FIXTURE / "mtp-manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")


def _normalize(text: str) -> str:
    """The report's `seconds` is wall-clock; nothing else about the output is volatile."""
    return re.sub(r'"seconds": [0-9.]+', '"seconds": 0.0', text)


def run(arm: str, out: pathlib.Path) -> str:
    p = subprocess.run([sys.executable, str(ROOT / "tools" / "mtp_pack.py"), "--src", str(FIXTURE),
                        "--experts", arm, "--out", str(out)], capture_output=True, text=True)
    if p.returncode != 0:
        sys.exit("mtp_pack.py (%s) failed: %s" % (arm, (p.stderr or p.stdout).strip()))
    return p.stdout


def golden_text() -> str:
    build_fixture()
    scratch = pathlib.Path(tempfile.mkdtemp(prefix="strata-mtp-pack-"))
    try:
        lines = ["# mtp_pack golden v1 - tools/mtp_pack.py's own output (gen_mtp_pack_golden.py)",
                 "fixture mtp_pack_fixture",
                 "geometry gate_up %s down %s norm1d %s bf2d %s"
                 % ("x".join(map(str, GATE_UP)), "x".join(map(str, DOWN)), NORM1D[0], "x".join(map(str, BF2D)))]
        for arm in ARMS:
            out = scratch / ("mtp-%s.gguf" % arm)
            stdout = run(arm, out)
            blob = out.read_bytes()
            report = _normalize(pathlib.Path(str(out) + ".report.json").read_text(encoding="utf-8"))
            lines.append("## " + arm)
            lines.append("file mtp-%s.gguf bytes %d sha256 %s" % (arm, len(blob), hashlib.sha256(blob).hexdigest()))
            for line in stdout.rstrip("\n").split("\n"):
                lines.append("stdout " + line.replace(str(out), "<out>"))
            lines.append("head_hex " + blob[:160].hex())
            lines.append("report_begin")
            lines.append(report)
            lines.append("report_end")
        return "\n".join(lines) + "\n"
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="compare with the committed golden instead of writing")
    a = ap.parse_args()
    text = golden_text()
    if a.check:
        if not GOLDEN.exists():
            sys.exit("the golden is missing: " + str(GOLDEN))
        if GOLDEN.read_text(encoding="utf-8") != text:
            sys.exit("the golden is stale: re-run without --check")
        print("the mtp_pack golden is current")
        return 0
    GOLDEN.write_text(text, encoding="utf-8")
    print("wrote %s" % GOLDEN.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
