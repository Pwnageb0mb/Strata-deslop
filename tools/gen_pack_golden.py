"""tools/gen_pack_golden.py - a small model and the pack the REFERENCE packer makes of it.

`tools/iq_pack.py` is the reference.  A C++ `strata-pack` has to write the same pack, byte for byte, and the only
honest way to know is to compare against the reference's own output - so this builds a synthetic model small
enough to keep in the tree, runs the reference over it, and leaves both:

    tests/artifact/pack_golden/model.gguf     the model (a few KB)
    tests/artifact/pack_golden/reference/     what tools/iq_pack.py wrote for it
    tests/artifact/pack_golden/manifest.json  sha256 of each file, for a quick check

    .venv/bin/python tools/gen_pack_golden.py            # rewrite the fixture and the pack
    .venv/bin/python tools/gen_pack_golden.py --check    # fail if the pack no longer matches the reference

The tensors are chosen to reach every branch of the packer's conversion table - see TENSORS below - because the
branch a port gets wrong is the one with no fixture behind it.  One complete expert layer is included, so the
expert path (expert_layout, native_experts.txt, experts.bin, the sidecar) has a fixture too.  No model weights are
needed: the values are synthetic and the quantization is gguf-py's own.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import sys
import contextlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from _paths import add_gguf_py  # noqa: E402

add_gguf_py()
from gguf import GGUFWriter, GGMLQuantizationType as Q, quants  # noqa: E402
import iq_pack  # noqa: E402

OUT = ROOT / "tests" / "artifact" / "pack_golden"
# The fixture's expert count.  The real models ship 512; 32 is the smallest that still exercises the layout AND can
# be quantized, because gguf-py (and ggml) lay a Q8_0 block along the LAST axis: shape[2] is the expert count, so it
# has to be a multiple of 32 for the expert tensors to be storable quantized at all.
N_EXPERT = 32


def bf16_exact() -> np.ndarray:
    """F32 values that are exactly BF16 (low 16 bits zero): the exact path, no rounding allowed.

    This is also the router (`blk.0.ffn_gate_inp.weight`), whose shape must be [hidden, n_expert] - the expert
    path reads the expert count off it, and every layer's expert tensors must agree with it."""
    return np.tile(np.array([0.5, -0.25, 1.0, 2.0, 0.125, -0.75, 3.5, 0.0625], dtype=np.float32),
                   32 * N_EXPERT // 8).reshape(32, N_EXPERT)


def not_bf16_exact() -> np.ndarray:
    """F32 values that are NOT exactly BF16: only --compat-bf16 may take these, and it rounds."""
    return np.linspace(-1.0, 1.0, 4 * 32, dtype=np.float32).reshape(4, 32)


def quant_values() -> np.ndarray:
    return (np.linspace(-2.0, 2.0, 4 * 32, dtype=np.float32) * 0.5).reshape(4, 32)


def expert_values(a: float, b: float) -> np.ndarray:
    """One expert tensor: [per_expert, 1, n_expert], as the expert path requires (shape[2] == n_expert)."""
    return np.linspace(a, b, 32 * N_EXPERT, dtype=np.float32).reshape(32, 1, N_EXPERT)


# (name, values, stored type).  One entry per branch of iq_pack.convert() and of index_standalone():
TENSORS = [
    # quantized, no FORM entry: served from the GGUF (index kind 0, shape only)
    ("blk.0.attn_qkv.weight", quant_values, Q.Q8_0),
    # F32 and FORM says F32: written as it is (kind 2)
    ("blk.0.attn_q_norm.weight", quant_values, Q.F32),
    # F32 whose values ARE BF16: converted exactly (kind 4, method "the values are BF16's")
    ("blk.0.ffn_gate_inp.weight", bf16_exact, Q.F32),
    # F32 whose values are NOT BF16: only --compat-bf16 takes it, and it rounds
    ("blk.0.hc_attn_down.weight", not_bf16_exact, Q.F32),
    # quantized where FORM says BF16: dequantized, then rounded to BF16
    ("blk.0.ssm_alpha.weight", quant_values, Q.Q8_0),
    # the PLE key in a native encoding: served from the GGUF even though FORM says BF16
    ("blk.1.ple_key.weight", quant_values, Q.Q8_0),
    # FORM says F16: the F32 bytes go in and the loader narrows them (kind 3)
    ("blk.0.ple_conv1d.weight", quant_values, Q.F32),
    # F16 in the file, FORM says F32: widened exactly
    ("blk.0.ple_norm_conv.weight", quant_values, Q.F16),
    # the 28.8 GB PLE table in the real model: not in the pack at all
    ("per_layer_token_embd.weight", quant_values, Q.Q8_0),
    # ---- one complete expert layer, for the expert path (expert_layout / native_experts.txt / experts.bin)
    ("blk.0.ffn_gate_exps.weight", lambda: expert_values(-1.0, 1.0), Q.Q8_0),
    ("blk.0.ffn_up_exps.weight", lambda: expert_values(-0.5, 0.5), Q.Q8_0),
    ("blk.0.ffn_down_exps.weight", lambda: expert_values(-2.0, 2.0), Q.Q8_0),
]


def write_model(path: Path) -> None:
    writer = GGUFWriter(path, "qwen4exp")
    for name, values, kind in TENSORS:
        raw = values() if callable(values) else values
        writer.add_tensor(name, quants.quantize(raw, kind), raw_dtype=kind)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def build() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    # "reference", not "pack": .gitignore's bare `pack/` rule matches any directory of that name, and a golden
    # that git silently drops is not a golden.
    pack = OUT / "reference"
    if pack.exists():
        shutil.rmtree(pack)
    pack.mkdir()
    model = OUT / "model.gguf"
    write_model(model)

    parsed = iq_pack.Model(model)
    with contextlib.redirect_stdout(io.StringIO()):
        code = iq_pack.index_standalone(model, pack, parsed, compat_bf16=True)
    if code != 0:
        raise SystemExit(f"the reference refused the fixture (exit {code})")

    # compat-bf16.json records the SOURCE PATH it was given, so it is different on every machine and cannot be a
    # committed golden.  It is a note for a human about what the flag did, not something the engine reads, so the
    # fixture keeps the artifacts that are the pack: index.txt, dense.bin and conversions.json.
    (pack / "compat-bf16.json").unlink(missing_ok=True)

    # ---- the expert path: the table the engine reads (native_experts.txt), the blobs cut from the shards
    # (experts.bin) and the sidecar that says which model they were cut from.  Written the way iq_pack.main() writes
    # them, so the fixture is the reference's output and not a re-derivation of it.
    got = iq_pack.expert_layout(parsed, model)
    if isinstance(got, str):
        raise SystemExit("the reference refused the fixture's experts: " + got)
    layout, text, n_expert, offset = got
    (pack / "native_experts.txt").write_text(text, encoding="utf-8", newline="\n")
    with open(pack / "experts.bin", "wb") as fo:
        for _l, _gu, _d, _off, blob, ts in layout:
            chunk = np.concatenate([parsed.bytes(t.name).reshape(n_expert, -1) for t in ts], axis=1)
            assert chunk.shape == (n_expert, blob), (chunk.shape, n_expert, blob)
            fo.write(chunk.tobytes())
    sidecar = iq_pack.experts_source(parsed, text, offset)
    (pack / "experts.bin.src.json").write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")

    files = {}
    for f in sorted(pack.iterdir()):
        files[f.name] = {"sha256": hashlib.sha256(f.read_bytes()).hexdigest(), "bytes": f.stat().st_size}
    manifest = {"model": {"name": model.name, "sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
                          "bytes": model.stat().st_size},
                "pack": files}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser(description="record the reference packer's output for a synthetic model")
    ap.add_argument("--check", action="store_true", help="do not write; fail when the pack is out of date")
    a = ap.parse_args()
    if a.check:
        want = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
        got = build()
        if got != want:
            print("the reference packer's output changed: rerun tools/gen_pack_golden.py", file=sys.stderr)
            return 1
        print("the pack golden is current")
        return 0
    m = build()
    print(f"wrote {OUT}: model {m['model']['bytes']} B, pack files {sorted(m['pack'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
