"""tools/gen_cache_replay_golden.py - the reference's own cache-policy replay, over a synthetic routing trace.

`tools/replay_cache_policy.py` models `generate.cpp`'s adaptive tier (`adapt()`): usage counting per routed entry,
a round every `--every` windows, per layer the best non-resident against the worst resident, the best `--swaps`
over all layers, `usage *= --decay`.  This writes a trace that actually forms COMPLETE windows - the reference
keeps only windows with one group per layer, `len(w) == 48`, so a toy trace of a few records loads as zero
windows - runs the reference over it, and records what it printed.

    .venv/bin/python tools/gen_cache_replay_golden.py           # rewrite
    .venv/bin/python tools/gen_cache_replay_golden.py --check    # fail if the reference's output changed

The window GROUPING is recorded too, as the layers and ids of every group: it is the subtle part (a window is a
run of records whose layer rises, and consecutive records with one layer are one group), and checking only the
final rates would let a wrong grouping through.
"""
from __future__ import annotations

import argparse
import pathlib
import struct
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import replay_cache_policy as RP  # noqa: E402
import make_profile as MP  # noqa: E402

OUT = ROOT / "tests" / "eviction" / "cache_replay_golden.txt"
DIR = ROOT / "tests" / "eviction"

NL, NE = 48, 512
K = 2
N_WINDOWS = 6
SLOTS = 8        # the cache holds exactly this many experts: the profile's top `slots`, then one-for-one swaps

# The profile the run starts from.  Its experts are routed in the FIRST window only, so their usage stays at 1.0
# (below the engine's 2.0 candidate threshold) while they hold a slot: they are the victims the adaptive round
# swaps out.  A profile nothing routes would give `profile only, no adaptation: 0.0000` and a share of 0 in every
# configuration - a golden a port that prints zeros would pass.
PROFILE = [(l, 100 + l) for l in range(SLOTS)]
# Routed in EVERY window by layers >= SLOTS, so by the last window its usage is 6.0: a candidate.
HOT = lambda l: (l * 7) % NE            # noqa: E731
COLD = lambda l, w: (l * 13 + w) % NE   # noqa: E731


def records():
    """One record per (window, layer): (layer, [id0, id1])."""
    out = []
    for w in range(N_WINDOWS):
        for layer in range(NL):
            if w == 0 and layer < SLOTS:
                out.append((layer, [100 + layer, COLD(layer, w)]))
            else:
                out.append((layer, [HOT(layer), COLD(layer, w)]))
    return out


def trace_bytes(recs) -> bytes:
    blob = bytearray()
    for layer, ids in recs:
        assert len(ids) == K
        blob += struct.pack("<ii", layer, K)
        blob += struct.pack("<%di" % K, *ids)
        blob += struct.pack("<%df" % K, *([0.5] * K))    # the weights: the reference reads the ids only
    return bytes(blob)


# (name, argv) - parameter sets.  --every 2 fires the round at windows 2, 4 and 6; --every 1 at every window.
# The defaults are the reference's own, and --slots stays small so the cache actually fills and swaps.
CONFIGS = [
    ("every2", ["--every", "2"]),
    ("every1", ["--every", "1"]),
]


def build() -> str:
    DIR.mkdir(parents=True, exist_ok=True)
    trace = DIR / "replay-trace.bin"
    trace.write_bytes(trace_bytes(records()))
    # its own profile, not the shared base.bin: this fixture needs a profile whose experts the trace routes
    profile = DIR / "replay-profile.bin"
    MP.write_profile(profile, PROFILE)

    lines = ["# cache-replay golden: tools/replay_cache_policy.py over a synthetic trace that forms complete windows",
             "# the trace record: <int32 layer><int32 k><k int32 ids><k float32 weights>",
             "trace replay-trace.bin", "profile replay-profile.bin", f"n_layers {NL}", f"n_expert {NE}",
             f"k {K}", f"n_windows {N_WINDOWS}", f"slots {SLOTS}"]
    # the grouping, as the reference loaded it: the subtle part, and the only way to catch a wrong window
    windows, k = RP.load_trace(str(trace))
    lines.append(f"loaded_windows {len(windows)}")
    lines.append(f"loaded_k {k}")
    for wi, w in enumerate(windows):
        groups = " ".join(f"{l}:" + ",".join(str(int(x)) for x in ids.ravel()) for l, ids in w)
        lines.append(f"window {wi} {groups}")

    for name, extra in CONFIGS:
        out = _run([str(trace), str(profile), str(SLOTS), *extra])
        for line in out.split("\n"):
            lines.append(f"stdout {name} {line}")
    return "\n".join(lines) + "\n"


def _run(argv) -> str:
    """The reference as the CLI it is: its argv is TRACE PROFILE SLOTS [flags], so a subprocess is the tool."""
    r = subprocess.run([sys.executable, str(ROOT / "tools" / "replay_cache_policy.py"), *argv],
                       cwd=DIR, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"replay_cache_policy refused {argv}: {r.stderr}")
    return r.stdout.rstrip("\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="do not write; fail when the golden is out of date")
    a = ap.parse_args()
    text = build()
    if a.check:
        if not OUT.exists() or OUT.read_text(encoding="utf-8") != text:
            raise SystemExit("the cache-replay golden is out of date - rerun without --check")
        print("the cache-replay golden is current")
        return 0
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)} ({len(CONFIGS)} parameter sets, {N_WINDOWS} windows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
