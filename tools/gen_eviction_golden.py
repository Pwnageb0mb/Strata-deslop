"""tools/gen_eviction_golden.py - the reference's own routing-trace reading and profile ranking, for fixtures.

`tools/make_profile.py` reads `--dump-routing` traces, ranks every (layer, expert) pair (the base's order, then the
traces' pairs most frequent first, then the fill) and writes a profile.  `tools/bench_eviction.py` replays the same
traces against candidate eviction policies.  Both walk the same trace record format, and the C++ port has ONE
reader for it - so this records what the reference's reader produces (the frequencies AND the order, which the two
tools each compute separately) and what its ranking produces for every mode.

    .venv/bin/python tools/gen_eviction_golden.py           # rewrite
    .venv/bin/python tools/gen_eviction_golden.py --check    # fail if the reference's output changed

The fixtures are committed beside the golden: the two traces (tiny, and deliberately awkward - out-of-range ids,
the maximum layer and expert, repeated pairs that force a tie) and the base profile.  The ranked output is 24,576
pairs, so it is recorded as the reference's own profile BYTES (a committed binary per mode) rather than as text.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import pathlib
import struct
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import bench_eviction as BE  # noqa: E402
import make_profile as MP  # noqa: E402

OUT = ROOT / "tests" / "eviction" / "trace_golden.txt"
DIR = ROOT / "tests" / "eviction"

# (layer, k, ids, weights) records.  Out-of-range ids are what the reader skips; the repeats and the equal counts
# are what force the tie-break.  A record whose k does not match its ids is a reader bug, not a fixture.
RECORDS = [
    (0, 2, [3, 7]),
    (0, 2, [7, 3]),
    (1, 3, [0, 1, 2]),
    (1, 1, [5]),
    (2, 2, [511, 0]),
    (47, 1, [1]),
    (48, 1, [1]),          # layer 48: outside 0..47, skipped
    (0, 2, [-1, 3]),       # expert -1: skipped, the 3 still counts
    (5, 2, [2, 512]),      # expert 512: outside 0..511, skipped
    (2, 1, [0]),
]
WEIGHTS = [0.5, 0.25, 0.125, 0.0625, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
BASE_RANKED = [(10, 10), (10, 11), (11, 5)]      # a base that ranks only a few pairs, so the fill has work to do
MODES = ["default", "reorder", "nobase"]


def trace_bytes(records) -> bytes:
    out = bytearray()
    for i, (layer, k, ids) in enumerate(records):
        assert len(ids) == k, f"record {i}: k={k} but {len(ids)} ids"
        out += struct.pack("<ii", layer, k)
        out += struct.pack("<%di" % k, *ids)
        out += struct.pack("<%df" % k, *([WEIGHTS[i]] * k))
    return bytes(out)


def build() -> str:
    DIR.mkdir(parents=True, exist_ok=True)
    lines = ["# eviction golden: tools/make_profile.py + tools/bench_eviction.py over committed fixtures",
             "# the trace record: <int32 layer><int32 k><k int32 ids><k float32 weights>",
             f"n_layers {MP.N_LAYER}", f"n_expert {MP.N_EXPERT}"]

    traces = []
    for i, records in enumerate((RECORDS[:6], RECORDS[6:]), start=1):
        path = DIR / f"trace-{i}.bin"
        path.write_bytes(trace_bytes(records))
        traces.append(path)
        lines.append(f"trace trace-{i}.bin")   # the fixture is committed beside the golden; the test reads it

    base = DIR / "base.bin"
    MP.write_profile(base, BASE_RANKED)
    lines.append("base base.bin")
    lines.append("base_ranked " + " ".join(f"{l},{e}" for l, e in BASE_RANKED))

    # the reference's reader, from BOTH tools: make_profile aggregates to frequencies, bench_eviction keeps the
    # order.  The port derives the frequencies from the order, so both have to be recorded to prove that is equal.
    freq = {}
    for t in traces:
        for p, c in MP.read_trace(t).items():
            freq[p] = freq.get(p, 0) + c
    for (l, e) in sorted(freq):
        lines.append(f"freq {l} {e} {freq[(l, e)]}")
    codes = []
    for t in traces:
        codes.extend(BE.read_trace(t))
    lines.append("codes " + " ".join(f"{BE.decode(c)[0]},{BE.decode(c)[1]}" for c in codes))

    # the ranking, every mode.  The pairs come back from the file the reference wrote, so the comparison is
    # against the reference's own bytes and not against a reading of its rank_profile.
    for mode in MODES:
        out = DIR / f"profile-{mode}.bin"
        MP.write_profile(out, MP.rank_profile(
            BASE_RANKED, freq, no_base=(mode == "nobase"), reorder=(mode == "reorder"))[0])
        blob = out.read_bytes()
        ranked = MP.read_profile(out)
        lines.append(f"profile {mode} bytes {len(blob)} sha256 {hashlib.sha256(blob).hexdigest()}")
        # the head of the ranking is what the modes actually change; the tail is the fill in both cases
        lines.append(f"profile_head {mode} " + " ".join(f"{l},{e}" for l, e in ranked[:12]))
        lines.append(f"profile_tail {mode} " + " ".join(f"{l},{e}" for l, e in ranked[-4:]))
        lines.append(f"profile_count {mode} {len(ranked)}")

    # ---- the eviction benchmark's own report and CSV.  Two configurations, between them covering every policy,
    # a multi-window split, both scopes, a prefill, and two slot counts at once.
    base_pairs = MP.read_profile(base)
    cfgs = [
        # TIGHT slots: 10 distinct pairs against 4 and 8 slots, so the policies actually evict and diverge.
        # Without this the fixture fits every slot count and all four policies report the same rate, which is a
        # golden a wrong port would pass.
        ("global-tight", dict(slots_list=[4, 8], per_layer=False,
                              policy_names=["fill-only", "lru", "lfu-decay", "oracle"], windows=4,
                              decay_window=65536, prefill_pairs=None), "global"),
        # per-layer with slots >= the layer count.  NOT below it: q = slots // n_layers is 0 there, every layer
        # but the last has no capacity at all, and the reference CRASHES on that (Lru pops an empty OrderedDict,
        # KeyError: 'dictionary is empty') - so there is no reference output to compare against.  The port refuses
        # the configuration instead; see eviction_config_error.
        ("perlayer", dict(slots_list=[4105, 96], per_layer=True,
                          policy_names=["fill-only", "lru", "lfu-decay", "oracle"], windows=1,
                          decay_window=65536, prefill_pairs=None), "per-layer"),
        # the profile-seated start the engine actually uses, and a small decay window so the increment doubles
        ("prefill-decay", dict(slots_list=[4105], per_layer=False,
                               policy_names=["fill-only", "lru", "lfu-decay"], windows=1, decay_window=4,
                               prefill_pairs=base_pairs), "global"),
    ]
    lines.append("summary %s lookups, %s distinct (layer, expert) pairs"
                 % (format(len(codes), ","), format(len(set(codes)), ",")))

    # ---- make_profile_mix: the weighted-mix profile, run as the reference CLI it is (its ranking is inline in
    # main()), so the comparison is against the tool's own output rather than a reading of its source.  The specs
    # cover a weight, a ZERO weight (a pair present with score 0 is ranked, not filled) and an EMPTY trace, which
    # the reference skips.  BEFORE the eviction blocks: a block runs to the next marker, so anything after the
    # first one would be swallowed, and expert_profile_test stops reading at that marker.
    (DIR / "trace-empty.bin").write_bytes(b"")
    mixes = [("weighted", ["trace-1.bin:2.0", "trace-2.bin:0", "trace-empty.bin"]),
             ("plain", ["trace-1.bin", "trace-2.bin"])]
    for name, specs in mixes:
        mix_out = DIR / f"mix-{name}.bin"
        r = subprocess.run([sys.executable, str(ROOT / "tools" / "make_profile_mix.py"), *specs,
                            "--out", str(mix_out)], cwd=DIR, capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit(f"make_profile_mix refused {name}: {r.stderr}")
        blob = mix_out.read_bytes()
        ranked = MP.read_profile(mix_out)
        lines.append(f"mix {name} specs {' '.join(specs)} bytes {len(blob)} "
                     f"sha256 {hashlib.sha256(blob).hexdigest()} count {len(ranked)}")
        lines.append(f"mix_head {name} " + " ".join(f"{l},{e}" for l, e in ranked[:12]))
        lines.append(f"mix_tail {name} " + " ".join(f"{l},{e}" for l, e in ranked[-4:]))
        skipped = 0
        for line in r.stdout.strip().split("\n"):
            if line.endswith("empty trace, skipped"):
                skipped += 1
            elif line.startswith("wrote ") and ": " in line:
                # the path is machine-specific; the counts after it are the assertion
                lines.append(f"mix_stdout {name} " + line.split(": ", 1)[1])
        lines.append(f"mix_skipped {name} {skipped}")
    # ALL the declarations first, then the blocks: a block runs until the next marker, so a configuration declared
    # after another's CSV would be swallowed by that CSV and never parsed.
    for name, kw, _scope in cfgs:
        for key, val in (("slots", ",".join(str(s) for s in kw["slots_list"])),
                         ("per_layer", "1" if kw["per_layer"] else "0"),
                         ("policies", ",".join(kw["policy_names"])),
                         ("windows", str(kw["windows"])),
                         ("decay_window", str(kw["decay_window"])),
                         ("prefill", "1" if kw["prefill_pairs"] else "0")):
            lines.append(f"cfg {name} {key} {val}")
    for name, kw, scope in cfgs:
        rows = BE.run(codes, MP.N_LAYER, MP.N_EXPERT, **kw)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            BE.report(rows, kw["windows"])
        csv_path = DIR / f"evict-{name}.csv"
        BE.write_csv(csv_path, rows, scope)
        lines.append(f"# report-begin {name}")
        lines.append(out.getvalue().rstrip("\n"))
        lines.append(f"# csv-begin {name}")
        lines.append(csv_path.read_text().rstrip("\n"))
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="do not write; fail when the golden is out of date")
    a = ap.parse_args()
    text = build()
    if a.check:
        if not OUT.exists() or OUT.read_text(encoding="utf-8") != text:
            raise SystemExit("the eviction golden is out of date - rerun without --check")
        print("the eviction golden is current")
        return 0
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)} ({len(RECORDS)} records, {len(MODES)} modes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
