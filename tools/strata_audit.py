"""tools/strata_audit.py - the upstream audit loop.

Strata changes almost every day.  This is how the audit keeps up without re-doing it by hand: see exactly what
changed since the commit the last audit covered, and run the same scanners and the same port analysis again - on
the new tree, in a throwaway git worktree, so the checkout you are working in is not touched.

The audit makes no network calls.  It compares the upstream ref already in this clone (default origin/main); to
pick up new upstream commits, run `git fetch origin` yourself first.

What it answers, in one page:

  - which files upstream changed (and how much Python was added - the port candidates);
  - whether a new outbound endpoint appeared (the phone-home question, on the delta only);
  - what the scanners find now, next to what they found last time, so a NEW finding is visible as a number
    that moved rather than a 300-line report to re-read.

    git fetch origin                             # yourself: the audit does not fetch
    python tools/strata_audit.py                 # report the delta, run the offline scanners
    python tools/strata_audit.py --cppcheck      # also the C++ static pass (about ten minutes)
    python tools/strata_audit.py --image         # also rebuild (no network) and scan the container image
    python tools/strata_audit.py --since <sha>   # audit from a commit other than the recorded one
    python tools/strata_audit.py --accept        # record upstream's tip as audited (read the report first)

State and reports live under sec-reports/: `state.json` holds the audited commit and the last run's counts,
`audit-<sha>/` holds one run's output.  The scanners are the ones docs/SECURITY_AUDIT.md names; a missing
tool is reported and skipped, never invented.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "sec-reports"
STATE = REPORTS / "state.json"

# What an outbound request looks like in the source.  One list, so the audit and the reader agree on what
# "phones home" means; docs/SECURITY_AUDIT.md walks through what each hit is for.
OUTBOUND = re.compile(
    r"""https?://[A-Za-z0-9._~:/?#\[\]@!$&'()*+,;=%-]+"""
    r"""|urllib\.request|urlopen\(|requests\.(?:get|post|Session)"""
    r"""|socket\.socket|http\.client|ftplib|smtplib"""
    r"""|pip install|curl\s|wget\s""")

HOST = re.compile(r"https?://([A-Za-z0-9._-]+)")


def run(cmd, cwd=None, timeout=1800):
    """(exit code, stdout, stderr).  A missing tool is a message, not a crash."""
    try:
        r = subprocess.run(cmd, cwd=str(cwd) if cwd else None, capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout, r.stderr
    except FileNotFoundError:
        return 127, "", f"{cmd[0]}: not found"
    except subprocess.TimeoutExpired:
        return 124, "", f"{cmd[0]}: timed out"


def git(*args, cwd=None) -> str:
    code, out, err = run(["git", *args], cwd=cwd or ROOT)
    if code != 0 and args[0] not in ("diff", "fetch"):
        sys.exit(f"git {' '.join(args)} failed:\n{err.strip()}")
    return out


def load_state() -> dict:
    try:
        return json.loads(STATE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_state(state: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(state, indent=1) + "\n", encoding="utf-8")


def changed_files(base: str, target: str) -> list[tuple[str, str]]:
    """[(status, path)] - A added, M modified, D deleted, R renamed."""
    out = []
    for line in git("diff", "--name-status", base, target, "--").splitlines():
        parts = line.split("\t")
        if len(parts) >= 2:
            out.append((parts[0][0], parts[-1]))
    return out


def python_added(base: str, target: str) -> list[tuple[int, int, str]]:
    """[(added, deleted, path)] for Python files, most added first - the port candidates.

    Test files are left out: a port keeps the reference implementation and its tests, so `test_*.py` moves in
    Python and only the module under it changes language (docs/CXX_PORT.md).
    """
    rows = []
    for line in git("diff", "--numstat", base, target, "--", "*.py").splitlines():
        parts = line.split("\t")
        if len(parts) == 3 and parts[0].isdigit() and not Path(parts[2]).name.startswith("test_"):
            rows.append((int(parts[0]), int(parts[1]), parts[2]))
    return sorted(rows, reverse=True)


def new_outbound(base: str, target: str) -> list[tuple[str, int, str]]:
    """[(path, line, text)] for outbound-shaped lines the delta ADDS - what the diff brings in, not the tree."""
    hits, path, line_no = [], "", 0
    for line in git("diff", "-U0", base, target, "--", "*.py", "*.sh", "*.ps1", "*.bat").splitlines():
        if line.startswith("+++ b/"):
            path, line_no = line[6:], 0
        elif line.startswith("@@"):
            m = re.search(r"\+(\d+)", line)
            line_no = int(m.group(1)) - 1 if m else 0
        elif line.startswith("+") and not line.startswith("+++"):
            line_no += 1
            text = line[1:].strip()
            if OUTBOUND.search(text):
                hits.append((path, line_no, text[:120]))
        elif not line.startswith("-"):
            line_no += 1
    return hits


def hosts_of(hits) -> list[str]:
    seen = []
    for _, _, text in hits:
        for h in HOST.findall(text):
            if h not in seen:
                seen.append(h)
    return seen


def _redact_gitleaks(path: Path) -> int:
    """Replace the values gitleaks MATCHED with a placeholder, in place.  -> findings redacted.

    A scanner that writes the secret it found into a file that is then committed has re-leaked it: the report
    outlives the fix, so cleaning the tree is not enough while the report still quotes the value.  That is
    exactly what happened here - `sec-reports/gitleaks.json` carried the leaked token verbatim, in two commits,
    so the reports were part of the leak rather than only the evidence of it.

    Only the matched text goes.  RuleID, File, StartLine, Commit and every count are kept, so the finding is
    still there to act on - the report stops being a copy of the secret without stopping being a finding.

    gitleaks is the one report that needs this: it is the only scanner whose output is a secret by definition.
    semgrep quotes source lines that are already in the tree and trivy reports packages, so neither creates a
    new copy of a value.
    """
    if not path.exists():
        return 0
    try:
        data = json.loads(path.read_text(encoding="utf-8") or "[]")
    except json.JSONDecodeError:
        return 0
    if not isinstance(data, list):
        return 0
    n = 0
    for finding in data:
        if not isinstance(finding, dict):
            continue
        for key in ("Secret", "Match"):
            if finding.get(key):
                finding[key] = "<redacted by the audit>"
                n += 1
    path.write_text(json.dumps(data, indent=1), encoding="utf-8")
    return n


def scan(tree: Path, report: Path, cppcheck: bool) -> dict:
    """Run the scanners over `tree`, writing their raw output into `report`.  -> counts per tool.

    Every tool runs with the worktree as its working directory, so what it sees is upstream's source; the
    output goes to an absolute path in `report` so nothing lands inside the worktree that is about to go.
    """
    report.mkdir(parents=True, exist_ok=True)
    counts, missing = {}, []

    def do(name, cmd, timeout=1200):
        if shutil.which(cmd[0]) is None:
            print(f"  {name}: not installed - skipped")
            counts[name], _ = None, missing.append(name)
            return None, ""
        code, out, err = run(cmd, cwd=tree, timeout=timeout)
        if code not in (0, 1):                 # 1 is bandit/gitleaks "findings are present", not a failure
            print(f"  {name}: exit {code} - {(err or out).strip().splitlines()[-1][:100] if (err or out).strip() else ''}")
        return code, out

    do("bandit", ["bandit", "-r", ".", "-x", "./third_party,./bench,./.git", "-f", "txt",
                  "-o", str(report / "bandit.txt")])
    counts["bandit"] = _count(report / "bandit.txt", r"^>> Issue:")

    # --no-git scans the TREE, not the history: deterministic, fast, and the same thing every run, so a count
    # that moved means something changed.  (The first pass over this repository scanned history and reported 44
    # hits; the tree scan reports more, because it also reads files the history walk skips.  Same rule, both.)
    do("gitleaks", ["gitleaks", "detect", "--source", ".", "--no-git", "--no-banner", "--report-format", "json",
                    "--report-path", str(report / "gitleaks.json"), "--exit-code", "0"])
    counts["gitleaks"] = _json_len(report / "gitleaks.json")
    _redact_gitleaks(report / "gitleaks.json")   # never write a secret value into a tracked report

    # semgrep's `--config auto` downloads its rules from the registry, so it runs only against a rules directory
    # already on disk, named by STRATA_SEMGREP_RULES.  Without one it is skipped, not run online.
    rules = os.environ.get("STRATA_SEMGREP_RULES", "")
    if rules and Path(rules).is_dir():
        do("semgrep", ["semgrep", "scan", "--config", rules, "--metrics=off", "--disable-version-check", "--json",
                       "-o", str(report / "semgrep.json"), "--exclude", "third_party", "--exclude", "bench"])
        counts["semgrep"] = _json_len(report / "semgrep.json", "results")
    else:
        print("  semgrep: no local rules (set STRATA_SEMGREP_RULES to a rules directory; the registry is online) - skipped")
        counts["semgrep"] = None

    # pip-audit asks PyPI's vulnerability service over the network, so it is not run here.  Run it by hand.
    print("  pip-audit: not run (it queries PyPI over the network; run it by hand)")
    counts["pip-audit"] = None

    if cppcheck:
        if shutil.which("cppcheck") is None:
            print("  cppcheck: not installed - skipped")
            counts["cppcheck"] = None
        else:
            print("  cppcheck: running (about ten minutes) ...")
            code, out, err = run(["cppcheck", "--enable=warning,style,performance,portability", "--inconclusive",
                                  "--inline-suppr", "--force", "-q", "--std=c++17", "-I", "include", "-I", "src",
                                  "src", "include", "tools/vision"], cwd=tree, timeout=3600)
            (report / "cppcheck.txt").write_text(err, encoding="utf-8")     # cppcheck prints to stderr
            counts["cppcheck"] = _count(report / "cppcheck.txt", r": (?:error|warning):")
            print(f"  cppcheck: {counts['cppcheck']} errors/warnings")

    for name in missing:
        counts[name] = None
    return counts


def _count(path: Path, pattern: str):
    try:
        return len(re.findall(pattern, path.read_text(encoding="utf-8", errors="replace"), re.M))
    except OSError:
        return None


def _json_len(path: Path, key=None):
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return len(d.get(key, [])) if key else len(d)


def port_ledger() -> dict:
    """{python path: (status, held_by)} from docs/port-ledger.tsv.

    The answer to "has this been ported, and what would have to be re-verified if upstream changed it".  A row
    with no `held_by` test is not a port: the acceptance is the reference's own output, so the audit can only
    name what to re-run when the ledger names a test.
    """
    out = {}
    path = ROOT / "docs" / "port-ledger.tsv"
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        f = line.split("\t")
        if len(f) < 4 or f[0] == "python":
            continue
        out[f[0]] = (f[3], f[2].strip())
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="audit what upstream changed, and scan it again")
    ap.add_argument("--ref", default="origin/main",
                    help="the upstream ref to audit, already fetched into this clone (default origin/main)")
    ap.add_argument("--since", help="audit from this commit instead of the recorded one")
    ap.add_argument("--cppcheck", action="store_true", help="also the C++ static pass (about ten minutes)")
    ap.add_argument("--image", action="store_true", help="also rebuild and scan the container image")
    ap.add_argument("--accept", action="store_true", help="record upstream's tip as audited and exit")
    ap.add_argument("--auto-accept", action="store_true",
                    help="move the baseline on by itself when the delta is empty: nothing added outbound, no "
                         "scanner count moved.  What a scheduled run uses - an unattended job should not sit on a "
                         "report nobody needs to read, and should not swallow one somebody does.")
    a = ap.parse_args()

    code, _, _ = run(["git", "rev-parse", "--verify", "--quiet", f"{a.ref}^{{commit}}"], cwd=ROOT)
    if code != 0:
        sys.exit(f"[strata] the upstream ref {a.ref!r} is not in this clone. This audit makes no network calls: "
                 "run `git fetch <remote>` yourself (for example `git fetch origin`), then run it again.")

    target = git("rev-parse", a.ref).strip()
    state = load_state()
    base = a.since or state.get("audited") or git("merge-base", "HEAD", a.ref).strip()

    if a.accept:
        state["audited"] = target
        save_state(state)
        print(f"[strata] audited commit recorded: {target[:12]}")
        return 0

    if base == target:
        print(f"[strata] nothing new: upstream is still {target[:12]}")
        return 0

    print(f"[strata] auditing {base[:12]} -> {target[:12]}")
    short = git("rev-parse", "--short", target).strip()
    report = REPORTS / f"audit-{short}"
    report.mkdir(parents=True, exist_ok=True)

    files = changed_files(base, target)
    added = python_added(base, target)
    hits = new_outbound(base, target)
    log = git("log", "--oneline", "--no-decorate", f"{base}..{target}")

    print(f"[strata] {len(files)} files changed, {sum(r[0] for r in added)} non-test Python lines added")
    print(f"[strata] {len(hits)} outbound-shaped lines added")
    _led = port_ledger()
    if added:
        _ported = sum(1 for _, _, p in added if _led.get(p, ("", ""))[0] == "ported")
        print(f"[strata] {_ported} of the changed Python files are ported: re-verify the test the ledger names "
              f"(docs/port-ledger.tsv)")

    # A worktree of the target: the scanners see upstream's tree, the checkout stays where it is.
    tmp = Path(tempfile.mkdtemp(prefix="strata-audit-"))
    tree = tmp / "tree"
    print("[strata] making a worktree of the target and scanning it ...")
    code, out, err = run(["git", "worktree", "add", "--detach", "--quiet", str(tree), target], cwd=ROOT)
    if code != 0:
        print(f"[strata] could not make the worktree: {err.strip()}")
        return 1
    try:
        counts = scan(tree, report, a.cppcheck)
    finally:
        run(["git", "worktree", "remove", "--force", str(tree)], cwd=ROOT)
        shutil.rmtree(tmp, ignore_errors=True)

    prev = state.get("last") or {}
    lines = [
        f"# Upstream audit: {base[:12]} -> {short}",
        "",
        f"Date: {datetime.date.today().isoformat()}",
        f"Upstream: `{git('remote', 'get-url', 'origin').strip()}` at `{a.ref}`",
        "",
        "## What changed",
        "",
        f"- {len(files)} files",
    ]
    for status in ("A", "M", "D"):
        n = sum(1 for s, _ in files if s == status)
        if n:
            lines.append(f"- {n} {'added' if status == 'A' else 'modified' if status == 'M' else 'deleted'}")
    lines += ["", "### Commits", "", "```"] + log.splitlines()[:40] + ["```"]

    lines += ["", "## Port candidates (non-test Python lines added; docs/CXX_PORT.md, docs/port-ledger.tsv)", ""]
    if added:
        led = port_ledger()
        lines += ["| added | file | ledger |", "| ---: | --- | --- |"]
        for n, _, p in added[:15]:
            status, held = led.get(p, ("not in the ledger", ""))
            if status == "ported":
                note = f"ported - re-verify `{held}`" if held else "ported, but the ledger names no test"
            else:
                note = status
            lines.append(f"| {n} | `{p}` | {note} |")
        ported = sum(1 for _, _, p in added if led.get(p, ("", ""))[0] == "ported")
        lines += ["", f"{sum(r[0] for r in added)} non-test Python lines added in {len(added)} files; "
                      f"{ported} of them already ported (re-verify the test the ledger names)."]
    else:
        lines.append("No non-test Python changed.")

    lines += ["", "## New outbound lines", ""]
    if hits:
        lines += [f"- `{p}:{n}` {t}" for p, n, t in hits[:40]]
        lines += ["", "Hosts: " + ", ".join(f"`{h}`" for h in hosts_of(hits))]
    else:
        lines.append("None. No new address, download or probe entered the tree.")

    lines += ["", "## Scanner counts (last run in brackets)", ""]
    for name in ("bandit", "gitleaks", "semgrep", "pip-audit", "cppcheck"):
        if name not in counts:
            continue
        now, was = counts[name], prev.get(name)
        move = "" if was is None or now is None else f" (was {was}; {'+' if now >= was else ''}{now - was})"
        lines.append(f"- {name}: {now}{move}")
    lines.append("- Not run here, because they fetch from the internet: pip-audit; semgrep unless "
                 "STRATA_SEMGREP_RULES names a local rules directory. Run them by hand.")

    if a.image:
        lines += ["", "## Container image", ""]
        tag = f"strata:audit-{short}"
        print("[strata] building the image (about fifteen minutes) ...")
        # no network in the build, and no pull: the base image must already be on this machine
        code, out, err = run(["docker", "build", "--pull=false", "--network", "none", "-t", tag,
                              "--build-arg", "CUDA_VERSION=12.9.1",
                              "--build-arg", "STRATA_CUDA_TOOLKIT=12", "--build-arg", "CUDA_ARCHITECTURES=70",
                              str(tree)], timeout=3600)
        lines.append(f"`docker build` exit {code}")
        if code == 0 and shutil.which("trivy"):
            # offline: the vulnerability DB must already be on disk (`trivy image --download-db-only`, by hand)
            run(["trivy", "image", "--scanners", "vuln,misconfig", "--severity", "HIGH,CRITICAL", "-f", "json",
                 "--offline-scan", "--skip-db-update", "--skip-java-db-update", "--skip-version-check",
                 "-o", str(report / "trivy-image.json"), tag], timeout=3600)
            n = _json_len(report / "trivy-image.json", "Results")
            lines.append(f"trivy: {n} result sets in `trivy-image.json`")

    lines += ["", "## Next", "",
              "1. Read the sections above; a moved number is the finding that matters.",
              "2. If a new outbound line is not explained by docs/SECURITY_AUDIT.md, add it there.",
              "3. New Python on the hot path: check it against docs/CXX_PORT.md.",
              "4. `python tools/strata_audit.py --accept` to make this commit the new baseline.", ""]

    (report / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (report / "delta.json").write_text(json.dumps(
        {"base": base, "target": target, "files": files, "python_added": added,
         "outbound": hits, "counts": counts}, indent=1) + "\n", encoding="utf-8")

    state.setdefault("runs", []).append({"target": target, "date": datetime.date.today().isoformat()})
    state["runs"] = state["runs"][-20:]
    state["last"] = {k: v for k, v in counts.items() if v is not None}
    if a.auto_accept:
        moved = any(prev.get(k) is not None and counts.get(k) != prev.get(k) for k in counts)
        calm = not hits and not moved
        if calm:
            state["audited"] = target
            print("[strata] nothing to look at: baseline moved to " + target[:12])
        else:
            print("[strata] the delta needs a look (new outbound line or a scanner count moved); "
                  "baseline left at " + base[:12])
    save_state(state)

    print(f"[strata] report: {report / 'REPORT.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
