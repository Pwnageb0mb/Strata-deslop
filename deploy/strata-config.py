#!/usr/bin/env python3
"""deploy/strata-config.py - reconcile the install config in the volume with what the conf declares.

Run inside the container, because the /data volume is not readable from the host:

    docker run --rm -i --entrypoint /opt/strata/.venv/bin/python \\
        -v strata-data:/data -v deploy:/deploy:ro \\
        strata:v100 /deploy/strata-config.py <tag> "<flags>" -

Why this exists: docker-entrypoint.sh builds its own argument list for setup.py (`set --`, then
`exec setup.py "$@"`) and discards whatever `docker run` was given; on a later start it forwards only the
card and layer-split settings.  So neither an engine flag nor the API key can be handed to the container
from outside.  setup.py writes both into the install config - the install-derived values on a setup pass,
and the ones it is given on the command line under its `keep` - and the engine and serve/server.py read
them from there.  That config is therefore the one place either can be set from outside, and this tool is
what sets them.

Two keys are reconciled, each only when it actually differs, so a run with nothing to do rewrites nothing:

  * "args" - the engine flags.  The flags this tool last added are recorded in the config itself, under
    "strata_engine_args" (a key setup.py's carry_over keeps across a setup pass); every run removes those,
    plus any flag named in the new list, then appends the new list.  Removing a flag from ENGINE_ARGS
    therefore removes it from the config, and a setup pass that drops hand-added options (setup.py's
    `args_dropped`) is repaired on the next start.

  * "api_key" - the key serve/server.py enforces on every route but /health.  The key is read from stdin
    (fourth argument "-") or from a file, never from an argument or an environment variable, so the value
    never appears on a command line or in `docker inspect`.  Reading it from the host file directly does
    not work: that file is mode 0600 and the container does not run as its owner.  Given no fourth
    argument the key is left alone; given one, its value (whitespace stripped) replaces the config's.  The
    value is never printed - only its length.

Exit codes: 0 = done (changed or not), 1 = the config is unusable, 2 = bad usage,
3 = the volume has no install config yet (setup.py writes one on this start).
"""
from __future__ import annotations

import json
import pathlib
import sys

MARKER = "strata_engine_args"
SAVED = "strata_engine_args_saved"   # the install's own value of each flag ENGINE_ARGS overrides


def strip(args: list[str], names: set[str]) -> list[str]:
    """`args` without any flag in `names` and the value that follows it (setup.py's own convention:
    a token after a flag is its value unless it is itself a flag)."""
    out: list[str] = []
    i = 0
    while i < len(args):
        if args[i] in names:
            i += 2 if i + 1 < len(args) and not args[i + 1].startswith("--") else 1
        else:
            out.append(args[i])
            i += 1
    return out


def take(args: list[str], name: str) -> list[str]:
    """The tokens of flag `name` in `args` (the flag and its value, by strip's convention); [] when absent."""
    for i, a in enumerate(args):
        if a == name:
            return args[i:i + 2] if i + 1 < len(args) and not args[i + 1].startswith("--") else [a]
    return []


def read_key(path: str) -> str | None:
    """The API key from `path`, whitespace stripped, or None when it is missing or holds nothing.
    `path` of "-" reads stdin, which is how the start script passes it: the host file is mode 0600 and
    the container does not run as its owner, and stdin keeps the value off the command line and out of
    the environment."""
    try:
        key = sys.stdin.read() if path == "-" else pathlib.Path(path).read_text(encoding="utf-8")
    except OSError as e:
        print("strata-config: cannot read the API key (%s)" % e, file=sys.stderr)
        return None
    return key.strip() or None


def main() -> int:
    if len(sys.argv) not in (3, 4):
        print('usage: strata-config.py <tag> "<flags>" [<api-key-file>|-]', file=sys.stderr)
        return 2
    tag, extra = sys.argv[1], sys.argv[2].split()
    key = read_key(sys.argv[3]) if len(sys.argv) == 4 else None

    path = pathlib.Path("/data/config/strata-%s.json" % tag)
    if not path.is_file():
        return 3
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(cfg.get("args"), list):
        print("strata-config: %s has no \"args\" list" % path, file=sys.stderr)
        return 1

    args = [str(a) for a in cfg["args"]]
    managed = cfg.get(MARKER) if isinstance(cfg.get(MARKER), list) else []
    saved = cfg.get(SAVED) if isinstance(cfg.get(SAVED), dict) else {}
    want = [a for a in extra if a.startswith("--")]
    # The install's own args: without the flags this tool added, and with the values those flags replaced put
    # back - so dropping a flag from ENGINE_ARGS restores what setup.py wrote (e.g. --prefill auto) instead of
    # leaving the flag out altogether.
    base = strip(args, set(managed))
    for flag in managed:
        base += [str(t) for t in saved.get(flag, [])]
    keep = {f: take(base, f) for f in want if take(base, f)}
    keep.update({f: v for f, v in saved.items() if f in want})
    new = strip(base, set(want)) + extra

    changed = []
    if new != args or want != managed or keep != saved:
        cfg["args"] = new
        if want:
            cfg[MARKER] = want
        else:
            cfg.pop(MARKER, None)
        if keep:
            cfg[SAVED] = keep
        else:
            cfg.pop(SAVED, None)
        changed.append("%d -> %d args (%s)" % (len(args), len(new), " ".join(extra) or "none"))
    if key is not None and cfg.get("api_key") != key:
        cfg["api_key"] = key
        changed.append("api key replaced (%d chars)" % len(key))

    if not changed:
        return 0
    tmp = path.parent / (path.name + ".tmp")
    tmp.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    tmp.replace(path)
    print("strata-config: " + "; ".join(changed))
    return 0


if __name__ == "__main__":
    sys.exit(main())
