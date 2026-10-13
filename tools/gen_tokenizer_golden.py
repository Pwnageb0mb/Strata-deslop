"""tools/gen_tokenizer_golden.py - the tokenizer's expected ids, from the reference.

`tools/strata_tokenizer.py` is the reference.  The C++ port has to produce the same ids for the same text and
the same vocabulary, and the honest way to know is to ask the reference - so this records its answers.

    .venv/bin/python tools/gen_tokenizer_golden.py            # rewrite tests/tokenizer/tokenizer_golden.txt
    .venv/bin/python tools/gen_tokenizer_golden.py --check    # fail if it is stale (CI)

A real vocabulary is 248,320 tokens, which is not something to carry in a test file.  The vocabs here are
SYNTHETIC and small - the same trick the reference's own test uses (`tools/test_strata_tokenizer.py`'s `toy()`):
256 byte tokens, so any UTF-8 text is representable, plus a handful of merges.  That is enough to pin the
things a port gets wrong: which byte maps to which token, that merges apply by RANK and not left to right, that
a merge chain reaches its longest form, and that specials are matched as literals - type 4 always, type 3 only
with parse_special.

The file:

    V <vocab_id> <token_count> <merge_count>
    T <token_hex> <token_type>   (token_count lines; type 3 CONTROL, 4 USER_DEFINED, else 0)
    M <merge_hex>                (merge_count lines)
    X <vocab_id> <parse_special> <text_hex> <id,id,...> <decoded_hex>

Fields are TAB-separated, not space-separated: the empty text's hex is an empty field, and a reader that splits
on whitespace loses the alignment and reads the ids as the text.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import strata_tokenizer as ST  # noqa: E402

OUT = ROOT / "tests" / "tokenizer" / "tokenizer_golden.txt"
B = ST.BYTE_TO_UNICODE
SP = B[32]                                              # the byte-level spelling of a space


def byte_vocab() -> list[str]:
    return [B[b] for b in range(256)]


def vocab_toy() -> tuple[list[str], list[str]]:
    """The reference test's own toy: 256 byte tokens and ten merges."""
    merges = ["t h", "th e", "i n", "a n", "an d", f"{SP} t", f"{SP}t h", f"{SP}th e", "e r", "o n"]
    merges = [m.replace("\u0120", SP) for m in merges]
    tokens = byte_vocab() + [m.split(" ")[0] + m.split(" ")[1] for m in merges]
    return tokens, merges


def vocab_chain() -> tuple[list[str], list[str]]:
    """A vocabulary where a rank order and a left-to-right order disagree, plus a chain that only forms if the
    lowest rank is applied first: "a b c" with "b c" ranked before "a b"."""
    merges = ["b c", "a bc", "x y", "xy z", "h i", "hi j"]
    tokens = byte_vocab()
    for m in merges:
        a, b = m.split(" ")
        tokens.append(a + b)
    return tokens, merges


def vocab_specials() -> tuple[list[str], list[str], list[int]]:
    """A vocabulary with the two kinds of literal: a USER_DEFINED one (matched always) and a CONTROL one
    (matched only with parse_special), plus a near-miss that must stay ordinary text."""
    tokens, merges = vocab_toy()
    tokens += ["\u0120hold", "hold"]
    types = [0] * len(tokens)
    tokens += ["<|im_start|>", "<|im_end|>", "<tool_response>"]
    types += [3, 3, 4]
    return tokens, merges, types


VOCABS = {
    1: ("toy", vocab_toy()),
    2: ("chain", vocab_chain()),
    3: ("specials", vocab_specials()),
}

TEXTS = [
    "",
    "the",
    "and",
    "on",
    "there",
    "other",
    "in",
    "an",
    "a",
    "abc",
    "xyz",
    "hij",
    "hi j",
    "café",
    "日本語",
    "🙂",
    "e\u0301",
    "the café 日本語 🙂\n",
    "  ",
    "\n\n",
    "the and other there",
    "x" * 70,
    "x" * 64,
    "x" * 65,
    "x" * 300,
    "日" * 40,
    "日本語" * 40,
    "the" * 60,
    "and" * 60,
    ("the and in on " * 20),
    "café" * 40,
    "hold \u0120hold",
    "<|im_start|>hello<|im_end|>",
    "<tool_response>ok",
    "<|im_star",
    "a<tool_response>b",
    "\x00\x01\x02",
    "the\u00a0and",
]


def hexs(s: str) -> str:
    return s.encode("utf-8").hex()


def build() -> str:
    lines = ["# synthetic vocabularies and the reference's answers (tools/strata_tokenizer.py)"]
    made = {}
    for vid, (name, spec) in VOCABS.items():
        tokens, merges = spec[0], spec[1]
        types = spec[2] if len(spec) > 2 else [0] * len(tokens)
        lines.append(f"V\t{vid}\t{len(tokens)}\t{len(merges)}")
        lines += [f"T\t{hexs(t)}\t{types[i]}" for i, t in enumerate(tokens)]
        lines += [f"M\t{hexs(m)}" for m in merges]
        tok = ST.Tokenizer(tokens, merges, types)
        made[vid] = tok
    for vid, tok in made.items():
        for text in TEXTS:
            for parse_special in ((False, True) if vid == 3 else (False,)):
                ids = tok.encode(text, parse_special=parse_special)
                # the reference must round-trip its own answer on a byte-level vocabulary
                assert tok.decode(ids) == text, f"reference round trip failed on {text!r}: {ids!r}"
                lines.append(f"X\t{vid}\t{1 if parse_special else 0}\t{hexs(text)}\t"
                             f"{','.join(str(i) for i in ids)}\t{hexs(text)}")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="record the tokenizer's ids from the reference")
    ap.add_argument("--check", action="store_true", help="do not write; fail when the golden is out of date")
    a = ap.parse_args()
    text = build()
    if a.check:
        if not OUT.exists() or OUT.read_text(encoding="utf-8") != text:
            print(f"{OUT} is out of date: run tools/gen_tokenizer_golden.py", file=sys.stderr)
            return 1
        print(f"{OUT} is current")
        return 0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT}: {len(VOCABS)} vocabularies, "
          f"{sum(1 for l in text.splitlines() if l.startswith('X\t'))} cases")
    return 0


if __name__ == "__main__":
    sys.exit(main())
