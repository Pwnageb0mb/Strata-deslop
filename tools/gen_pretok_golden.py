"""tools/gen_pretok_golden.py - the pre-tokenizer's expected pieces, recorded from the reference.

`tools/strata_tokenizer.py` splits with `QWEN35_PATTERN` through Python's `regex` module.  The C++ port has to
produce the SAME pieces, and the only honest way to know it does is to compare against that module - so this
writes the module's own `findall` output to a golden file the C++ test reads.

    .venv/bin/python tools/gen_pretok_golden.py            # rewrite tests/tokenizer/pretok_golden.txt
    .venv/bin/python tools/gen_pretok_golden.py --check    # fail if it is stale (CI)

The file is one case a line: the number of pieces, a TAB, the text as hex, then a TAB and each piece as hex.  Hex
because the corpus is deliberately full of the things that break an escaping scheme - CR, LF, NUL, combining
marks, emoji - and because C++ can read a hex line without a JSON parser.  The count leads the line so that no
case can be an empty line: the empty text has no pieces, and a reader that skips blank lines would drop it.

The corpus is chosen to hit every branch of the pattern, including the ones where the first successful
alternative is not the longest match: contractions, a space absorbed into a following word, a whitespace run
with and without a newline in it, whitespace followed by a non-space (where the branch matches all but the last
character), digits alone, punctuation runs, combining marks, and text that is only whitespace.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import strata_tokenizer as ST  # noqa: E402
import regex  # noqa: E402

OUT = ROOT / "tests" / "tokenizer" / "pretok_golden.txt"
PAT = regex.compile(ST.QWEN35_PATTERN)

CORPUS = [
    "",
    "a",
    "A",
    "3",
    "hello world",
    " the",
    "  two",
    "a  b",
    "a   b",
    "a   ",
    "   a",
    "  ",
    "\t",
    "\n",
    "\r\n",
    "\r\n\r\n",
    " \n ",
    "  \n  \n  ",
    "a\n\nb",
    "a\r\n\r\nb",
    "line one\nline two\r\nline three",
    "don't",
    "it's",
    "we're",
    "I've",
    "I'm",
    "we'll",
    "he'd",
    "'s",
    "'ll",
    "'le",
    "'L",
    "'",
    "don't stop",
    "Café",
    "日本語のテキスト",
    "🙂",
    "🙂🙂",
    "e\u0301",
    "n\u0303o",
    "e\u0301e\u0301",
    "_j",
    "x_",
    "_",
    "a_b_c",
    "!@#$%^&*()",
    "!! ?? !!!",
    " - ",
    "a-b",
    "a - b",
    "3.14",
    "12ab",
    "ab12",
    "123",
    "\u00a0",
    "a\u00a0b",
    "a\u2003b",
    "\u2028",
    "«»…",
    "…",
    "\x00",
    "a\x00b",
    "\x01\x02\x03",
    "a\u0301 \u0301",
    "\u0301",
    " \u0301",
    "x" * 70,
    "def f(x):\n    return x  # trailing\n",
    "mixed: café 日本語 🙂 123 !!! \t\n",
    "  leading and trailing  \n",
    "tab\tseparated\twords",
    "a\tb",
]


def hexs(s: str) -> str:
    return s.encode("utf-8").hex()


def build() -> str:
    lines = ["# pieces<TAB>text_hex<TAB>piece_hex...   "
             "(tools/strata_tokenizer.py, QWEN35_PATTERN, via Python regex)"]
    for text in CORPUS:
        pieces = PAT.findall(text)
        # the module must agree with itself: the pieces rebuild the text exactly
        assert "".join(pieces) == text, f"reference did not cover {text!r}: {pieces!r}"
        # the count leads the line so no case can be an empty line: the empty text has no pieces, and a line
        # that is only whitespace would otherwise be skipped by a reader (it was, once)
        lines.append(f"{len(pieces)}\t{hexs(text)}" + "".join("\t" + hexs(p) for p in pieces))
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="record the pre-tokenizer's pieces from the reference")
    ap.add_argument("--check", action="store_true", help="do not write; fail when the golden is out of date")
    a = ap.parse_args()
    text = build()
    cases = len(CORPUS)
    if a.check:
        if not OUT.exists() or OUT.read_text(encoding="utf-8") != text:
            print(f"{OUT} is out of date: run tools/gen_pretok_golden.py", file=sys.stderr)
            return 1
        print(f"{OUT} is current ({cases} cases)")
        return 0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT}: {cases} cases")
    return 0


if __name__ == "__main__":
    sys.exit(main())
