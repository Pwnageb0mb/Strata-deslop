"""tools/gen_draft_vocab_golden.py - the reference's own draft-vocabulary builder, over a synthetic vocabulary.

`tools/draft_vocab.py` picks the vocabulary rows the MTP draft head may propose: whole scripts (`--add`), and the
most frequent tokens of a text corpus that cover `--coverage` of its occurrences (`--corpus`).  This writes a GGUF
carrying a byte-level BPE vocabulary - the 256 byte tokens, a handful of merges, and tokens spelling Han, Kana,
Hangul, CJK-punctuation and Cyrillic characters - plus a base subset and a corpus, runs the reference over them,
and records its stdout and the subset it wrote.

    .venv/bin/python tools/gen_draft_vocab_golden.py           # rewrite
    .venv/bin/python tools/gen_draft_vocab_golden.py --check    # fail if the reference's output changed

The vocabulary is deliberately awkward: it carries a token whose bytes are a PREFIX of a multi-byte character, so
the reference's `token_text` returns None for it (its bytes are not whole UTF-8 characters), and a control token, so
`token_type` is exercised rather than assumed to be all-normal.
"""
from __future__ import annotations

import argparse
import hashlib
import pathlib
import struct
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import strata_tokenizer as ST  # noqa: E402
from gen_tokenizer_golden import byte_vocab  # noqa: E402
from gguf_writer import GGUFWriter  # noqa: E402

OUT = ROOT / "tests" / "tokenizer" / "draft_vocab_golden.txt"
DIR = ROOT / "tests" / "tokenizer"
B = ST.BYTE_TO_UNICODE


def spell(s: str) -> str:
    """The byte-level spelling of a text: one vocabulary character per UTF-8 byte, which is what BPE stores."""
    return "".join(B[b] for b in s.encode("utf-8"))


def spell_bytes(bs) -> str:
    return "".join(B[b] for b in bs)


# The characters the fixture must be able to SEE a script in.  Each is added as its own token, spelled
# byte-level, so token_bytes(id) gives whole UTF-8 characters back and the script ranges match.
HAN = "中文漢字"
KANA = "あア"
HANGUL = "한국"
CJK_PUNCT = "，。"
CYRILLIC = "ЖЯ"


def vocabulary():
    tokens = byte_vocab()
    merges = [f"{B[0x20]} {B[0x74]}"]                       # a merge, so encode has something to apply
    for m in merges:
        a, b = m.split(" ")
        if a + b in tokens:
            continue
        tokens.append(a + b)
    for text in (HAN, KANA, HANGUL, CJK_PUNCT, CYRILLIC):
        for ch in text:
            t = spell(ch)
            if t not in tokens:
                tokens.append(t)
    # a token whose bytes are the FIRST TWO bytes of 中: not a whole UTF-8 character, so token_text gives None
    partial = spell_bytes([0xE4, 0xB8])
    if partial not in tokens:
        tokens.append(partial)
    types = [1] * len(tokens)          # 1 = NORMAL
    tokens.append("<|im_end|>")        # a CONTROL literal, so token_type is exercised
    types.append(3)
    return tokens, merges, types


def write_gguf(path, tokens, merges, types):
    w = GGUFWriter()
    # no type names: the writer infers `string` for a str and `array:<elem>` for a list
    w.add("general.architecture", "qwen4exp")
    w.add("tokenizer.ggml.model", "gpt2")
    w.add("tokenizer.ggml.pre", "qwen35")
    w.add("tokenizer.ggml.tokens", tokens)
    w.add("tokenizer.ggml.merges", merges)
    w.add("tokenizer.ggml.token_type", types)
    w.write(path)


BASE_IDS = list(range(0, 40))          # a base subset of the low byte tokens, so `outside` has something to miss
CORPUS = "中文中文中文 あア ЖЯ 한국 ，。 the 中文 あ Ж"
CONFIGS = [
    ("add-cjk", ["--add", "cjk"]),
    ("add-cyrillic", ["--add", "cyrillic"]),
    ("corpus", ["--corpus", "corpus.txt", "--coverage", "0.9"]),
    ("stats-only", ["--stats"]),
]


def build() -> str:
    DIR.mkdir(parents=True, exist_ok=True)
    tokens, merges, types = vocabulary()
    gguf = DIR / "draft-vocab.gguf"
    write_gguf(gguf, tokens, merges, types)
    base = DIR / "draft-base.bin"
    base.write_bytes(struct.pack("<%di" % len(BASE_IDS), *BASE_IDS))
    (DIR / "corpus.txt").write_text(CORPUS, encoding="utf-8")

    lines = ["# draft-vocab golden: tools/draft_vocab.py over a synthetic byte-level BPE vocabulary",
             f"gguf draft-vocab.gguf", f"base draft-base.bin", f"corpus corpus.txt",
             f"vocab {len(tokens)}", f"base_ids {len(BASE_IDS)}"]
    for name, extra in CONFIGS:
        out = DIR / f"draft-{name}.bin"
        argv = ["--gguf", str(gguf), "--base", str(base), *extra]
        if "--stats" not in extra:
            argv += ["--out", str(out)]
        r = subprocess.run([sys.executable, str(ROOT / "tools" / "draft_vocab.py"), *argv],
                           cwd=DIR, capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit(f"draft_vocab refused {name}: {r.stderr}")
        lines.append(f"cfg {name} args {' '.join(extra)}")
        for line in r.stdout.rstrip("\n").split("\n"):
            # the paths in the output are machine-specific; the counts around them are the assertion
            line = line.replace(str(out), "<out>").replace(str(gguf), "<gguf>").replace(str(base), "<base>")
            lines.append(f"stdout {name} {line}")
        if "--stats" not in extra:
            blob = out.read_bytes()
            lines.append(f"subset {name} bytes {len(blob)} sha256 {hashlib.sha256(blob).hexdigest()} "
                         f"ids {len(blob) // 4}")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="do not write; fail when the golden is out of date")
    a = ap.parse_args()
    text = build()
    if a.check:
        if not OUT.exists() or OUT.read_text(encoding="utf-8") != text:
            raise SystemExit("the draft-vocab golden is out of date - rerun without --check")
        print("the draft-vocab golden is current")
        return 0
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)} ({len(CONFIGS)} configurations)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
