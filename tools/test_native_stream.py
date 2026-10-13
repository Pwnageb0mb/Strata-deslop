"""tools/test_native_stream.py - libstrata_text's output side (serve/native_stream.py) against the Python reference.

The Python Detokenizer, StopMatcher, OutputParser and reasoning_repeat_coverage (serve/server.py, serve/frontend.py)
are the oracle: the same input goes through both and every event, state and output is compared.

    STRATA_TEXT_LIB=build/libstrata_text.so STRATA_TOKENIZER=<pack>/tokenizer python -m unittest tools.test_native_stream
    STRATA_FUZZ=20000  (texts per parser mode; default 1500)    STRATA_SEED=n

Skipped without the library.  The vocabulary tests also need a tokenizer directory.
"""
from __future__ import annotations

import codecs
import json
import os
import random
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from serve import native_stream as NS  # noqa: E402
from serve.frontend import OutputParser  # noqa: E402
from serve.server import Detokenizer, StopMatcher, reasoning_repeat_coverage  # noqa: E402

LIB = NS.load() if os.environ.get("STRATA_TEXT_LIB") else None
FUZZ = int(os.environ.get("STRATA_FUZZ", "1500"))
SEED = int(os.environ.get("STRATA_SEED", "20260512"))

TOOLS = [
    {"name": "write", "parameters": {"properties": {"path": {"type": "string"}, "count": {"type": "integer"},
                                                       "data": {"type": "object"}, "flag": {"type": "boolean"}}}},
    {"name": "bash", "parameters": {"properties": {"command": {"type": "string"}}}},
    {"name": "noargs"},
]


def find_tokenizer():
    for c in [os.environ.get("STRATA_TOKENIZER", "")] + [str(p) for p in ROOT.glob("packs/*/tokenizer")]:
        if c and (Path(c) / "vocab.json").exists():
            return Path(c)
    return None


# ------------------------------------------------------------------------------------------------ parser
def norm(evs, ids):
    """Events as comparable tuples; call ids become their order of first appearance."""
    out = []
    for e in evs:
        call = None
        if e.call is not None:
            idx = ids.setdefault(e.call.id, len(ids))
            call = (idx, e.call.name, json.dumps(e.call.arguments, ensure_ascii=False))
        out.append((e.kind, e.text, call))
    return out


def snapshot(p):
    return (p.state, p.buf, len(p.pending), p.rescued, p.refused)


def run_parser(cls, text, chunks, finish, **kw):
    p, ids, trace = cls(**kw), {}, []
    try:
        for c in chunks:
            trace.append(("feed", norm(p.feed(c), ids), snapshot(p)))
        trace.append(("finish", norm(p.finish(finish), ids), snapshot(p)))
    except ValueError as e:
        trace.append(("ValueError", str(e), snapshot(p)))
    return trace


def chunked(text, rnd, width):
    if width == 0:
        return [text]
    out, i = [], 0
    while i < len(text):
        k = width if width > 0 else rnd.randint(1, 9)
        out.append(text[i:i + k])
        i += k
    return out


FRAGS = [
    "Let me think. ", "I will read the file.", "\n", "\n\n", "  ", " ", "```\n", "```python\n", "~~~\n", "`", "``",
    "</think>", "</think>\n\n", "\"</think>\"", "`</think>`", "<think>", "<tool_call>", "</tool_call>", "<tool_call>\n",
    "<function=write>", "<function=bash>", "<function= write>", "<function=nope>", "<function=noargs>\n</function>",
    "\n<function=write>\n", "</function>", "<parameter=path>", "<parameter=count>", "<parameter=data>",
    "<parameter=flag>", "<parameter=command>", "<parameter=write>", "<parameter=bash>", "</parameter>",
    "</parameter>\n", "\n</parameter>\n", "file.txt", "a/b.txt", "42", "-0", "1.50", "1e400", "00", "{\"a\": [1, 2.5e3, \"x\", null]}",
    "{\"a\":1,\"a\":2}", "[1,", "true", "null", "NaN", "\"quoted\"", "é", "日本語", "😀", "\u00a0", "\u2028", "\\n", "\"",
    "<", ">", "</", "<tool_", "</thi", "<parameter", "</param", "<func", "Σ", "ǅ", "\t",
    '{"name": "write", "arguments": {"path": "a"}}', '{"name": "bash", "parameters": {"command": "ls"}}',
    '{"name": "bash", "arguments": "{\\"command\\": \\"pwd\\"}"}', '{"name": "zzz", "arguments": {}}', "{", "}",
    "x = 1\n", "rm -rf build", "<tool_call>\n<function=write>\nx<parameter=zz</tool_call>", "x<parameter=zz", "<tool_call>\n<function=write>\n<parameter=path>\nfile.txt\n</parameter>\n</function>\n</tool_call>",
    "<tool_call>\n<function=bash>\n<parameter=command>\nls -la\n</parameter>\n</function>\n</tool_call>",
    "<tool_call>\n<function=write>\n<parameter=data>\n{\"k\": [1, 2, {\"z\": \"\\u00e9\\ud83d\\ude00\"}]}\n</parameter>\n</function>\n</tool_call>",
    "<tool_call>\n<function=write>\n<parameter=path>\nnotes</parameter> and </function> inside\n</parameter>\n</function>\n</tool_call>",
    "<tool_call>\n<function=write>\n<parameter=count>\n7\n</parameter>\n<parameter=flag>\ntrue\n</parameter>\n</function>\n</tool_call>",
    "<tool_call>\n<function=write>\n<parameter=path>\n", "<parameter=path>\nx\n</parameter>\n</function>\n</tool_call>",
]


def random_text(rnd):
    n = rnd.randint(1, 24)
    return "".join(rnd.choice(FRAGS) for _ in range(n))


@unittest.skipIf(LIB is None, "STRATA_TEXT_LIB is not set to a libstrata_text with the stream API")
class ParserDifferential(unittest.TestCase):
    def same(self, text, chunks, finish, **kw):
        want = run_parser(OutputParser, text, chunks, finish, **kw)
        got = run_parser(NS.NativeOutputParser, text, chunks, finish, **kw)
        if want != got:
            for i, (a, b) in enumerate(zip(want, got)):
                if a != b:
                    self.fail(f"step {i} differs\n kw={kw} finish={finish}\n chunks={chunks!r}\n python={a!r}\n native={b!r}")
            self.fail(f"traces differ in length: {len(want)} vs {len(got)}\n kw={kw} chunks={chunks!r}")

    def modes(self):
        for thinking in (True, False):
            for tools in (None, TOOLS):
                for stream_tools in (False, True):
                    for recover in (False, True):
                        yield dict(thinking=thinking, tools=tools, stream_tools=stream_tools, recover=recover)

    def test_specimens_and_fixtures(self):
        rnd = random.Random(SEED)
        specs = json.loads((ROOT / "serve/fixtures/rcall_specimens.json").read_text(encoding="utf-8"))["specimens"]
        texts = [(s["input"], s.get("finish")) for s in specs]
        call = "<tool_call>\n<function=write>\n<parameter=path>\nfile.txt\n</parameter>\n</function>\n</tool_call>"
        texts += [(call, None), ("think</think>\n\n" + call, "stop"), ("a" + call + "b", "length"),
                  ("I'll use a <tool_call> block.", None), ("x\n<function=bash>\n<parameter=command>\nls\n</parameter>\n</function>", None)]
        for text, finish in texts:
            for kw in self.modes():
                for width in (0, 1, 2, 3, 7, -1):
                    self.same(text, chunked(text, rnd, width), finish, **kw)

    def test_random_texts_random_chunkings(self):
        rnd = random.Random(SEED + 1)
        seen = {"tool_call": 0, "tool_args": 0, "reasoning": 0, "content": 0, "error": 0}
        for n in range(FUZZ):
            text = random_text(rnd)
            kw = rnd.choice(list(self.modes()))
            finish = rnd.choice((None, None, "stop", "length"))
            chunks = chunked(text, rnd, rnd.choice((-1, -1, 1, 2, 5, 0)))
            self.same(text, chunks, finish, **kw)
            for step in run_parser(OutputParser, text, chunks, finish, **kw):
                if step[0] == "ValueError":
                    seen["error"] += 1
                else:
                    for e in step[1]:
                        seen[e[0]] = seen.get(e[0], 0) + 1
        print("\nparser fuzz coverage:", seen)
        if FUZZ >= 500:
            self.assertTrue(all(seen[k] > 5 for k in seen), seen)       # the corpus reaches every kind of event

    def test_every_mode_on_each_text(self):
        rnd = random.Random(SEED + 2)
        for n in range(max(50, FUZZ // 10)):
            text = random_text(rnd)
            for kw in self.modes():
                self.same(text, chunked(text, rnd, -1), rnd.choice((None, "length")), **kw)

    def test_long_values_and_a_runaway_call(self):
        rnd = random.Random(SEED + 3)
        body = "".join(rnd.choice(["word ", "é", "\n", "`", "</parameter", "x" * 50]) for _ in range(3000))
        for text in ("<tool_call>\n<function=write>\n<parameter=path>\n" + body + "\n</parameter>\n</function>\n</tool_call>",
                     "think\n<tool_call>\n<function=write>\n<parameter=path>\n" + body, "t " * 20000 + "\n<tool_call>x"):
            for kw in self.modes():
                self.same(text, chunked(text, rnd, -1), None, **kw)
                self.same(text, chunked(text, rnd, 97), "stop", **kw)

    def test_unicode_whitespace_and_surrogates(self):
        rnd = random.Random(SEED + 4)
        text = ("\u00a0\u2003<tool_call>\u2028<function=write>\u3000<parameter=path>\n\u1680a\u202f</parameter>\u205f"
                "</function>\u0085</tool_call>\u00a0")
        for kw in self.modes():
            self.same(text, chunked(text, rnd, 1), None, **kw)
            self.same(text, [text], None, **kw)
        text = "<tool_call>\n<function=write>\n<parameter=path>\n\ud800x\n</parameter>\n<parameter=data>\n\"\\ud800\"\n</parameter>\n</function>\n</tool_call>"
        for kw in self.modes():
            self.same(text, [text], None, **kw)


# ------------------------------------------------------------------------------------------------ detokenizer
class FakeTok:
    def __init__(self, table):
        self.table = table

    def token_bytes(self, i):
        return self.table[i]

    def decode(self, ids, errors="replace"):
        return b"".join(self.table[i] for i in ids).decode("utf-8", errors=errors)


@unittest.skipIf(LIB is None, "STRATA_TEXT_LIB is not set")
class DetokenizerDifferential(unittest.TestCase):
    def compare(self, tok, ids):
        a, b = Detokenizer(tok), NS.NativeDetokenizer(tok)
        for k, t in enumerate(ids):
            ra, rb = a.push(t), b.push(t)
            self.assertEqual((ra, a.pending()), (rb, b.pending()), f"token {k} of {ids[:k + 1]}")
        # and against the whole decode: the held tail aside, the same text
        whole = b"".join(tok.token_bytes(t) for t in ids)
        dec = codecs.getincrementaldecoder("utf-8")(errors="replace")
        ref = Detokenizer(tok)
        self.assertEqual(dec.decode(whole), "".join(ref.push(t) for t in ids))

    def test_random_bytes(self):
        rnd = random.Random(SEED + 10)
        pool = [0x00, 0x41, 0x7F, 0x80, 0xBF, 0xC0, 0xC1, 0xC2, 0xDF, 0xE0, 0xE1, 0xEC, 0xED, 0xEE, 0xF0, 0xF1, 0xF4, 0xF5, 0xFF,
                0xA0, 0x9F, 0x90, 0x8F, 0x91]
        for _ in range(400):
            table = [bytes(rnd.choice(pool) if rnd.random() < .8 else rnd.randrange(256) for _ in range(rnd.randint(0, 4)))
                     for _ in range(40)]
            self.compare(FakeTok(table), [rnd.randrange(40) for _ in range(rnd.randint(1, 40))])

    def test_valid_text_split_everywhere(self):
        rnd = random.Random(SEED + 11)
        text = "héllo 日本語 😀 ΑΣ \u00a0 \ud7ff \uffff 𐍈 e\u0301 \n" * 3
        raw = text.encode("utf-8")
        for _ in range(300):
            cuts = sorted(rnd.sample(range(1, len(raw)), rnd.randint(1, 30)))
            parts = [raw[a:b] for a, b in zip([0] + cuts, cuts + [len(raw)])]
            self.compare(FakeTok(parts), list(range(len(parts))))

    @unittest.skipIf(find_tokenizer() is None, "no tokenizer directory (STRATA_TOKENIZER)")
    def test_real_vocabulary(self):
        from serve.native_text import NativeTokenizer
        tok = NativeTokenizer(find_tokenizer())
        rnd = random.Random(SEED + 12)
        n = len(tok)
        for _ in range(200):
            self.compare(tok, [rnd.randrange(n) for _ in range(rnd.randint(1, 60))])
        sample = "Hello wörld! 日本語のテキスト 😀🎉 Привет, мир ΑΣΣ ```py\nprint(1)\n```\n" * 4
        ids = tok.encode(sample)
        self.compare(tok, ids)
        self.assertEqual(NS.new_detokenizer(tok, Detokenizer).__class__, NS.NativeDetokenizer)


# ------------------------------------------------------------------------------------------------ the fused pipe
class PieceTok(FakeTok):
    """A tokenizer that is a table of byte pieces, with the blob the C pipe reads."""

    def __init__(self, table):
        import ctypes
        super().__init__(table)
        self.raw_blob = ctypes.create_string_buffer(b"".join(table) or b"\0", max(1, sum(map(len, table))))
        offs = [0]
        for t in table:
            offs.append(offs[-1] + len(t))
        self.raw_offsets = (ctypes.c_int64 * len(offs))(*offs)

    def __len__(self):
        return len(self.table)


def server_chain(tok, parser, detok, stops):
    """The server's own sequence (serve/server.py before the pipe), written out as the oracle."""
    from serve.frontend import Event

    def cut(evs):
        if stops is None:
            return evs
        out = []
        for ev in evs:
            if stops.hit is not None:
                break
            if ev.kind == "content":
                text = stops.push(ev.text or "")
                if text:
                    out.append(Event("content", text))
                continue
            held = stops.flush()
            if held:
                out.append(Event("content", held))
            out.append(ev)
        return out
    return cut


def flags_of(parser, detok, stops):
    f = {"reasoning": 0, "content": 1, "call": 2, "json": 3, "rcall": 4}[parser.state]
    f |= NS.F_BUF if parser.buf else 0
    f |= NS.F_PENDING if parser.pending else 0
    f |= NS.F_DETOK if detok.pending() else 0
    f |= NS.F_HIT if stops is not None and stops.hit is not None else 0
    f |= NS.F_THINK_END if parser.buf == "</think>" else 0
    f |= NS.F_THINKING if parser.state in ("reasoning", "rcall") else 0
    return f


@unittest.skipIf(LIB is None, "STRATA_TEXT_LIB is not set")
class PipeDifferential(unittest.TestCase):
    STOPS = [["\n\n"], ["END", "é"], ["file"], ["<tool"], ["</think>\n"], ["x" * 3], []]

    def one(self, rnd, text, kw, stop_list, finish):
        raw = text.encode("utf-8", "surrogatepass")
        cuts = sorted(rnd.sample(range(1, len(raw)), min(len(raw) - 1, rnd.randint(0, max(1, len(raw) // 2))))) if len(raw) > 1 else []
        table = [raw[a:b] for a, b in zip([0] + cuts, cuts + [len(raw)])]
        tok = PieceTok(table)
        want_p, want_d = OutputParser(**kw), Detokenizer(tok)
        want_s = StopMatcher(stop_list) if stop_list else None
        cut = server_chain(tok, want_p, want_d, want_s)
        got_p, got_d = NS.NativeOutputParser(**kw), NS.NativeDetokenizer(tok)
        got_s = NS.NativeStopMatcher(stop_list) if stop_list else None
        pipe = NS.new_pipe(tok, got_p, got_d, got_s)
        self.assertIsInstance(pipe, NS.NativePipe)
        py_p, py_d = OutputParser(**kw), Detokenizer(tok)
        py_s = StopMatcher(stop_list) if stop_list else None
        pypipe = NS.PyPipe(py_p, py_d, py_s)
        ids_w, ids_g, ids_y = {}, {}, {}
        tail = ""
        self.assertEqual(pipe.fl, flags_of(want_p, want_d, want_s))
        for k in range(len(table)):
            track = rnd.random() < .8
            where = f"token {k}/{len(table)} kw={kw} stops={stop_list} text={text!r} table={table!r}"
            try:
                piece = want_d.push(k)
                if track:
                    tail = (tail + piece)[-2:]
                w = ("ok", norm(cut(want_p.feed(piece)), ids_w))
            except ValueError as e:
                w = ("ValueError", str(e))
            for name, pp, ids in (("native", pipe, ids_g), ("python chain", pypipe, ids_y)):
                try:
                    g = ("ok", norm(pp.step(k, track), ids))
                except ValueError as e:
                    g = ("ValueError", str(e))
                self.assertEqual(w, g, f"{name}: {where}")
                self.assertEqual(flags_of(want_p, want_d, want_s), pp.fl, f"{name} flags: {where}")
                self.assertEqual(tail, pp.tail, f"{name} tail: {where}")
                self.assertEqual(want_p.state, pp.state)
            if w[0] == "ValueError":
                return "error"
            self.assertEqual(want_p.buf, got_p.buf)
            self.assertEqual(want_s.held if want_s else None, got_s.held if got_s else None)
            if want_s is not None and want_s.hit is not None:
                break
        w = norm(cut(want_p.finish(finish)), ids_w)
        self.assertEqual(w, norm(pipe.cut(got_p.finish(finish)), ids_g))
        self.assertEqual(w, norm(pypipe.cut(py_p.finish(finish)), ids_y))
        return "ok"

    def test_fuzz(self):
        rnd = random.Random(SEED + 30)
        modes = list(ParserDifferential.modes(None))
        seen = {"ok": 0, "error": 0}
        for _ in range(FUZZ):
            seen[self.one(rnd, random_text(rnd), rnd.choice(modes), rnd.choice(self.STOPS),
                          rnd.choice((None, "stop", "length")))] += 1
        print("\npipe fuzz:", seen)

    def test_overflow_and_a_bad_id(self):
        import ctypes
        big = "z" * 200000
        tok = PieceTok([big.encode(), b"a", b"<tool_call>\n<function=write>\n<parameter=path>\n" + big.encode()])
        for kw in (dict(thinking=False), dict(thinking=False, tools=TOOLS, stream_tools=True)):
            pipe = NS.new_pipe(tok, NS.NativeOutputParser(**kw), NS.NativeDetokenizer(tok), None)
            ref = NS.PyPipe(OutputParser(**kw), Detokenizer(tok), None)
            for t in (0, 1, 2):
                self.assertEqual(norm(pipe.step(t), {}), norm(ref.step(t), {}))
            with self.assertRaises(IndexError):
                pipe.step(3)
            with self.assertRaises(IndexError):
                pipe.step(-1)

    @unittest.skipIf(find_tokenizer() is None, "no tokenizer directory (STRATA_TOKENIZER)")
    def test_real_vocabulary(self):
        from serve.native_text import NativeTokenizer
        tok = NativeTokenizer(find_tokenizer())
        text = ("<think>\nHello wörld 日本語 😀\n</think>\n\nAnswer. END more\n<tool_call>\n<function=write>\n"
                "<parameter=path>\nfile.txt\n</parameter>\n</function>\n</tool_call>")
        ids = tok.encode(text)
        for stops in (None, ["END"], ["\n\n"]):
            kw = dict(thinking=True, tools=TOOLS, stream_tools=True)
            a = NS.new_pipe(tok, NS.NativeOutputParser(**kw), NS.NativeDetokenizer(tok),
                            NS.NativeStopMatcher(stops) if stops else None)
            self.assertIsInstance(a, NS.NativePipe)
            b = NS.PyPipe(OutputParser(**kw), Detokenizer(tok), StopMatcher(stops) if stops else None)
            ia, ib = {}, {}
            for t in ids:
                self.assertEqual(norm(a.step(t), ia), norm(b.step(t), ib))
                self.assertEqual(a.fl, b.fl)
                if a.fl & NS.F_HIT:
                    break


# ------------------------------------------------------------------------------------------------ stop strings
@unittest.skipIf(LIB is None, "STRATA_TEXT_LIB is not set")
class StopDifferential(unittest.TestCase):
    def test_random(self):
        rnd = random.Random(SEED + 20)
        alpha = "ab</>é日\n😀"
        for _ in range(3000):
            stops = [("".join(rnd.choice(alpha) for _ in range(rnd.randint(1, 5)))) for _ in range(rnd.randint(1, 4))]
            text = "".join(rnd.choice(alpha) for _ in range(rnd.randint(0, 40)))
            a, b = StopMatcher(stops), NS.NativeStopMatcher(stops)
            i = 0
            while i < len(text):
                k = rnd.randint(1, 4)
                self.assertEqual((a.push(text[i:i + k]), a.hit, a.held), (b.push(text[i:i + k]), b.hit, b.held), (stops, text, i))
                i += k
            self.assertEqual((a.flush(), a.held, a.hit), (b.flush(), b.held, b.hit))
            self.assertEqual(a.push("ab"), b.push("ab"))

    def test_an_empty_stop_string_and_ties(self):
        for stops, text in ((["", "a"], "xa"), (["ab", "b"], "xabx"), (["b", "ab"], "xabx"), (["abc"], "xxab")):
            a, b = StopMatcher(stops), NS.NativeStopMatcher(stops)
            self.assertEqual((a.push(text), a.hit, a.held), (b.push(text), b.hit, b.held))


# ------------------------------------------------------------------------------------------------ repeat coverage
@unittest.skipIf(LIB is None, "STRATA_TEXT_LIB is not set")
class CoverageDifferential(unittest.TestCase):
    def same(self, text, msg=""):
        self.assertEqual(NS.repeat_coverage(text), reasoning_repeat_coverage(text), msg or text[:60])

    def test_the_servers_own_cases(self):
        loop = "Need maybe Agent support agent seven b. " * 700
        for t in (" ".join(str(i) for i in range(4000)), loop, "Counting the digits of 9^104. " + "58747693387" * 3000,
                  "Digits: " + "".join(str(i) for i in range(1000)), "58747693387" * 100, "", "x", "a b " * 3000):
            self.same(t)

    def test_random_repetition_noise_and_cases(self):
        rnd = random.Random(SEED + 30)
        scripts = {
            "latin": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_",
            "greek": "αβγδεζηθικλμνξοπρστυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩςάέή",
            "cyr": "абвгдежзийклмнопрстуфхцчшщъыьэюяАБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЭЮЯ",
            "tr": "İıIiŞşĞğÜüÖöÇç",
            "de": "ßẞSsÄäÖöÜü",
            "cjk": "日本語中文字漢字한국어ひらがなカタカナ",
            "mix": "éÉ ǅǆǄ ⅧⅷⒶⓐ ʰ· ’'.:ΣσΑ٣٤",
        }
        for name, alpha in scripts.items():
            for rep in range(4):
                vocab = ["".join(rnd.choice(alpha) for _ in range(rnd.randint(1, 6))) for _ in range(300)]
                block = [rnd.choice(vocab + [",", ".", "(", ")", "\n", "\u00a0", "—"]) for _ in range(rnd.randint(40, 400))]
                words = []
                for _ in range(rnd.randint(8, 30)):
                    seg = list(block)
                    if rnd.random() < .5:
                        seg.insert(rnd.randrange(len(seg)), rnd.choice(vocab))
                    words += seg
                text = " ".join(words)
                for f in (str, str.upper, str.lower, str.title, str.swapcase):
                    self.same(f(text), f"{name} {f.__name__}")

    def test_random_code_points_repeated(self):
        rnd = random.Random(SEED + 31)
        for _ in range(12):
            cps = [c for c in (rnd.randrange(0x110000) for _ in range(400)) if not 0xD800 <= c < 0xE000]
            block = "".join(map(chr, cps)).replace("\x00", "")
            for f in (str, str.upper, str.swapcase):
                self.same(f(" ".join([block[i:i + 5] for i in range(0, len(block), 5)] * 30)), "random code points")

    def test_the_tail_period_rule(self):
        rnd = random.Random(SEED + 32)
        for p in (1, 2, 11, 63, 64, 65):
            unit = "".join(rnd.choice("abcΣσ0123") for _ in range(p))
            for n in (2047, 2048, 2049, 5000):
                self.same("head. " + (unit * (n // p + 2))[:n])
                self.same("HEAD İ " + (unit * (n // p + 2))[:n].upper())

    def test_lowercase_mapping_of_every_cased_code_point(self):
        """Words that differ only by case are the same word only if lower() agrees: put every cased code point (and
        a sample of the rest) in a passage, repeat it in upper / swapped case, and compare the coverage."""
        rnd = random.Random(SEED + 33)
        cased = [c for c in range(0x1F000) if not 0xD800 <= c < 0xE000 and (chr(c).lower() != chr(c) or chr(c).upper() != chr(c))]
        rest = [c for c in (rnd.randrange(0x110000) for _ in range(2400)) if not 0xD800 <= c < 0xE000]
        cps = cased + rest + [0x3A3, 0x3C2, 0x3C3, 0x130, 0x131, 0xDF, 0x1E9E, 0x1C5, 0x2160, 0x24B6]
        for i in range(0, len(cps), 70):
            block = []
            for c in cps[i:i + 70]:
                block += [f"f{c}x{k}" for k in range(20)] + ["q" + chr(c) + "q", "q" + chr(c)]
            for f in (str.upper, str.swapcase):
                text = " ".join(block) + " " + f(" ".join(block)) + " " + " ".join(block)
                self.same(text, f"{f.__name__} U+{cps[i]:04X}..")


if __name__ == "__main__":
    unittest.main()
