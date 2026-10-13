"""tools/test_native_request.py - libstrata_text's request translation (serve/native_request.py) against the Python
reference (serve/frontend.py openai_to_messages / anthropic_to_messages).

The same request goes through both and the results are compared: ==, and the json.dumps text (so dict key order),
or, where the reference raises, the exception type and message.  Inputs:
  * every request the test suites (serve/test_server.py, test_responses.py, test_security.py, test_frontend.py) send to
    the two functions, recorded by running those suites once (cached; STRATA_REQUEST_CORPUS=<file> picks the cache);
  * a randomized generator of valid and invalid request shapes (STRATA_FUZZ per API, default 4000; STRATA_SEED).
A native "leave it to Python" answer is allowed only where the reference raises something other than ValueError.

    STRATA_TEXT_LIB=build/libstrata_text.so python -m unittest tools.test_native_request
    STRATA_TEXT_LIB=build/libstrata_text.so python tools/test_native_request.py --bench      (per-call timing)
"""
from __future__ import annotations

import json
import math
import os
import random
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

CORPUS = Path(os.environ.get("STRATA_REQUEST_CORPUS") or Path(tempfile.gettempdir()) / "strata_request_corpus.json")
FUZZ = int(os.environ.get("STRATA_FUZZ", "4000"))
SEED = int(os.environ.get("STRATA_SEED", "20260513"))
SUITES = ["serve.test_server", "serve.test_responses", "serve.test_security", "serve.test_frontend"]


def harvest(out: Path):
    """Run the suites once with the two functions recording their arguments (a subprocess: --harvest <file>)."""
    os.environ["STRATA_NATIVE_REQUEST"] = "0"
    from serve import frontend
    rec, seen = [], set()

    def note(api, req, think):
        try:
            text = json.dumps(req)
        except (TypeError, ValueError):
            return
        key = (api, text, think)
        if key not in seen:
            seen.add(key)
            rec.append([api, text, think])
    orig_o, orig_a = frontend.openai_to_messages, frontend.anthropic_to_messages

    def openai(req):
        note(0, req, True)
        return orig_o(req)

    def anthropic(req, think_unasked=True):
        note(1, req, think_unasked)
        return orig_a(req, think_unasked)
    frontend.openai_to_messages, frontend.anthropic_to_messages = openai, anthropic
    suite = unittest.defaultTestLoader.loadTestsFromNames(SUITES)
    with open(os.devnull, "w") as dn:
        unittest.TextTestRunner(stream=dn, verbosity=0).run(suite)
    out.write_text(json.dumps(rec))


def load_corpus():
    if not CORPUS.is_file():
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "--harvest", str(CORPUS)], cwd=ROOT, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=3000)
    return [(api, json.loads(text), think) for api, text, think in json.loads(CORPUS.read_text())]


if __name__ == "__main__" and len(sys.argv) == 3 and sys.argv[1] == "--harvest":
    harvest(Path(sys.argv[2]))
    sys.exit(0)

from serve import frontend, native_request  # noqa: E402

LIB = native_request.load() if os.environ.get("STRATA_TEXT_LIB") else None
PY = (frontend.openai_to_messages, frontend.anthropic_to_messages)
NATIVE = native_request.converters(*PY, lib=LIB) if LIB else PY


# ---------------------------------------------------------------------------------------------------- comparing
def outcome(fn, req, think):
    frontend._TOOL_ORDERS.clear()
    try:
        return ("ok", fn(req) if think is None else fn(req, think))
    except Exception as e:  # noqa: BLE001
        return ("err", type(e).__name__, str(e))


def dumps(x):
    return json.dumps(x, ensure_ascii=False)


def same(a, b):
    """a == b, with NaN equal to NaN and True != 1 (as json.dumps shows them)."""
    if type(a) is not type(b):
        return False
    if isinstance(a, float):
        return a == b or (math.isnan(a) and math.isnan(b))
    if isinstance(a, list):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, tuple):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, dict):
        return list(a) == list(b) and all(same(a[k], b[k]) for k in a)
    return a == b


def compare(api, req, think):
    """None when the two agree; else a description."""
    before = dumps(req)
    py = outcome(PY[api], req, None if api == 0 else think)
    mid = dumps(req)
    nat = outcome(NATIVE[api], req, None if api == 0 else think)
    if dumps(req) != mid or mid != before:
        return "the request was changed"
    if py[0] != nat[0] or (py[0] == "err" and py[1:] != nat[1:]):
        return f"python {py!r:.600}\nnative {nat!r:.600}"
    if py[0] == "ok" and (not same(py[1], nat[1]) or dumps(py[1]) != dumps(nat[1])):
        return f"python {dumps(py[1]):.900}\nnative {dumps(nat[1]):.900}"
    return None


def native_kind(api, req, think):
    data = json.dumps(req).encode("ascii")
    n = LIB.strata_request_convert(api, data, len(data), 1 if think else 0)
    return 0 if n >= 0 else LIB.strata_request_error_kind()


def needless_fallback(api, req, think):
    """True when the library gave the case to Python although the reference does not raise a non-ValueError."""
    if native_kind(api, req, think) != 2:
        return False
    py = outcome(PY[api], req, None if api == 0 else think)
    return py[0] == "ok" or py[1] == "ValueError"


# ---------------------------------------------------------------------------------------------------- generator
TEXTS = ["", "hi", "hello world", "héllo", "emoji \U0001F600 ok", "a\ud800b", "<think>x</think>", "line\nbreak\ttab",
         "quote's \" both", "  padded  ", "{\"a\": 1}", "[1, 2]", "42", "null", " sep", "\x7f\x01ctl", "­​x"]
BILLING = ["x-anthropic-billing-header: cc_version=2.1.170.bf4; cc_entrypoint=sdk-cli; cch=b145e;",
           "x-anthropic-billing-header: cc_version=2.1.170.bf4;cch=b145e;\nYou are Claude.",
           "x-anthropic-billing-header: cch=1234567890abcdef0;cc_version=1.2.3;rest",
           "x-anthropic-billing-header: cc_version=1.2.3.4.5.6; cch=zz;",
           "x-anthropic-billing-header: cc_version=1.2.3.; cch=;",
           "x-anthropic-billing-header: éé cc_version=9.9.9.abc; \U0001F600 cch=abc;",
           "x-anthropic-billing-header:" + " " * 170 + "cch=aa; cc_version=1.2.3.4;",
           "x-anthropic-billing-header: cc_version=" + "1." * 40 + "x; cch=" + "a" * 17 + ";",
           "no header cc_version=1.2.3.4; cch=ab;"]
EFFORTS = ["none", "off", "minimal", "disabled", "false", "low", "medium", "high", "xhigh", "max", "maximum", "LOW", " High ",
           "Medium\n", "bogus", "", "é", "it's", 'say "x"', "it's \"q\"", True, False, None, 0, 1, 2.5, [], ["low"], {"a": "b"},
           float("nan"), float("inf"), -1, 10 ** 30]
BUDGETS = [0, 1, 100, 2047, 2048, 4096, 8191, 8192, 100000, -5, 1.5, 2047.9, 8192.0, "1000", " 9000 ", "1_000", "x", "", "12x",
           "-3", "+9999", "007", "1__0", "_1", "1_", "99999999999999999999999", True, False, None, [], {"a": 1}, float("nan"),
           1e30, -1e30, 10 ** 25, -(10 ** 25)]


class Gen:
    def __init__(self, seed):
        self.r = random.Random(seed)

    def p(self, x):
        return self.r.random() < x

    def pick(self, seq):
        return seq[self.r.randrange(len(seq))]

    def text(self):
        return self.pick(TEXTS)

    def junk(self, depth=0):
        k = self.r.randrange(14 if depth < 2 else 11)
        if k < 11:
            return [None, True, False, 0, 7, -3, 2.5, "s", "", self.text(), 1e300][k] if k != 10 else float("nan")
        if k in (11, 12):
            return [self.junk(depth + 1) for _ in range(self.r.randrange(3))]
        return {self.text(): self.junk(depth + 1) for _ in range(self.r.randrange(3))}

    def maybe_junk(self, v, q=0.05):
        return self.junk() if self.p(q) else v

    def jsonish(self, depth=0):
        k = self.r.randrange(8 if depth < 3 else 5)
        if k == 0:
            return self.text()
        if k == 1:
            return self.r.randrange(-5, 100)
        if k == 2:
            return self.pick([None, True, False, 0.5, 1e-7, 1.0, 123456789.125])
        if k == 3:
            return "k"
        if k == 4:
            return self.text()
        if k == 5:
            return [self.jsonish(depth + 1) for _ in range(self.r.randrange(4))]
        return {self.pick(["a", "b", "path", "xé", "n"]): self.jsonish(depth + 1) for _ in range(self.r.randrange(4))}

    def url(self):
        return self.pick(["http://x/y.png", "data:image/png;base64,AAAA", "/tmp/a.png", "", "é.png"])

    def image_source(self):
        k = self.r.randrange(8)
        if k == 0:
            return {"type": "base64", "media_type": self.pick(["image/jpeg", "image/png", None, 5, ["x"]]), "data": self.pick(["QUJD", "", None, 12])}
        if k == 1:
            return {"type": "base64", "data": "QUJD"}
        if k == 2:
            return {"type": "url", "url": self.url()}
        if k == 3:
            return {"path": self.pick(["/a/b", "", None, 3]), "url": self.pick(["", None])}
        if k == 4:
            return self.pick([None, {}, "str", [], 0, True])
        if k == 5:
            return {"type": "base64"}
        return {"type": self.pick(["url", "other", None]), "url": self.maybe_junk(self.url(), 0.2), "path": self.url()}

    def part(self, anth):
        t = self.pick(["text", "text", "text", "input_text", "image_url", "input_image", "image", "image", "refusal", "file",
                       "audio", None, "MISSING", 5, "thinking", "tool_use", "tool_result", "redacted_thinking"] if anth else
                      ["text", "text", "input_text", "image_url", "image_url", "input_image", "image", "refusal", "file", None,
                       "MISSING", 5, ["text"]])
        d = {}
        if t != "MISSING":
            d["type"] = t
        if self.p(0.8) or t in ("text", "input_text"):
            d["text"] = self.maybe_junk(self.text(), 0.08)
        if t in ("image_url", "input_image") or self.p(0.1):
            d["image_url"] = self.pick([self.url(), {"url": self.url()}, {"url": None}, {}, None, 7, {"url": ""}])
        if t == "image" or self.p(0.1):
            d["source"] = self.image_source()
        if t == "refusal" or self.p(0.1):
            d["refusal"] = self.pick(["no", "", None, 4])
        if t == "thinking" or self.p(0.1):
            d["thinking"] = self.maybe_junk(self.text(), 0.05)
        if t == "tool_use" or self.p(0.05):
            d["name"] = self.pick(["bash", "read", None, 3, ""])
            d["id"] = "toolu_1"
            if self.p(0.9):
                d["input"] = self.args()
        if t == "tool_result" or self.p(0.05):
            d["tool_use_id"] = "toolu_1"
            if self.p(0.9):
                d["content"] = self.content(anth, nested=True)
        if self.p(0.1):
            d["extra"] = self.jsonish()
        keys = list(d)
        if self.p(0.3):
            self.r.shuffle(keys)
        return {k: d[k] for k in keys}

    def content(self, anth, nested=False):
        k = self.r.randrange(20)
        if k < 6:
            return self.text()
        if k < 15:
            return [self.part(anth) if not self.p(0.06) else self.junk() for _ in range(self.r.randrange(5))]
        if k == 15:
            return None
        if k == 16:
            return self.pick([{}, {"type": "text", "text": "x"}, 0, 5, 2.5, True, False, [], ""])
        if k == 17 and not nested:
            return [self.part(anth) for _ in range(self.r.randrange(1, 4))]
        return self.text()

    def args(self):
        k = self.r.randrange(14)
        if k < 3:
            return {self.pick(["path", "cmd", "a b", "xé"]): self.jsonish() for _ in range(self.r.randrange(4))}
        if k == 3:
            return json.dumps({"path": self.text(), "n": self.jsonish()})
        if k == 4:
            return self.pick(["", "   ", None, "{bad", "[1, 2]", "42", "null", "\"s\"", "{\"a\": 1, \"a\": 2}", "{\"a\":NaN}", "true",
                              "﻿{}", " {\"x\": 1} \n", "{\"z\": 1e400}", "{\"s\": \"\\ud83d\\ude00\"}"])
        if k == 5:
            return self.pick([[1, "a", None], 42, 2.5, True, False, [], [{"a": "é"}], float("inf"), 10 ** 40, "é"])
        return self.maybe_junk({}, 0.5)

    def tool_call(self):
        k = self.r.randrange(10)
        if k < 5:
            return {"id": "c1", "type": "function", "function": {"name": self.pick(["bash", "read", None, 3]), "arguments": self.args()}}
        if k == 5:
            return {"name": "bare", "arguments": self.args()}
        if k == 6:
            return {"function": self.pick([None, "str", [], 5, {}])}
        if k == 7:
            return {"function": {"arguments": self.args()}}
        return self.junk() if self.p(0.3) else {"function": {"name": "f", "arguments": self.args()}}

    def messages_value(self, items):
        k = self.r.randrange(40)
        if k == 0:
            return None
        if k == 1:
            return json.dumps(items)
        if k == 2:
            return self.junk()
        if k == 3:
            return json.dumps(items) + "x"
        if k == 4:
            return json.dumps(self.junk())
        return items

    def openai_msg(self):
        m = {}
        if self.p(0.97):
            m["role"] = self.pick(["user", "user", "assistant", "assistant", "system", "developer", "tool", "function", None,
                                   self.junk()])
        m["content"] = self.content(False) if self.p(0.95) else None
        if "content" in m and self.p(0.05):
            del m["content"]
        if self.p(0.15):
            m["reasoning_content"] = self.pick(["because", "", None, 0, ["r"], self.text()])
        if self.p(0.25):
            tc = [self.tool_call() for _ in range(self.r.randrange(4))]
            m["tool_calls"] = self.pick([tc, tc, tc, json.dumps(tc), None, [], "", self.junk(), json.dumps(tc)[:-1]])
        if self.p(0.1):
            m["tool_call_id"] = "c1"
        return m

    def tool_openai(self):
        k = self.r.randrange(12)
        name = self.pick(["bash", "read", "write", "glob", "mcp__x__y", "", None, 5])
        fn = {"name": name, "description": self.pick(["d", "", None, 5]), "parameters": self.pick(
            [{"type": "object", "properties": {"a": {"type": "string"}}}, {}, None, "str", [], 0, True])}
        if self.p(0.4):
            fn.pop("description")
        if self.p(0.3):
            fn.pop("parameters")
        if k < 6:
            return {"type": "function", "function": fn}
        if k == 6:
            return fn
        if k == 7:
            return {"type": "function", "name": name}
        if k == 8:
            return {"type": "custom", "function": fn}
        if k == 9:
            return {"type": "function", "function": self.pick([None, "x", [], 3])}
        if k == 10:
            return {"name": name, "input_schema": {}}
        return self.junk()

    def tools_openai(self):
        names = self.r.sample(["bash", "read", "write", "glob", "grep", "edit"], self.r.randrange(1, 6))
        tools = [{"type": "function", "function": {"name": n, "description": "d " + n, "parameters": {"type": "object"}}} for n in names]
        if self.p(0.6):
            tools = [self.tool_openai() for _ in range(self.r.randrange(4))] if self.p(0.5) else tools
        k = self.r.randrange(30)
        if k == 0:
            return "auto"
        if k == 1:
            return json.dumps(tools)
        if k == 2:
            return self.junk()
        if k == 3:
            return ["bash", "read"]
        if k == 4:
            return {"type": "function"}
        if k == 5:
            return None
        return tools

    def openai_req(self):
        req = {}
        if self.p(0.97):
            req["messages"] = self.messages_value([self.openai_msg() if not self.p(0.03) else self.junk()
                                                   for _ in range(self.r.randrange(0, 7))])
        if self.p(0.4):
            req["tools"] = self.tools_openai()
        if self.p(0.3):
            req["reasoning_effort"] = self.pick(EFFORTS)
        if self.p(0.2):
            req["reasoning"] = self.pick([{"effort": self.pick(EFFORTS)}, {}, None, "x", [], {"effort": None}, {"summary": "auto"}])
        if self.p(0.3):
            ctk = {}
            for k in self.r.sample(["enable_thinking", "reasoning_effort", "other"], self.r.randrange(4)):
                ctk[k] = self.pick(EFFORTS) if k == "reasoning_effort" else self.pick([True, False, None, 0, 1, "", "x"])
            req["chat_template_kwargs"] = self.pick([ctk, ctk, ctk, None, {}, "str", [], [1], 0, True])
        for k in ("model", "temperature", "stream", "tool_choice", "max_tokens"):
            if self.p(0.2):
                req[k] = self.jsonish()
        return req

    def anth_msg(self):
        m = {}
        if self.p(0.97):
            m["role"] = self.pick(["user", "user", "assistant", "assistant", "system", None, 4])
        m["content"] = self.content(True) if self.p(0.97) else None
        if self.p(0.04):
            del m["content"]
        return m

    def tool_anth(self):
        k = self.r.randrange(10)
        name = self.pick(["bash", "read", "write", "glob", "mcp__x__y", "", None, 5])
        t = {"name": name, "description": self.pick(["d", "", None, 5]), "input_schema": self.pick(
            [{"type": "object", "properties": {"a": {"type": "string"}}}, {}, None, "str", [], 0, True, 2.5])}
        if self.p(0.4):
            t.pop("description")
        if self.p(0.3):
            t.pop("input_schema")
        if k == 7:
            return {"type": "function", "function": {"name": "f"}}
        if k == 8:
            return self.junk()
        if k == 9:
            return {"type": "custom", **t}
        return t

    def tools_anth(self):
        names = self.r.sample(["bash", "read", "write", "glob", "grep", "edit"], self.r.randrange(1, 6))
        tools = [{"name": n, "description": "d " + n, "input_schema": {"type": "object"}} for n in names]
        if self.p(0.6):
            tools = [self.tool_anth() for _ in range(self.r.randrange(4))] if self.p(0.5) else tools
        k = self.r.randrange(30)
        if k == 0:
            return "auto"
        if k == 1:
            return json.dumps(tools)
        if k == 2:
            return self.junk()
        if k == 3:
            return None
        return tools

    def anthropic_req(self):
        req = {}
        if self.p(0.5):
            sys_ = self.pick([self.pick(BILLING), self.text(), None, "", [{"type": "text", "text": self.pick(BILLING)}],
                              [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}, {"type": "image"}, self.junk()],
                              [{"type": "refusal", "refusal": "x"}], {}, {"a": 1}, self.junk(), 5, True, []])
            req["system"] = sys_
        if self.p(0.97):
            req["messages"] = self.messages_value([self.anth_msg() if not self.p(0.03) else self.junk()
                                                   for _ in range(self.r.randrange(0, 7))])
        if self.p(0.4):
            req["tools"] = self.tools_anth()
        if self.p(0.5):
            th = self.pick([{"type": "disabled"}, {"type": "enabled", "budget_tokens": self.pick(BUDGETS)},
                            {"type": "enabled"}, {"budget_tokens": self.pick(BUDGETS)}, None, "x", [], {}, {"type": "adaptive"},
                            {"type": "disabled", "budget_tokens": 5}])
            if th is not None or self.p(0.3):
                req["thinking"] = th
        if self.p(0.3):
            req["output_config"] = self.pick([{"effort": self.pick(EFFORTS)}, {}, None, "x", [], {"effort": None}, {"format": 1}])
        if self.p(0.15):
            req["reasoning_budget_tokens"] = self.pick([0, 100, None, "x", True])
        for k in ("model", "max_tokens", "stream", "tool_choice", "metadata"):
            if self.p(0.2):
                req[k] = self.jsonish()
        return req


def strip_unicode_budget(req):
    """The library leaves int(str) with Unicode digits to Python on purpose; the generator has none, this is a guard."""
    return req


@unittest.skipUnless(LIB, "STRATA_TEXT_LIB (libstrata_text with the request API) is not set")
class NativeRequest(unittest.TestCase):
    def check(self, cases, label):
        bad, needless = [], []
        for n, (api, req, think) in enumerate(cases):
            why = compare(api, req, think)
            if why:
                bad.append(f"[{label} #{n} api={api} think={think}] {dumps(req):.1500}\n{why}")
            elif needless_fallback(api, req, think):
                needless.append(f"[{label} #{n} api={api}] {dumps(req):.600}")
            if len(bad) >= 5:
                break
        self.assertEqual(bad, [], "\n\n".join(bad))
        self.assertEqual(needless[:5], [], f"{len(needless)} needless fallbacks:\n" + "\n".join(needless[:5]))

    def test_suite_requests(self):
        cases = load_corpus()
        self.assertGreater(len(cases), 20)
        print(f"\n  suites: {len(cases)} distinct requests", file=sys.stderr)
        self.check(cases, "suite")

    def test_generated_openai(self):
        g = Gen(SEED)
        self.check([(0, g.openai_req(), True) for _ in range(FUZZ)], "openai")

    def test_generated_anthropic(self):
        g = Gen(SEED + 1)
        self.check([(1, g.anthropic_req(), g.p(0.5)) for _ in range(FUZZ)], "anthropic")

    def test_each_request_through_both_apis(self):
        g = Gen(SEED + 2)
        cases = []
        for _ in range(FUZZ // 2):
            cases.append((1, g.openai_req(), g.p(0.5)))
            cases.append((0, g.anthropic_req(), True))
        self.check(cases, "crossed")

    def test_fallback_is_the_reference(self):
        # shapes the library leaves to Python: the wrapper must give exactly what Python gives, errors included
        cases = [(0, {"messages": [{"role": "user", "content": 5}]}), (0, {"messages": [{"role": "user", "content": [{"text": None}]}]}),
                 (0, {"messages": [], "chat_template_kwargs": "x"}), (1, {"messages": [{"content": "x"}]}),
                 (1, {"messages": [{"role": "user", "content": [1]}]}),
                 (1, {"messages": [], "thinking": {"budget_tokens": "١٢٣"}}),
                 (1, {"messages": [], "thinking": {"budget_tokens": float("inf")}}), (1, [1, 2]), (0, "x")]
        for api, req in cases:
            self.assertIsNone(compare(api, req, True), (api, req))
        self.assertEqual(NATIVE[1]({"messages": [], "thinking": {"budget_tokens": "٨٠٠٠"}})[2],
                         PY[1]({"messages": [], "thinking": {"budget_tokens": "٨٠٠٠"}})[2])

    def test_error_messages(self):
        for api, req in [(0, {"messages": "[1"}), (0, {"messages": [1]}), (0, {"tools": "auto", "messages": []}),
                         (0, {"messages": [], "reasoning_effort": "it's"}), (0, {"messages": [], "reasoning_effort": [1, "a'b", {"k": None}]}),
                         (1, {"messages": [], "tools": [{"name": "x", "input_schema": 5}]}),
                         (1, {"messages": [], "output_config": {"effort": 2.5}}),
                         (0, {"messages": [{"role": "assistant", "tool_calls": [{"function": 3}]}]})]:
            self.assertIsNone(compare(api, req, True), (api, req))
            self.assertEqual(outcome(NATIVE[api], req, None if api == 0 else True)[0], "err")

    def test_tool_order_is_shared(self):
        frontend._TOOL_ORDERS.clear()
        a = {"messages": [], "tools": [{"name": "a"}, {"name": "b"}, {"name": "c"}]}
        b = {"messages": [], "tools": [{"name": "c"}, {"name": "a"}, {"name": "b"}]}
        first = NATIVE[1](a)[1]
        second = PY[1](b)[1]             # the Python path sees the order the native path registered
        self.assertEqual([t["name"] for t in first], ["a", "b", "c"])
        self.assertEqual([t["name"] for t in second], ["a", "b", "c"])


# ---------------------------------------------------------------------------------------------------- benchmark
def big_request(n=200, tools=24):
    g = Gen(5)
    msgs = []
    for i in range(n):
        k = i % 5
        if k == 0:
            msgs.append({"role": "user", "content": f"question {i}: " + "lorem ipsum dolor sit amet " * 20})
        elif k == 1:
            msgs.append({"role": "assistant", "content": "I will look.", "tool_calls": [
                {"id": f"c{i}", "type": "function", "function": {"name": "read", "arguments": json.dumps({"path": f"/a/b/{i}.py", "n": i})}}]})
        elif k == 2:
            msgs.append({"role": "tool", "tool_call_id": f"c{i - 1}", "content": "file body\n" * 40})
        elif k == 3:
            msgs.append({"role": "assistant", "content": [{"type": "text", "text": "answer " * 30}]})
        else:
            msgs.append({"role": "user", "content": [{"type": "text", "text": "next " * 10}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]})
    tl = [{"type": "function", "function": {"name": f"tool{i}", "description": "does a thing " * 5,
                                            "parameters": {"type": "object", "properties": {"a": {"type": "string"}, "b": {"type": "integer"}}}}} for i in range(tools)]
    return {"messages": msgs, "tools": tl, "reasoning_effort": "high"}, g


def big_anthropic(n=200, tools=24):
    msgs = []
    for i in range(n):
        k = i % 5
        if k == 0:
            msgs.append({"role": "user", "content": f"question {i}: " + "lorem ipsum dolor sit amet " * 20})
        elif k == 1:
            msgs.append({"role": "assistant", "content": [{"type": "thinking", "thinking": "hmm " * 20}, {"type": "text", "text": "I will look."},
                                                          {"type": "tool_use", "id": f"t{i}", "name": "read", "input": {"path": f"/a/b/{i}.py", "n": i}}]})
        elif k == 2:
            msgs.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": f"t{i - 1}", "content": [{"type": "text", "text": "file body\n" * 40}]}]})
        elif k == 3:
            msgs.append({"role": "assistant", "content": [{"type": "text", "text": "answer " * 30}]})
        else:
            msgs.append({"role": "user", "content": [{"type": "text", "text": "next " * 10}]})
    tl = [{"name": f"tool{i}", "description": "does a thing " * 5, "input_schema": {"type": "object", "properties": {"a": {"type": "string"}}}}
          for i in range(tools)]
    return {"system": BILLING[0] + "\nYou are a coding agent. " * 40, "messages": msgs, "tools": tl, "thinking": {"type": "enabled", "budget_tokens": 5000}}


def bench(reps=300):
    import ctypes
    oreq, _ = big_request()
    areq = big_anthropic()
    for name, api, req in (("openai", 0, oreq), ("anthropic", 1, areq)):
        assert compare(api, req, True) is None
        py, nat = PY[api], NATIVE[api]
        res = {}
        for label, fn in (("python", py), ("native", nat)):
            for _ in range(20):
                fn(req)
            best = []
            for _ in range(5):
                t = time.perf_counter()
                for _ in range(reps):
                    fn(req)
                best.append((time.perf_counter() - t) / reps)
            res[label] = min(best)
        data = json.dumps(req).encode("ascii")
        t = time.perf_counter()
        for _ in range(reps):
            json.dumps(req).encode("ascii")
        t_dump = (time.perf_counter() - t) / reps
        t = time.perf_counter()
        for _ in range(reps):
            LIB.strata_request_convert(api, data, len(data), 1)
        t_conv = (time.perf_counter() - t) / reps
        n = LIB.strata_request_convert(api, data, len(data), 1)
        buf = ctypes.create_string_buffer(n)
        LIB.strata_request_result(buf, n)
        raw = buf.raw
        t = time.perf_counter()
        for _ in range(reps):
            json.loads(raw)
        t_load = (time.perf_counter() - t) / reps
        print(f"{name:10s} {len(req['messages'])} messages, request {len(data) // 1024} KiB, result {n // 1024} KiB: "
              f"python {res['python'] * 1e6:8.1f} us   native {res['native'] * 1e6:8.1f} us   "
              f"(json.dumps {t_dump * 1e6:.1f} + convert {t_conv * 1e6:.1f} + json.loads {t_load * 1e6:.1f})   "
              f"native/python = {res['native'] / res['python']:.2f}")


if __name__ == "__main__":
    if "--bench" in sys.argv:
        if not LIB:
            sys.exit("set STRATA_TEXT_LIB")
        bench()
    else:
        unittest.main()
