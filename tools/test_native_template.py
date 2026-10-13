"""tools/test_native_template.py - the C++ chat-template renderer (libstrata_text, llama.cpp's Jinja engine) must
render byte-for-byte what jinja2 renders, for the deployed template and serve/chat_template.jinja.

  STRATA_TEXT_LIB=<build>/libstrata_text.so python -m unittest tools.test_native_template
  STRATA_CHAT_TEMPLATE=<path>   one more template to check (default: the pack's tokenizer/chat_template.jinja, if given)

Skipped when STRATA_TEXT_LIB is not set.  Also times a 200-turn conversation (printed, not asserted).
"""
from __future__ import annotations

import copy
import json
import os
import sys
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

LIB = os.environ.get("STRATA_TEXT_LIB")
TEMPLATES = [ROOT / "serve" / "chat_template.jinja"]
if os.environ.get("STRATA_CHAT_TEMPLATE"):
    TEMPLATES.append(Path(os.environ["STRATA_CHAT_TEMPLATE"]))

U = lambda c: {"role": "user", "content": c}                      # noqa: E731
A = lambda c, **kw: {"role": "assistant", "content": c, **kw}     # noqa: E731
S = lambda c: {"role": "system", "content": c}                    # noqa: E731
T = lambda c: {"role": "tool", "content": c}                      # noqa: E731
CALL = lambda name, args: {"function": {"name": name, "arguments": args}}        # noqa: E731

TOOLS = {
    "none": None,
    "empty list": [],
    "flat": [{"name": "read", "description": "Read a file", "parameters": {
        "type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}],
    "openai wrapped": [{"type": "function", "function": {"name": "get_weather", "description": "Weather in a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "Name"},
                       "unit": {"type": "string", "enum": ["c", "f"]}}, "required": ["city"]}}}],
    "unicode": [{"name": "traduire", "description": "Übersetze é中文 \U0001F600 \"quoted\" <b>&amp;</b>",
                 "parameters": {"type": "object", "properties": {"texte": {"type": "string", "default": "café"}}}}],
    "nested": [{"name": "deep", "description": "d", "parameters": {"type": "object", "properties": {
        "a": {"type": "array", "items": {"type": "object", "properties": {"b": {"type": ["string", "null"]},
              "c": {"anyOf": [{"type": "integer", "minimum": 0}, {"type": "number", "maximum": 1.5}]}}}},
        "flag": {"type": "boolean", "default": True}, "n": {"type": "null"}, "z": {"type": "integer", "default": 0}},
        "additionalProperties": False}}],
    "empty params": [{"name": "noargs", "description": "", "parameters": {}}],
    "two": [{"name": f"t{i}", "description": f"tool {i}", "parameters": {"type": "object", "properties": {}}}
            for i in range(2)],
    "escapes": [{"name": "esc", "description": "line1\nline2\ttab \\ back / slash \x01 ctl",
                 "parameters": {"type": "object", "properties": {"k\"ey": {"type": "string"}}}}],
}

IMG = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
CONVOS = {
    "single user": [U("Hello")],
    "system+user": [S("You are terse."), U("Hi")],
    "developer": [{"role": "developer", "content": "Be careful."}, U("Hi")],
    "multi-turn": [S("sys"), U("a"), A("b"), U("c"), A("d"), U("e")],
    "reasoning_content": [U("a"), A("answer", reasoning_content="why\nbecause"), U("b")],
    "reasoning only history": [U("a"), A("", reasoning_content="thinking"), U("b")],
    "think tags in content": [U("a"), A("<think>\nin content\n</think>\n\nanswer"), U("b")],
    "tool call dict args": [U("read it"), A("", tool_calls=[CALL("read", {"path": "/a.txt", "n": 3})]), T("data"), U("ok")],
    "tool call str args": [U("read it"), A("", tool_calls=[CALL("read", '{"path": "/a.txt"}')]), T("data")],
    "tool call mixed value types": [U("x"), A("c", tool_calls=[CALL("f", {"s": "str", "i": 1, "f": 1.5, "b": True,
                                                                          "nul": None, "l": [1, "a", {"k": "v"}],
                                                                          "o": {"x": {"y": [1, 2]}}, "u": "café 中"})]),
                                    T("r"), A("done")],
    "parallel calls": [U("x"), A("", tool_calls=[CALL("a", {"p": 1}), CALL("b", {"q": "two"})]), T("ra"), T("rb"), U("y")],
    "tool result list content": [U("x"), A("", tool_calls=[CALL("a", {})]), {"role": "tool", "content": [{"type": "text", "text": "r1"}]}],
    "tool call with id": [U("x"), A("", tool_calls=[{"id": "call_1", "type": "function", "function": {"name": "a", "arguments": {"z": 1}}}]),
                          {"role": "tool", "tool_call_id": "call_1", "content": "res"}, U("z")],
    "image part": [U([{"type": "text", "text": "what is this"}, IMG])],
    "two images": [U([IMG, {"type": "text", "text": "and"}, {"type": "image", "source": "b.png"}])],
    "text parts": [S("s"), U([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]), A([{"type": "text", "text": "ans"}])],
    "video part": [U([{"type": "video", "video": "v.mp4"}])],
    "none content": [U("x"), A(None, tool_calls=[CALL("f", {})]), T("r")],
    "empty content": [U(""), A(""), U("again")],
    "empty assistant mid": [U("a"), A(""), U("b")],
    "unicode": [U("café 中文 \U0001F600 <|im_end|> {{ x }} {% y %}")],
    "template syntax in text": [U("{{ 7*7 }} {%- if x -%} ... {# c #}")],
    "whitespace": [S("  padded  \n"), U("\n\n x \n\n"), A("  y  ")],
    "last is assistant": [U("a"), A("partial")],
    "tool last": [U("a"), A("", tool_calls=[CALL("f", {"a": 1})]), T("r")],
    "image in system (raises)": [{"role": "system", "content": [IMG]}, U("x")],
    "no user (raises)": [S("only system")],
    "empty list (raises)": [],
    "unknown content type (raises)": [U(5)],
    "bad item (raises)": [U([{"type": "audio"}])],
    "system after user": [U("a"), S("late system")],
    "tool first": [T("r"), U("a")],
    "unknown role": [{"role": "narrator", "content": "x"}, U("a")],
}
KWARGS = {
    "none": {},
    "enable_thinking false": {"enable_thinking": False},
    "enable_thinking true": {"enable_thinking": True},
    "effort low": {"reasoning_effort": "low"},
    "effort medium": {"reasoning_effort": "medium"},
    "effort xhigh": {"reasoning_effort": "xhigh"},
    "effort+thinking off": {"enable_thinking": False, "reasoning_effort": "low"},
    "unknown effort": {"reasoning_effort": "bogus"},
    "preserve_thinking": {"preserve_thinking": True},
    "chat_template_kwargs": {"add_vision_id": True, "custom": {"a": [1, 2]}, "s": "x"},
}


def outcome(fn):
    try:
        return ("ok", fn())
    except Exception as e:                                          # noqa: BLE001
        return (type(e).__name__, str(e))


@unittest.skipUnless(LIB and Path(LIB).is_file(), "STRATA_TEXT_LIB not set")
class NativeTemplateParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.pop("STRATA_PY_TOKENIZER", None)
        from serve import frontend
        cls.frontend = frontend
        cls.pairs = []
        for path in TEMPLATES:
            if not path.is_file():
                continue
            nat = frontend.ChatTemplate(path)
            os.environ["STRATA_PY_TOKENIZER"] = "1"
            try:
                py = frontend.ChatTemplate(path)
            finally:
                del os.environ["STRATA_PY_TOKENIZER"]
            assert nat.engine == "native" and py.engine == "jinja2", (nat.engine, py.engine)
            # the native side must not quietly fall back to jinja2
            nat._jinja = lambda: (_ for _ in ()).throw(AssertionError("native render fell back to jinja2"))
            cls.pairs.append((path, nat, py))

    def compare(self, nat, py, msgs, tools, gen, kw):
        a = outcome(lambda: nat.render(copy.deepcopy(msgs), tools=copy.deepcopy(tools), add_generation_prompt=gen, **kw))
        b = outcome(lambda: py.render(copy.deepcopy(msgs), tools=copy.deepcopy(tools), add_generation_prompt=gen, **kw))
        if b[0] == "TemplateRequestError":
            self.assertEqual(a, b)                                  # the template's own message, same type
        else:
            self.assertEqual(a, b)

    def test_conversations(self):
        n = 0
        for path, nat, py in self.pairs:
            for cname, msgs in CONVOS.items():
                for kname, kw in KWARGS.items():
                    for gen in (True, False):
                        for tname in ("none", "flat", "unicode"):
                            with self.subTest(template=path.name, convo=cname, kwargs=kname, gen=gen, tools=tname):
                                self.compare(nat, py, msgs, TOOLS[tname], gen, kw)
                                n += 1
        print(f"\nconversation grid: {n} renders compared", file=sys.stderr)

    def test_tool_shapes(self):
        convo = [S("s"), U("go"), A("", tool_calls=[CALL("deep", {"a": [{"b": None}]})]), T("r"), U("again")]
        for path, nat, py in self.pairs:
            for tname, tools in TOOLS.items():
                for kw in ({}, {"enable_thinking": False}, {"reasoning_effort": "low"}):
                    with self.subTest(template=path.name, tools=tname, kw=kw):
                        self.compare(nat, py, convo, tools, True, kw)

    def test_golden_pack_cases(self):
        golden = json.loads((ROOT / "serve" / "chat_golden.json").read_text(encoding="utf-8"))
        for path, nat, py in self.pairs:
            for case in golden:
                with self.subTest(template=path.name, case=case["name"]):
                    self.compare(nat, py, case["messages"], case.get("tools"), True, {})

    def test_raise_exception_is_a_request_error(self):
        for path, nat, py in self.pairs:
            with self.subTest(template=path.name):
                with self.assertRaises(self.frontend.TemplateRequestError) as a:
                    nat.render([])
                with self.assertRaises(self.frontend.TemplateRequestError) as b:
                    py.render([])
                self.assertEqual(str(a.exception), str(b.exception))
                self.assertIsInstance(a.exception, ValueError)
                self.assertNotIn("Jinja Exception", str(a.exception))

    def test_caps_and_source(self):
        for path, nat, py in self.pairs:
            self.assertEqual(nat.caps, py.caps)
            self.assertEqual(nat.source, py.source)

    def test_empty_turn_filter_and_threads(self):
        import threading
        path, nat, py = self.pairs[0]
        msgs = [U("a"), A("x" * 50), U("b")]
        want = py.render(msgs)
        errs = []

        def work():
            for _ in range(50):
                if nat.render(msgs) != want:
                    errs.append(1)
        ts = [threading.Thread(target=work) for _ in range(8)]
        [t.start() for t in ts]
        [t.join() for t in ts]
        self.assertEqual(errs, [])

    def test_bad_template_falls_back(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "t.jinja"
            p.write_text("{% if %}", encoding="utf-8")
            with self.assertRaises(Exception):
                self.frontend.ChatTemplate(p)                       # a bad template fails at load, as before

    def test_long_conversation_timing(self):
        """Prints (does not assert) native vs jinja2 for a 201-message conversation with 3 tools: per render
        end to end, then the part of the native render that is JSON serialisation in Python.  Best of 5 runs of 20."""
        msgs = [S("You are a helpful assistant.")]
        for i in range(100):
            msgs += [U(f"question {i} " + "lorem ipsum " * 30),
                     A(f"answer {i} " + "dolor sit amet " * 30, reasoning_content=f"thought {i} " * 10)]
        msgs.append(U("last"))
        tools = TOOLS["nested"] + TOOLS["flat"] + TOOLS["openai wrapped"]

        def best_ms(fn, runs=5, n=20):
            best = float("inf")
            for _ in range(runs):
                t0 = time.perf_counter()
                for _ in range(n):
                    fn()
                best = min(best, (time.perf_counter() - t0) / n * 1000)
            return best

        for path, nat, py in self.pairs:
            self.compare(nat, py, msgs, tools, True, {"enable_thinking": True})
            t = {name: best_ms(lambda tpl=tpl: tpl.render(msgs, tools=tools, enable_thinking=True))
                 for name, tpl in (("native", nat), ("jinja2", py))}
            dump = best_ms(lambda: (json.dumps(msgs, ensure_ascii=False).encode("utf-8", "replace"),
                                    json.dumps(tools, ensure_ascii=False).encode("utf-8", "replace"),
                                    json.dumps({"enable_thinking": True}).encode()))
            print(f"\n[{path.name}] 201 messages, {len(nat.render(msgs, tools=tools))} chars, per render: "
                  f"native {t['native']:.2f} ms ({dump:.2f} ms of it json.dumps), jinja2 {t['jinja2']:.2f} ms "
                  f"({t['jinja2'] / t['native']:.2f}x)", file=sys.stderr)

if __name__ == "__main__":
    unittest.main()
