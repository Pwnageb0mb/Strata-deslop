"""serve/native_stream.py - the output side of one generation, in C++ (libstrata_text, include/strata/text/strata_stream.h).

NativeDetokenizer, NativeStopMatcher and NativeOutputParser are drop-in replacements for server.Detokenizer,
server.StopMatcher and frontend.OutputParser (same methods, attributes and Event objects), and repeat_coverage() for
server.reasoning_repeat_coverage.  The Python classes stay as the reference and the test oracle
(tools/test_native_stream.py compares them); the server picks these when the library is there and
STRATA_PY_TOKENIZER is not "1" (see the new_* functions).

Where the port is not exact: nesting deeper than 500 in a tool argument's JSON reads as text (Python would raise
RecursionError); `parser.pending` is a list of None of the right length (the server only asks whether it is empty).
"""
from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path

from serve.frontend import THINK_END, Event, OutputParser, ToolCall

ABI_VERSION = 1
_STATES = ("reasoning", "content", "call", "json", "rcall")
_UTF8 = "surrogatepass"          # a Python str may hold a lone surrogate; it travels as its 3-byte form
_lib = None
_lib_path = None


class _Status(ctypes.Structure):
    _fields_ = [("state", ctypes.c_int32), ("pending", ctypes.c_int32), ("rescued", ctypes.c_int32),
                ("refused", ctypes.c_int32), ("buf_is_think_end", ctypes.c_int32), ("reserved", ctypes.c_int32),
                ("buf_len", ctypes.c_int64)]


def _bind(lib):
    c = ctypes
    p, i64, i32, cp = c.c_void_p, c.c_int64, c.c_int, c.c_char_p
    lib.strata_stream_abi_version.restype = i32
    lib.strata_stream_error.restype = cp
    sigs = {
        "strata_detok_new": ([], p), "strata_detok_free": ([p], None), "strata_detok_push": ([p, cp, i64], i64),
        "strata_detok_data": ([p], p), "strata_detok_pending": ([p], i32),
        "strata_stop_new": ([cp, p, i64], p), "strata_stop_free": ([p], None), "strata_stop_push": ([p, cp, i64], i64),
        "strata_stop_flush": ([p], i64), "strata_stop_data": ([p], p), "strata_stop_hit": ([p], i64),
        "strata_stop_held_len": ([p], i64), "strata_stop_held": ([p], p),
        "strata_parser_new": ([i32, i32, i32, cp, i64], p), "strata_parser_free": ([p], None),
        "strata_parser_feed": ([p, cp, i64, p], i64), "strata_parser_finish": ([p, i32, p], i64),
        "strata_parser_status_get": ([p, p], None), "strata_parser_data": ([p], p), "strata_parser_buf": ([p], p),
        "strata_repeat_coverage": ([cp, i64], c.c_double),
        "strata_pipe_new": ([p, p, i64, p, p, p, p, p, i64], p), "strata_pipe_free": ([p], None),
        "strata_pipe_step": ([p, c.c_int32, i32], i64), "strata_pipe_status": ([p], i64),
        "strata_pipe_data": ([p], p), "strata_pipe_tail": ([p], i64), "strata_pipe_tail_data": ([p], p),
    }
    for name, (args, res) in sigs.items():
        f = getattr(lib, name)
        f.argtypes, f.restype = args, res
    return lib


def load(path=None):
    """The bound library, or None when it is not there (or lacks this API).  Cached."""
    global _lib, _lib_path
    if _lib is not None and path is None:
        return _lib
    from serve.native_text import find_library
    found = Path(path) if path else find_library()
    if found is None:
        return None
    try:
        lib = _bind(ctypes.CDLL(str(found)))
        if lib.strata_stream_abi_version() != ABI_VERSION:
            return None
    except (OSError, AttributeError):
        return None
    _lib, _lib_path = lib, found
    return lib


def enabled() -> bool:
    """Use the native classes: the library has them and the Python implementations were not asked for."""
    return os.environ.get("STRATA_PY_TOKENIZER") != "1" and load() is not None


def _err(lib) -> str:
    return (lib.strata_stream_error() or b"").decode("utf-8", "replace")


def _text(ptr, n: int) -> str:
    return ctypes.string_at(ptr, n).decode("utf-8", _UTF8)


class NativeDetokenizer:
    """server.Detokenizer for a tokenizer with token_bytes: the incremental UTF-8 decoder is C++."""

    def __init__(self, tok):
        self._lib = load()
        self._h = self._lib.strata_detok_new()
        self.tok = tok

    def pending(self) -> bool:
        return bool(self._lib.strata_detok_pending(self._h))

    def push(self, t: int) -> str:
        b = self.tok.token_bytes(t)
        n = self._lib.strata_detok_push(self._h, b, len(b))
        return _text(self._lib.strata_detok_data(self._h), n) if n else ""

    def __del__(self):
        h, self._h = getattr(self, "_h", None), None
        if h:
            self._lib.strata_detok_free(h)


class NativeStopMatcher:
    """server.StopMatcher."""

    def __init__(self, stops: list[str]):
        self._lib = load()
        self.stops = stops
        enc = [s.encode("utf-8", _UTF8) for s in stops]
        offs = [0]
        for e in enc:
            offs.append(offs[-1] + len(e))
        arr = (ctypes.c_int64 * len(offs))(*offs)
        self._h = self._lib.strata_stop_new(b"".join(enc), arr, len(enc))
        if not self._h:
            raise MemoryError(_err(self._lib))

    @property
    def hit(self):
        i = self._lib.strata_stop_hit(self._h)
        return self.stops[i] if i >= 0 else None

    @property
    def held(self) -> str:
        n = self._lib.strata_stop_held_len(self._h)
        return _text(self._lib.strata_stop_held(self._h), n) if n else ""

    def push(self, text: str) -> str:
        b = text.encode("utf-8", _UTF8)
        n = self._lib.strata_stop_push(self._h, b, len(b))
        return _text(self._lib.strata_stop_data(self._h), n) if n else ""

    def flush(self) -> str:
        n = self._lib.strata_stop_flush(self._h)
        return _text(self._lib.strata_stop_data(self._h), n) if n else ""

    def __del__(self):
        h, self._h = getattr(self, "_h", None), None
        if h:
            self._lib.strata_stop_free(h)


def _tools_json(tools) -> bytes | None:
    """The request's tools reduced to what the parser reads: each name and its string-typed parameters."""
    if not tools:
        return None
    named = {}
    for t in tools:
        name = t.get("name")
        props = ((t.get("parameters") or {}).get("properties") or {})
        named[name] = [k for k, v in props.items() if isinstance(v, dict) and v.get("type") == "string"]
    return json.dumps({"any": bool(named), "tools": {k: v for k, v in named.items() if isinstance(k, str)}}
                      ).encode("utf-8")


class NativeOutputParser:
    """frontend.OutputParser: feed() text, get Events; state, buf, pending, rescued and refused as attributes."""

    def __init__(self, thinking: bool = True, tools: list[dict] | None = None, stream_tools: bool = False,
                 recover: bool = False):
        self._lib = load()
        tj = _tools_json(tools)
        self._h = self._lib.strata_parser_new(int(bool(thinking)), int(bool(stream_tools)), int(bool(recover)),
                                              tj, len(tj) if tj else 0)
        if not self._h:
            raise ValueError(_err(self._lib))
        self.stream_tools, self.recover = stream_tools, recover
        self._st = _Status()
        self._ref = ctypes.byref(self._st)
        self._calls: dict[int, ToolCall] = {}
        self._lib.strata_parser_status_get(self._h, self._ref)

    @property
    def state(self) -> str:
        return _STATES[self._st.state]

    @property
    def buf(self) -> str:
        n = self._st.buf_len
        if not n:
            return ""
        if self._st.buf_is_think_end:
            return THINK_END
        return _text(self._lib.strata_parser_buf(self._h), n)

    @property
    def pending(self) -> list:
        return [None] * self._st.pending

    @property
    def rescued(self) -> int:
        return self._st.rescued

    @property
    def refused(self) -> int:
        return self._st.refused

    def feed(self, delta: str) -> list[Event]:
        b = delta.encode("utf-8", _UTF8)
        n = self._lib.strata_parser_feed(self._h, b, len(b), self._ref)
        if n < 0:
            raise ValueError(_err(self._lib))
        return self._events(n)

    def finish(self, reason: str | None = None) -> list[Event]:
        n = self._lib.strata_parser_finish(self._h, int(reason in (None, "stop")), self._ref)
        if n < 0:
            raise ValueError(_err(self._lib))
        return self._events(n)

    def _events(self, n: int) -> list[Event]:
        if not n:
            return []
        return self._decode(ctypes.string_at(self._lib.strata_parser_data(self._h), n))

    def _decode(self, data: bytes) -> list[Event]:
        """The events of an encoded record buffer (strata_parser_data's format)."""
        n = len(data)
        out, pos, calls = [], 0, self._calls
        u32 = int.from_bytes
        while pos < n:
            kind = data[pos]
            pos += 1
            if kind <= 2:
                ln = u32(data[pos:pos + 4], "little")
                pos += 4
                out.append(Event("reasoning" if kind == 1 else "content", data[pos:pos + ln].decode("utf-8", _UTF8)))
                pos += ln
                continue
            cid = u32(data[pos:pos + 4], "little", signed=True)
            pos += 4
            if kind == 4:
                ln = u32(data[pos:pos + 4], "little")
                pos += 4
                call = calls[cid] = ToolCall(name=data[pos:pos + ln].decode("utf-8", _UTF8), arguments={})
                pos += ln
                out.append(Event("tool_start", call=call))
            elif kind == 5:
                ln = u32(data[pos:pos + 4], "little")
                pos += 4
                out.append(Event("tool_args", data[pos:pos + ln].decode("utf-8", _UTF8), call=calls[cid]))
                pos += ln
            else:
                same = data[pos]
                pos += 1
                ln = u32(data[pos:pos + 4], "little")
                pos += 4
                name = data[pos:pos + ln].decode("utf-8", _UTF8)
                pos += ln
                ln = u32(data[pos:pos + 4], "little")
                pos += 4
                args = data[pos:pos + ln].decode("utf-8", _UTF8)
                pos += ln
                if same:
                    call = calls[cid]
                else:
                    call = ToolCall(name=name, arguments=json.loads(args))
                    if cid >= 0:
                        call.id = calls[cid].id
                out.append(Event("tool_call", call=call))
        return out

    def __del__(self):
        h, self._h = getattr(self, "_h", None), None
        if h:
            self._lib.strata_parser_free(h)


def repeat_coverage(text: str) -> float:
    """server.reasoning_repeat_coverage."""
    b = text.encode("utf-8", _UTF8)
    r = load().strata_repeat_coverage(b, len(b))
    if r < 0:
        raise RuntimeError(_err(load()))
    return r


# ---- the fused step ----------------------------------------------------------------------------------------------
# What the server reads after every token, as one int (strata_stream.h's STRATA_PIPE_* bits): the parser state in the
# low three bits (0 reasoning, 1 content, 2 call, 3 json, 4 rcall) and these.
F_STATE = 7
F_BUF = 0x8            # parser.buf is not empty
F_PENDING = 0x10       # parser.pending is not empty
F_DETOK = 0x20         # detok.pending()
F_HIT = 0x40           # a stop string matched
F_THINK_END = 0x80     # parser.buf == THINK_END
F_THINKING = 0x100     # the state is reasoning or rcall
_OVERFLOW, _ONE_R, _ONE_C = 0x200, 0x400, 0x800
_STATE_IX = {s: i for i, s in enumerate(_STATES)}


class _Pipe:
    """token id -> events, as the server runs it: detok.push, parser.feed, then cut() at the stop strings.  `fl` holds
    the flags above as they were after the last step / sync(); `tail` the last two characters the tracked steps wrote."""
    fl = 0

    def __init__(self, parser, detok, stops):
        self.parser, self.detok, self.stops = parser, detok, stops

    @property
    def state(self) -> str:
        return _STATES[self.fl & F_STATE]

    def cut(self, evs):
        """The events with the answer's text cut at a stop string (OpenAI stop, Anthropic stop_sequences)."""
        stops = self.stops
        if stops is None:
            return evs
        out = []
        for ev in evs:
            if stops.hit is not None:
                break                               # nothing after the stop string is sent
            if ev.kind == "content":
                text = stops.push(ev.text or "")
                if text:
                    out.append(Event("content", text))
                continue
            held = stops.flush()                    # a stop string does not run across a tool call or thinking
            if held:
                out.append(Event("content", held))
            out.append(ev)
        return out


class PyPipe(_Pipe):
    """The chain over any detokenizer / parser / stop matcher (the Python classes, or the native ones one call each)."""

    def __init__(self, parser, detok, stops):
        super().__init__(parser, detok, stops)
        self.tail = ""
        self.sync()

    def sync(self) -> int:
        p, st = self.parser, self.parser.state
        fl = _STATE_IX[st]
        if p.buf:
            fl |= F_BUF
            if p.buf == THINK_END:
                fl |= F_THINK_END
        if p.pending:
            fl |= F_PENDING
        if self.detok.pending():
            fl |= F_DETOK
        if self.stops is not None and self.stops.hit is not None:
            fl |= F_HIT
        if st == "reasoning" or st == "rcall":
            fl |= F_THINKING
        self.fl = fl
        return fl

    def step(self, t: int, track: bool = True) -> list[Event]:
        piece = self.detok.push(t)
        if track:
            self.tail = (self.tail + piece)[-2:]
        try:
            evs = self.cut(self.parser.feed(piece))
        finally:
            self.sync()
        return evs


class NativePipe(_Pipe):
    """strata_pipe_step: the same chain in one C call per token (all three must be the native classes)."""
    _CAP = 1 << 16

    def __init__(self, tok, parser, detok, stops):
        super().__init__(parser, detok, stops)
        lib = self._lib = load()
        self._tok = tok
        self._buf = ctypes.create_string_buffer(self._CAP)
        self._h = lib.strata_pipe_new(ctypes.addressof(tok.raw_blob), ctypes.addressof(tok.raw_offsets), len(tok),
                                      detok._h, stops._h if stops is not None else None, parser._h, parser._ref,
                                      self._buf,
                                      self._CAP)
        if not self._h:
            raise MemoryError(_err(lib))
        self._h = ctypes.c_void_p(self._h)
        self._step = lib.strata_pipe_step
        self._decode = parser._decode
        self.fl = lib.strata_pipe_status(self._h)

    def sync(self) -> int:
        """Read the flags again after something other than step() changed the parser, detokenizer or stop matcher."""
        self.fl = self._lib.strata_pipe_status(self._h)
        return self.fl

    @property
    def tail(self) -> str:
        n = self._lib.strata_pipe_tail(self._h)
        return _text_lossy(self._lib.strata_pipe_tail_data(self._h), n)[-2:] if n else ""

    def step(self, t: int, track: bool = True) -> list[Event]:
        r = self._step(self._h, t, track)
        if r < 0:
            self.sync()
            if r == -2:
                raise IndexError("token id %d is outside the vocabulary (%d)" % (t, len(self._tok)))
            raise ValueError(_err(self._lib))
        self.fl = r & 0x1FF
        n = r >> 16
        if not n:
            return []
        if r & _OVERFLOW:
            data = ctypes.string_at(self._lib.strata_pipe_data(self._h), n)
        else:
            data = self._buf[:n]
        if r & _ONE_R:
            return [Event("reasoning", data.decode("utf-8", _UTF8))]
        if r & _ONE_C:
            return [Event("content", data.decode("utf-8", _UTF8))]
        return self._decode(data)

    def __del__(self):
        h, self._h = getattr(self, "_h", None), None
        if h:
            self._lib.strata_pipe_free(h)


def _text_lossy(ptr, n: int) -> str:
    return ctypes.string_at(ptr, n).decode("utf-8", "replace")


# ---- what the server calls: the native class when it is there, else the Python reference ------------------------
def new_detokenizer(tok, py_cls):
    return NativeDetokenizer(tok) if hasattr(tok, "token_bytes") and enabled() else py_cls(tok)


def new_stop_matcher(stops, py_cls):
    return NativeStopMatcher(stops) if enabled() else py_cls(stops)


def new_output_parser(**kw):
    return NativeOutputParser(**kw) if enabled() else OutputParser(**kw)


def new_pipe(tok, parser, detok, stops):
    """The fused one-call-per-token chain when all three are native and the tokenizer is libstrata_text's, else the
    same chain call by call."""
    if (isinstance(parser, NativeOutputParser) and isinstance(detok, NativeDetokenizer) and detok.tok is tok
            and (stops is None or isinstance(stops, NativeStopMatcher)) and hasattr(tok, "raw_blob")):
        return NativePipe(tok, parser, detok, stops)
    return PyPipe(parser, detok, stops)
