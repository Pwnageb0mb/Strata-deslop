"""serve/native_request.py - the request translation of serve/frontend.py, in C++ (libstrata_text,
include/strata/text/strata_request.h).

openai_to_messages / anthropic_to_messages take the request as JSON and give back {"messages", "tools", "kwargs"}
as JSON; the result is what the Python functions return, key order included (tools/test_native_request.py compares
them).  The Python functions stay the reference and the fallback: a request shape the library leaves to them (one
where the reference raises TypeError / AttributeError / KeyError) is run by them, so that behaviour is theirs.
The stable tool order (frontend._stable_tool_order) keeps process-wide state shared with the Python path, so it is
applied here, in Python, to the tools that come back.

converters(py_openai, py_anthropic) is the factory the server uses: the native pair when it is switched on and the
library is there, else the Python pair it was given.  STRATA_NATIVE_REQUEST=1 switches it on, =0 off;
STRATA_PY_TOKENIZER=1 forces Python as for the rest of libstrata_text.
"""
from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path

ABI_VERSION = 1
DEFAULT_ON = False      # tools/test_native_request.py --bench: see the comment in converters()

_lib = None


def _bind(lib):
    c = ctypes
    lib.strata_request_abi_version.restype = c.c_int
    lib.strata_request_error.restype = c.c_char_p
    lib.strata_request_error_kind.restype = c.c_int
    lib.strata_request_convert.argtypes = [c.c_int, c.c_char_p, c.c_int64, c.c_int]
    lib.strata_request_convert.restype = c.c_int64
    lib.strata_request_result.argtypes = [c.c_void_p, c.c_int64]
    lib.strata_request_result.restype = c.c_int64
    return lib


def load(path=None):
    """The bound library, or None when it is not there (or lacks this API).  Cached."""
    global _lib
    if _lib is not None and path is None:
        return _lib
    from serve.native_text import find_library
    found = Path(path) if path else find_library()
    if found is None:
        return None
    try:
        lib = _bind(ctypes.CDLL(str(found)))
        if lib.strata_request_abi_version() != ABI_VERSION:
            return None
    except (OSError, AttributeError):
        return None
    _lib = lib
    return lib


def enabled() -> bool:
    if os.environ.get("STRATA_PY_TOKENIZER") == "1":
        return False
    want = os.environ.get("STRATA_NATIVE_REQUEST")
    if want == "0" or (want != "1" and not DEFAULT_ON):
        return False
    return load() is not None


def _run(lib, api, req, think_unasked, python):
    if type(req) is not dict:
        return python()
    try:
        data = json.dumps(req).encode("ascii")
    except (TypeError, ValueError, RecursionError):
        return python()
    n = lib.strata_request_convert(api, data, len(data), 1 if think_unasked else 0)
    if n < 0:
        if lib.strata_request_error_kind() == 1:
            raise ValueError((lib.strata_request_error() or b"").decode("utf-8", "replace")) from None
        return python()
    buf = ctypes.create_string_buffer(n)
    lib.strata_request_result(buf, n)
    out = json.loads(buf.raw)
    tools = out["tools"]
    if tools is not None:
        from serve.frontend import _stable_tool_order
        tools = _stable_tool_order(tools, lambda t: t["name"])
    return out["messages"], tools, out["kwargs"]


def converters(py_openai, py_anthropic, lib=None):
    """(openai_to_messages, anthropic_to_messages): the native pair, or the two given."""
    lib = lib or (load() if enabled() else None)
    if lib is None:
        return py_openai, py_anthropic

    def openai_to_messages(req):
        return _run(lib, 0, req, True, lambda: py_openai(req))

    def anthropic_to_messages(req, think_unasked=True):
        return _run(lib, 1, req, think_unasked, lambda: py_anthropic(req, think_unasked))

    openai_to_messages.__doc__, anthropic_to_messages.__doc__ = py_openai.__doc__, py_anthropic.__doc__
    return openai_to_messages, anthropic_to_messages
