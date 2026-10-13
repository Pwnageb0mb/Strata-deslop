"""serve/native_text.py - the server's tokenizer, in C++ (libstrata_text, include/strata/text/strata_text.h).

NativeTokenizer has the interface the server uses from tools/strata_tokenizer.Tokenizer:
encode(text, parse_special=False, plain=()), decode(ids, errors="replace"), token_bytes(i), .tokens (a list-like of
the vocabulary strings, built on first use) and .control_tokens.  `plain` spans are Python str (code point)
offsets, as for the Python tokenizer; they are converted to the UTF-8 byte offsets the library takes.
"""
from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIB_NAME = "strata_text.dll" if os.name == "nt" else ("libstrata_text.dylib" if sys.platform == "darwin"
                                                    else "libstrata_text.so")
ABI_VERSION = 1


class NativeTextError(RuntimeError):
    pass


def find_library(engine_exe: str | os.PathLike | None = None) -> Path | None:
    """$STRATA_TEXT_LIB, else beside the engine binary, else ROOT/build/.  None when none of them exists."""
    env = os.environ.get("STRATA_TEXT_LIB")
    if env:
        return Path(env) if Path(env).is_file() else None
    cands = []
    if engine_exe:
        cands.append(Path(engine_exe).resolve().parent / LIB_NAME)
    for d in ("engine", "engine-cuda12"):
        cands.append(ROOT / d / LIB_NAME)
    cands.append(ROOT / "build" / LIB_NAME)
    cands.append(ROOT / "build" / "Release" / LIB_NAME)
    for c in cands:
        if c.is_file():
            return c
    return None


def _bind(lib: ctypes.CDLL) -> ctypes.CDLL:
    c = ctypes
    lib.strata_text_abi_version.restype = c.c_int
    lib.strata_text_error.restype = c.c_char_p
    lib.strata_text_load.argtypes = [c.c_char_p]
    lib.strata_text_load.restype = c.c_void_p
    lib.strata_text_free.argtypes = [c.c_void_p]
    lib.strata_text_free.restype = None
    lib.strata_text_vocab_size.argtypes = [c.c_void_p]
    lib.strata_text_vocab_size.restype = c.c_int64
    lib.strata_text_encode.argtypes = [c.c_void_p, c.c_char_p, c.c_int64, c.c_int, c.c_void_p, c.c_int64,
                                       c.c_void_p, c.c_int64]
    lib.strata_text_encode.restype = c.c_int64
    lib.strata_text_export_tokens.argtypes = [c.c_void_p, c.c_int, c.c_void_p, c.c_int64, c.c_void_p]
    lib.strata_text_export_tokens.restype = c.c_int64
    lib.strata_text_control_tokens.argtypes = [c.c_void_p, c.c_void_p, c.c_int64]
    lib.strata_text_control_tokens.restype = c.c_int64
    return lib


class NativeTokenizer:
    def __init__(self, tokenizer_dir, lib_path: str | os.PathLike | None = None, engine_exe=None):
        path = Path(lib_path) if lib_path else find_library(engine_exe)
        if path is None:
            raise NativeTextError(f"{LIB_NAME} not found (looked at $STRATA_TEXT_LIB, beside the engine, and in "
                                  f"{ROOT / 'build'}); build it with: cmake --build build --target strata_text")
        try:
            self._lib = _bind(ctypes.CDLL(str(path)))
        except (OSError, AttributeError) as e:
            raise NativeTextError(f"cannot load {path}: {e}") from e
        if self._lib.strata_text_abi_version() != ABI_VERSION:
            raise NativeTextError(f"{path} has ABI version {self._lib.strata_text_abi_version()}, "
                                  f"this server needs {ABI_VERSION}; rebuild it")
        self._h = self._lib.strata_text_load(os.fsencode(str(tokenizer_dir)))
        if not self._h:
            raise NativeTextError(f"cannot load the tokenizer in {tokenizer_dir}: {self._err()}")
        self.path = path
        self._n = int(self._lib.strata_text_vocab_size(self._h))
        self._table = self._blob_split(True)                       # raw bytes of every token, once
        self._tokens = None
        self.control_tokens = self._control()

    def _err(self) -> str:
        return (self._lib.strata_text_error() or b"").decode("utf-8", "replace")

    def _blob_split(self, raw: bool) -> list:
        offs = (ctypes.c_int64 * (self._n + 1))()
        size = self._lib.strata_text_export_tokens(self._h, int(raw), None, 0, offs)
        if size < 0:
            raise NativeTextError(self._err())
        buf = ctypes.create_string_buffer(max(size, 1))
        if self._lib.strata_text_export_tokens(self._h, int(raw), buf, size, offs) != size:
            raise NativeTextError(self._err())
        blob = buf.raw[:size]
        o = list(offs)
        if raw:
            self.raw_blob, self.raw_offsets = buf, offs     # kept for native_stream's pipe, which reads them in C
            return [blob[o[i]:o[i + 1]] for i in range(self._n)]
        return [blob[o[i]:o[i + 1]].decode("utf-8") for i in range(self._n)]

    def _control(self) -> list:
        size = self._lib.strata_text_control_tokens(self._h, None, 0)
        if size < 0:
            raise NativeTextError(self._err())
        if size == 0:
            return []
        buf = ctypes.create_string_buffer(size)
        self._lib.strata_text_control_tokens(self._h, buf, size)
        return buf.raw[:size].decode("utf-8").split("\n")

    @property
    def tokens(self) -> list:
        """The vocabulary strings (byte-level alphabet), like the Python tokenizer's; built on first use."""
        if self._tokens is None:
            self._tokens = self._blob_split(False)
        return self._tokens

    def __len__(self) -> int:
        return self._n

    def encode(self, text: str, parse_special: bool = False, plain=()) -> list[int]:
        data = text.encode("utf-8")
        spans = list(plain) if plain else []
        arr, n_spans = None, 0
        if spans:
            if not text.isascii():
                # code point offsets -> byte offsets (a prefix's UTF-8 length)
                cache: dict[int, int] = {}

                def to_byte(o: int) -> int:
                    o = max(0, min(o, len(text)))
                    if o not in cache:
                        cache[o] = len(text[:o].encode("utf-8"))
                    return cache[o]
                spans = [(to_byte(a), to_byte(b)) for a, b in spans]
            n_spans = len(spans)
            arr = (ctypes.c_int64 * (2 * n_spans))(*[x for ab in spans for x in ab])
        cap = len(data) + 1                                 # an id covers at least one input byte
        out = (ctypes.c_int32 * cap)()
        n = self._lib.strata_text_encode(self._h, data, len(data), 1 if parse_special else 0, arr, n_spans, out, cap)
        if n < 0:
            raise NativeTextError(self._err())
        if n > cap:
            out = (ctypes.c_int32 * n)()
            n = self._lib.strata_text_encode(self._h, data, len(data), 1 if parse_special else 0, arr, n_spans, out, n)
            if n < 0:
                raise NativeTextError(self._err())
        return out[:n]

    def token_bytes(self, i: int) -> bytes:
        if i < 0 or i >= self._n:
            raise IndexError("token id %d is outside the vocabulary (%d)" % (i, self._n))
        return self._table[i]

    def decode(self, ids, errors: str = "replace") -> str:
        t = self._table
        n = self._n
        for i in ids:
            if i < 0 or i >= n:
                raise IndexError("token id %d is outside the vocabulary (%d)" % (i, n))
        return b"".join([t[i] for i in ids]).decode("utf-8", errors=errors)

    def close(self):
        h, self._h = getattr(self, "_h", None), None
        if h:
            self._lib.strata_text_free(h)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
