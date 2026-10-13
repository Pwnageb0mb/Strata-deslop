"""serve/mcp_fake_server.py - a small MCP server for serve/test_mcp.py (no MCP SDK, no network beyond localhost).

    python serve/mcp_fake_server.py                  stdio: JSON-RPC lines on stdin/stdout

Tools: echo (returns its text), fail (a tool error, isError), sleep (answers after `seconds`), big (`n` characters),
die (the process exits mid-call), add (two integers).  tools/list pages two tools at a time (`--page N`), so the
client's pagination is exercised.  `--crash-at-start` exits before answering anything.  With FAKE_MCP_LOG set, the
server appends what it saw (calls, cancellations) to that file, one JSON line each.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time

TOOLS = [
    {"name": "echo", "description": "Returns the text it is given.",
     "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}},
    {"name": "fail", "description": "Always fails.", "inputSchema": {"type": "object", "properties": {}}},
    {"name": "sleep", "description": "Waits, then answers.",
     "inputSchema": {"type": "object", "properties": {"seconds": {"type": "number"}}}},
    {"name": "big", "description": "Returns n characters.",
     "inputSchema": {"type": "object", "properties": {"n": {"type": "integer"}}}},
    {"name": "die", "description": "Ends the server.", "inputSchema": {"type": "object", "properties": {}}},
    {"name": "add", "description": "Adds two integers.",
     "inputSchema": {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}}},
]
PAGE = 2
cancelled: set = set()


def log(event: dict):
    path = os.environ.get("FAKE_MCP_LOG")
    if path:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")


def text(t, error=False):
    return {"content": [{"type": "text", "text": t}], "isError": error}


def handle(msg: dict):
    """-> the response for a request, None for a notification."""
    method, rid, params = msg.get("method"), msg.get("id"), msg.get("params") or {}
    if "id" not in msg:
        if method == "notifications/cancelled":
            cancelled.add(params.get("requestId"))
            log({"cancelled": params.get("requestId")})
        return None
    if method == "initialize":
        return {"jsonrpc": "2.0", "id": rid, "result": {
            "protocolVersion": params.get("protocolVersion", "2025-06-18"), "capabilities": {"tools": {}},
            "serverInfo": {"name": "fake", "version": "0.1"}}}
    if method == "tools/list":
        start = int(params.get("cursor") or 0)
        page = {"tools": TOOLS[start:start + PAGE]}
        if start + PAGE < len(TOOLS):
            page["nextCursor"] = str(start + PAGE)
        return {"jsonrpc": "2.0", "id": rid, "result": page}
    if method == "tools/call":
        name, args = params.get("name"), params.get("arguments") or {}
        log({"call": name, "arguments": args})
        if name == "echo":
            result = text(str(args.get("text", "")))
        elif name == "fail":
            result = text("it failed on purpose", error=True)
        elif name == "sleep":
            time.sleep(float(args.get("seconds", 1)))
            result = text("slept")
        elif name == "big":
            result = text("y" * int(args.get("n", 100000)))
        elif name == "die":
            sys.stderr.write("dying on purpose\n")
            sys.stderr.flush()
            os._exit(3)
        elif name == "add":
            result = text(str(int(args.get("a", 0)) + int(args.get("b", 0))))
        else:
            return {"jsonrpc": "2.0", "id": rid, "error": {"code": -32602, "message": f"unknown tool {name}"}}
        return {"jsonrpc": "2.0", "id": rid, "result": result}
    return {"jsonrpc": "2.0", "id": rid, "error": {"code": -32601, "message": f"no method {method}"}}


def stdio():
    for f in (sys.stdin, sys.stdout, sys.stderr):      # MCP's stdio is UTF-8, whatever the Windows code page is
        f.reconfigure(encoding="utf-8")
    out = threading.Lock()

    def answer(msg):
        r = handle(msg)
        if r is not None and msg.get("id") not in cancelled:
            with out:
                sys.stdout.write(json.dumps(r) + "\n")
                sys.stdout.flush()

    for line in sys.stdin:
        if not line.strip():
            continue
        msg = json.loads(line)
        if msg.get("method") == "tools/call":           # calls on threads: a slow one must not block cancellation
            threading.Thread(target=answer, args=(msg,), daemon=True).start()
        else:
            answer(msg)


if __name__ == "__main__":
    if "--page" in sys.argv:
        PAGE = int(sys.argv[sys.argv.index("--page") + 1])
    if "--crash-at-start" in sys.argv:
        sys.stderr.write("cannot start: missing configuration\n")
        sys.exit(2)
    stdio()
