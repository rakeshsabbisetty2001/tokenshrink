#!/usr/bin/env python3
"""
PreToolUse hook for Claude Code.

Injects a brevity nudge into Claude's context once per session — before the
first tool call — so Claude skips narration, prefers bullets, and batches
related tool calls rather than interleaving them with commentary.

Input:  JSON on stdin — tool_name, tool_input, session_id, ...
Output: JSON on stdout — systemMessage (once per session, then silent)

Set TOKENSHRINK_TERSE=0 to disable without uninstalling.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path


NUDGE = (
    "[TokenShrink] Efficiency hints for this session: "
    "skip step-by-step narration; prefer bullet points over paragraphs; "
    "batch related tool calls into one response block when possible."
)


def _session_dir(session_id: str) -> Path:
    key = hashlib.md5(session_id.encode()).hexdigest()[:8]
    d = Path(tempfile.gettempdir()) / f"tokenshrink_{key}"
    d.mkdir(exist_ok=True)
    return d


def main() -> None:
    if os.environ.get("TOKENSHRINK_TERSE", "1") == "0":
        sys.exit(0)

    try:
        data = json.loads(sys.stdin.read())
    except (json.JSONDecodeError, ValueError):
        sys.exit(0)

    session_id = data.get("session_id", "default")
    sdir = _session_dir(session_id)
    flag = sdir / "terse_injected.flag"

    if flag.exists():
        sys.exit(0)

    flag.touch()
    print(json.dumps({"systemMessage": NUDGE}))


if __name__ == "__main__":
    main()
