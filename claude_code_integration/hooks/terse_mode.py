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

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from claude_code_integration.session_state import session_dir


NUDGE = (
    "[TokenShrink] Efficiency hints for this session: "
    "skip step-by-step narration; prefer bullet points over paragraphs; "
    "batch related tool calls into one response block when possible."
)


def main() -> None:
    if os.environ.get("TOKENSHRINK_TERSE", "1") == "0":
        sys.exit(0)

    try:
        data = json.loads(sys.stdin.read())
    except (json.JSONDecodeError, ValueError):
        sys.exit(0)

    session_id = data.get("session_id", "default")
    sdir = session_dir(session_id)
    flag = sdir / "terse_injected.flag"

    if flag.exists():
        sys.exit(0)

    flag.touch()
    print(json.dumps({"systemMessage": NUDGE}))


if __name__ == "__main__":
    main()
