#!/usr/bin/env python3
"""
PreToolUse hook for Claude Code.

Intercepts Bash tool calls and rewrites budget-expensive commands to
cheaper equivalents based on the current context fill fraction.

Input:  JSON on stdin  — tool_name, tool_input (with "command" key), session_id, ...
Output: JSON on stdout — hookSpecificOutput.updatedInput + systemMessage (if rewritten)
        Silent (no output) if no rewrite applies.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from claude_code_integration.command_rules import rewrite_command
from claude_code_integration.session_state import SessionState


def main() -> None:
    try:
        data = json.loads(sys.stdin.read())
    except (json.JSONDecodeError, ValueError):
        sys.exit(0)

    if data.get("tool_name") != "Bash":
        sys.exit(0)

    tool_input = data.get("tool_input") or {}
    command = tool_input.get("command", "")
    if not command:
        sys.exit(0)

    session_id = data.get("session_id", "default")
    fill = SessionState.load(session_id).fill_fraction

    rewritten, note = rewrite_command(command, fill)
    if note is None:
        sys.exit(0)

    response: dict = {
        "hookSpecificOutput": {
            "updatedInput": {**tool_input, "command": rewritten}
        },
        "systemMessage": f"[TokenShrink] Command {note}",
    }
    print(json.dumps(response))


if __name__ == "__main__":
    main()
