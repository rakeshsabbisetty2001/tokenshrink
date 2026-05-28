#!/usr/bin/env python3
"""
Stop hook for Claude Code.

After each assistant response, estimates cumulative session output tokens and
injects a budget status line into the next turn so both Claude and the user
have a real-time sense of context consumption.

Input:  JSON on stdin — session_id, stop_reason, message, ...
Output: JSON on stdout — systemMessage with token budget status

Set TOKENSHRINK_CONTEXT_WINDOW to override the assumed context size (default: 180000).
Set TOKENSHRINK_BUDGET=0 to disable without uninstalling.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path


CONTEXT_WINDOW = int(os.environ.get("TOKENSHRINK_CONTEXT_WINDOW", "180000"))
# Show budget status once output token usage exceeds this fraction of context
SHOW_THRESHOLD = float(os.environ.get("TOKENSHRINK_BUDGET_THRESHOLD", "0.15"))


def _session_dir(session_id: str) -> Path:
    key = hashlib.md5(session_id.encode()).hexdigest()[:8]
    d = Path(tempfile.gettempdir()) / f"tokenshrink_{key}"
    d.mkdir(exist_ok=True)
    return d


def _approx_tokens(text: str) -> int:
    return math.ceil(len(text.split()) * 1.3)


def _extract_text(content: object) -> str:
    """Pull plain text out of an Anthropic-style content field."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return " ".join(parts)
    return ""


def _fmt(n: int) -> str:
    return f"{n / 1000:.1f}k" if n >= 1000 else str(n)


def main() -> None:
    if os.environ.get("TOKENSHRINK_BUDGET", "1") == "0":
        sys.exit(0)

    try:
        data = json.loads(sys.stdin.read())
    except (json.JSONDecodeError, ValueError):
        sys.exit(0)

    session_id = data.get("session_id", "default")
    sdir = _session_dir(session_id)
    budget_file = sdir / "budget.json"

    # Load existing tally
    tally: dict = {"output_tokens": 0, "turns": 0}
    if budget_file.exists():
        try:
            tally = json.loads(budget_file.read_text())
        except Exception:
            pass

    # Add this turn's output tokens (assistant message content)
    message = data.get("message", {})
    if isinstance(message, dict):
        turn_text = _extract_text(message.get("content", ""))
        tally["output_tokens"] = tally.get("output_tokens", 0) + _approx_tokens(turn_text)

    tally["turns"] = tally.get("turns", 0) + 1
    budget_file.write_text(json.dumps(tally))

    output_tokens = tally["output_tokens"]
    turns = tally["turns"]
    fraction = output_tokens / CONTEXT_WINDOW

    # Stay silent for very short sessions unless debug mode
    if fraction < SHOW_THRESHOLD and not os.environ.get("TOKENSHRINK_DEBUG"):
        sys.exit(0)

    remaining = max(0, CONTEXT_WINDOW - output_tokens)
    pct_used = min(100, fraction * 100)

    if pct_used >= 80:
        hint = " — ⚠ run /compact now"
    elif pct_used >= 60:
        hint = " — consider /compact"
    elif pct_used >= 40:
        hint = " — approaching mid-session"
    else:
        hint = ""

    status = (
        f"[TokenShrink] Turn {turns} | "
        f"~{_fmt(output_tokens)} output tokens this session | "
        f"~{_fmt(remaining)} remaining{hint}"
    )

    print(json.dumps({"systemMessage": status}))


if __name__ == "__main__":
    main()
