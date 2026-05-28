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

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from claude_code_integration.session_state import SessionState, approx_tokens, session_dir


CONTEXT_WINDOW = int(os.environ.get("TOKENSHRINK_CONTEXT_WINDOW", "180000"))
SHOW_THRESHOLD = float(os.environ.get("TOKENSHRINK_BUDGET_THRESHOLD", "0.15"))
COMPACT_THRESHOLD = 0.70


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
    state = SessionState.load(session_id)

    message = data.get("message", {})
    usage = (message.get("usage", {}) or {}) if isinstance(message, dict) else {}

    # Prefer API-reported token counts from the usage object
    api_output = usage.get("output_tokens")
    api_input = usage.get("input_tokens")

    if api_output is not None:
        turn_output = int(api_output)
    elif isinstance(message, dict):
        turn_output = approx_tokens(_extract_text(message.get("content", "")))
    else:
        turn_output = 0

    state.output_tokens += turn_output

    # input_tokens from the API represents the full context size for this call
    if api_input is not None:
        state.input_tokens = int(api_input)

    state.turns += 1

    # fill_fraction: use API input_tokens (most accurate) or fall back to cumulative output
    context_used = state.input_tokens if state.input_tokens else state.output_tokens
    state.fill_fraction = context_used / CONTEXT_WINDOW

    state.save(session_id)

    turns = state.turns
    fraction = state.fill_fraction

    # One-shot imperative /compact directive when context crosses 70%
    sdir = session_dir(session_id)
    compact_flag = sdir / "compact_fired.flag"
    if fraction >= COMPACT_THRESHOLD and not compact_flag.exists():
        compact_flag.touch()
        pct = min(100, int(fraction * 100))
        msg = (
            f"[TokenShrink] IMPORTANT: Context is at {pct}%. "
            "You must call /compact before taking any other action to prevent hitting the context limit."
        )
        print(json.dumps({"systemMessage": msg}))
        sys.exit(0)

    # Stay silent for very short sessions unless debug mode
    if fraction < SHOW_THRESHOLD and not os.environ.get("TOKENSHRINK_DEBUG"):
        sys.exit(0)

    remaining = max(0, CONTEXT_WINDOW - context_used)
    pct_used = min(100, fraction * 100)

    if pct_used >= 80:
        hint = " — ⚠ run /compact now"
    elif pct_used >= 60:
        hint = " — consider /compact"
    elif pct_used >= 40:
        hint = " — approaching mid-session"
    else:
        hint = ""

    label = "context" if state.input_tokens else "output tokens"
    status = (
        f"[TokenShrink] Turn {turns} | "
        f"~{_fmt(context_used)} {label} this session | "
        f"~{_fmt(remaining)} remaining{hint}"
    )

    print(json.dumps({"systemMessage": status}))


if __name__ == "__main__":
    main()
