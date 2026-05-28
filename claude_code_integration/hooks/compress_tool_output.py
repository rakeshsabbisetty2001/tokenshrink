#!/usr/bin/env python3
"""
PostToolUse hook for Claude Code.

Compresses large tool outputs (Bash, Read, WebFetch, WebSearch) before they
are fed back to Claude, reducing token consumption from tool results.

Input:  JSON on stdin with tool_name, output (and other fields)
Output: JSON on stdout with hookSpecificOutput.updatedToolOutput (if compressed)
"""
from __future__ import annotations

import json
import sys
import os

# Allow running without installing the package (hook is called from anywhere)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from tokenshrink import TokenShrink

# Only compress outputs larger than this token threshold to avoid overhead
MIN_TOKENS_TO_COMPRESS = int(os.environ.get("TOKENSHRINK_MIN_TOKENS", "200"))

# Tools whose outputs we compress
COMPRESS_TOOLS = {"Bash", "Read", "WebFetch", "WebSearch"}


def main():
    try:
        data = json.loads(sys.stdin.read())
    except (json.JSONDecodeError, ValueError):
        sys.exit(0)

    tool_name = data.get("tool_name", "")
    if tool_name not in COMPRESS_TOOLS:
        sys.exit(0)

    output = data.get("output", "")
    if not isinstance(output, str) or not output.strip():
        sys.exit(0)

    ts = TokenShrink()
    original_tokens = ts.count_tokens(output)

    if original_tokens < MIN_TOKENS_TO_COMPRESS:
        sys.exit(0)

    result = ts.compress_prompt(output)

    if result.compressed_tokens >= original_tokens:
        # No benefit — don't replace
        sys.exit(0)

    response = {
        "hookSpecificOutput": {
            "updatedToolOutput": result.compressed_text
        }
    }

    # Optionally surface stats as a system message in debug mode
    if os.environ.get("TOKENSHRINK_DEBUG"):
        saved = original_tokens - result.compressed_tokens
        response["systemMessage"] = (
            f"[TokenShrink] {tool_name} output: {original_tokens} → {result.compressed_tokens} tokens "
            f"({result.reduction_pct:.0f}% reduction, {saved} tokens saved)"
        )

    print(json.dumps(response))


if __name__ == "__main__":
    main()
