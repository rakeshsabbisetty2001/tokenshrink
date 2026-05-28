#!/usr/bin/env python3
"""
UserPromptSubmit hook for Claude Code.

Compresses the user's prompt before it reaches Claude, removing redundant
whitespace and near-duplicate content.

Input:  JSON on stdin with user_prompt (and other fields)
Output: JSON on stdout with updatedInput.user_prompt (if compressed)
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from tokenshrink import TokenShrink
from claude_code_integration.session_state import SessionState


def _compression_profile(fill: float) -> tuple[list[str], int]:
    """Return (techniques, min_tokens) for user prompts based on context fill."""
    if fill >= 0.70:
        return ["whitespace", "deduplication", "format", "semantic_dedup"], 50
    if fill >= 0.40:
        return ["whitespace", "deduplication", "semantic_dedup"], 75
    return ["whitespace", "semantic_dedup"], 100


def main():
    try:
        data = json.loads(sys.stdin.read())
    except (json.JSONDecodeError, ValueError):
        sys.exit(0)

    user_prompt = data.get("user_prompt", "")
    if not isinstance(user_prompt, str) or not user_prompt.strip():
        sys.exit(0)

    session_id = data.get("session_id", "default")
    fill = SessionState.load(session_id).fill_fraction
    techniques, min_tokens = _compression_profile(fill)

    ts = TokenShrink(techniques=techniques)

    original_tokens = ts.count_tokens(user_prompt)
    if original_tokens < min_tokens:
        sys.exit(0)

    result = ts.compress_prompt(user_prompt)

    if result.compressed_tokens >= original_tokens:
        sys.exit(0)

    response = {
        "updatedInput": {
            "user_prompt": result.compressed_text
        }
    }

    if os.environ.get("TOKENSHRINK_DEBUG"):
        saved = original_tokens - result.compressed_tokens
        response["systemMessage"] = (
            f"[TokenShrink] Prompt: {original_tokens} → {result.compressed_tokens} tokens "
            f"({result.reduction_pct:.0f}% reduction, {saved} saved)"
        )

    print(json.dumps(response))


if __name__ == "__main__":
    main()
