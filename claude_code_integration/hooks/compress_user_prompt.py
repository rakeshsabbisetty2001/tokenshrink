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
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from tokenshrink import TokenShrink

# Only compress prompts larger than this threshold
MIN_TOKENS_TO_COMPRESS = int(os.environ.get("TOKENSHRINK_PROMPT_MIN_TOKENS", "100"))


def main():
    try:
        data = json.loads(sys.stdin.read())
    except (json.JSONDecodeError, ValueError):
        sys.exit(0)

    user_prompt = data.get("user_prompt", "")
    if not isinstance(user_prompt, str) or not user_prompt.strip():
        sys.exit(0)

    ts = TokenShrink(
        # For user prompts: only apply whitespace + semantic dedup (safest techniques)
        techniques=["whitespace", "semantic_dedup"]
    )

    original_tokens = ts.count_tokens(user_prompt)
    if original_tokens < MIN_TOKENS_TO_COMPRESS:
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
