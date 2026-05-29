from __future__ import annotations

import hashlib
import re


def compress(text: str, opts: dict) -> str:
    """Remove verbatim duplicate paragraphs/blocks using SHA-256 hashing."""
    if not isinstance(text, str):
        return str(text) if text is not None else ""
    if not text:
        return text

    min_block_tokens = opts.get("min_block_tokens", 5)
    blocks = _split_blocks(text)
    seen: set[str] = set()
    output: list[str] = []

    for block in blocks:
        stripped = block.strip()
        if not stripped:
            output.append(block)
            continue
        word_count = len(stripped.split())
        if word_count < min_block_tokens:
            # Too short to bother deduplicating — always keep
            output.append(block)
            continue
        digest = hashlib.sha256(stripped.encode()).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        output.append(block)

    return _join_blocks(output)


def _split_blocks(text: str) -> list[str]:
    """Split on double newlines (paragraph boundaries) or XML-tag boundaries."""
    # Also split on common RAG separators like ---
    return re.split(r"(\n{2,}|(?<=\n)---+\n)", text)


def _join_blocks(blocks: list[str]) -> str:
    result = "".join(blocks)
    # Clean up consecutive blank lines that may arise from removals
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()
