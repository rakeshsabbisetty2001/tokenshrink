from __future__ import annotations

from . import whitespace as ws
from . import deduplication as dedup
from . import semantic_dedup as sem


def compress_messages(
    messages: list[dict],
    keep_recent: int,
    counter,
    opts: dict,
) -> list[dict]:
    """Compress conversation history, keeping the N most recent turns verbatim."""
    if len(messages) <= keep_recent:
        return messages

    recent = messages[-keep_recent:]
    old = messages[:-keep_recent]

    compressed_old = []
    seen_content: set[str] = set()

    for msg in old:
        compressed = _compress_message(msg, seen_content, opts)
        if compressed is not None:
            compressed_old.append(compressed)

    return compressed_old + recent


def _compress_message(msg: dict, seen_content: set[str], opts: dict) -> dict | None:
    content = msg.get("content", "")
    if isinstance(content, list):
        # Multi-part content blocks — compress each text block
        new_blocks = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                compressed_text = _compress_text(block["text"], seen_content, opts)
                if compressed_text:
                    new_blocks.append({**block, "text": compressed_text})
            else:
                new_blocks.append(block)
        if not new_blocks:
            return None
        return {**msg, "content": new_blocks}
    elif isinstance(content, str):
        compressed = _compress_text(content, seen_content, opts)
        if not compressed:
            return None
        return {**msg, "content": compressed}
    return msg


def _compress_text(text: str, seen_content: set[str], opts: dict) -> str:
    if not text or not text.strip():
        return text

    # Apply lossless pipeline
    text = ws.compress(text, opts)
    text = dedup.compress(text, opts)
    text = sem.compress(text, {**opts, "unit": "sentence"})

    return text.strip()
