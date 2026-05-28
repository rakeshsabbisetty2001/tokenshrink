from __future__ import annotations

import math
import re
from collections import Counter

from . import whitespace as ws
from . import deduplication as dedup
from . import semantic_dedup as sem
from .rag_selector import _STOPWORDS, _tokenize
from .semantic_dedup import _SENTENCE_SPLIT


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
    seen_sigs: list[list[int]] = []  # shared across all old turns for cross-turn dedup

    for msg in old:
        compressed = _compress_message(msg, seen_content, seen_sigs, opts)
        if compressed is not None:
            compressed_old.append(compressed)

    return compressed_old + recent


def _compress_message(
    msg: dict,
    seen_content: set[str],
    seen_sigs: list[list[int]],
    opts: dict,
) -> dict | None:
    content = msg.get("content", "")
    if isinstance(content, list):
        # Multi-part content blocks — compress each text block
        new_blocks = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                compressed_text = _compress_text(block["text"], seen_content, seen_sigs, opts)
                if compressed_text:
                    new_blocks.append({**block, "text": compressed_text})
            else:
                new_blocks.append(block)
        if not new_blocks:
            return None
        return {**msg, "content": new_blocks}
    elif isinstance(content, str):
        compressed = _compress_text(content, seen_content, seen_sigs, opts)
        if not compressed:
            return None
        return {**msg, "content": compressed}
    return msg


def _compress_text(
    text: str,
    seen_content: set[str],
    seen_sigs: list[list[int]],
    opts: dict,
) -> str:
    if not text or not text.strip():
        return text

    # Apply lossless pipeline
    text = ws.compress(text, opts)
    text = dedup.compress(text, opts)
    text = sem.compress_with_seen(text, seen_sigs, {**opts, "unit": "sentence"})

    # Optional lossy extractive summarization
    if opts.get("summarize_old_turns"):
        ratio = int(opts.get("summarize_ratio", 3))
        text = _extractive_summarize(text, ratio)

    return text.strip()


def _extractive_summarize(text: str, ratio: int = 3) -> str:
    """Keep top-K sentences by TF-IDF score, preserving their original order."""
    sentences = [s for s in _SENTENCE_SPLIT.split(text) if s.strip()]
    if len(sentences) <= 2:
        return text

    keep_n = max(2, math.ceil(len(sentences) / ratio))
    if keep_n >= len(sentences):
        return text

    # Build TF per sentence and DF across all sentences
    tokenized = [_tokenize(s) for s in sentences]
    df: Counter[str] = Counter()
    for terms in tokenized:
        df.update(set(terms))

    n = len(sentences)
    scores: list[float] = []
    for terms in tokenized:
        if not terms:
            scores.append(0.0)
            continue
        tf = Counter(terms)
        score = sum(
            (tf[t] / len(terms)) * math.log((n + 1) / (df[t] + 1))
            for t in set(terms)
        )
        scores.append(score)

    # Pick top-K indices in original order
    top_indices = sorted(sorted(range(n), key=lambda i: scores[i], reverse=True)[:keep_n])
    kept = [sentences[i] for i in top_indices]
    summary = " ".join(kept)
    return f"[Summary: {n}->{keep_n} sentences] {summary}"
