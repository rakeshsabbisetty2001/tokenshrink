from __future__ import annotations

import re
import random
from math import log


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
_STOPWORDS = frozenset(
    "a an the is are was were be been being have has had do does did "
    "will would could should may might shall can i you he she it we they "
    "this that these those of in on at to for with by from as into about "
    "and or but not so yet nor both either neither just also only even "
    "all any each few more most other some such no nor not only same so "
    "than too very s t just don don't doesn doesn't didn didn't isn isn't "
    "aren aren't wasn wasn't weren weren't won won't wouldn wouldn't".split()
)

_NUM_HASH_FUNCTIONS = 128
_MINHASH_SEEDS = [random.Random(i).randint(0, 2**31) for i in range(_NUM_HASH_FUNCTIONS)]


def _shingles(text: str, k: int = 5) -> set[str]:
    normalized = re.sub(r"\s+", " ", text.lower().strip())
    return {normalized[i : i + k] for i in range(max(0, len(normalized) - k + 1))}


def _minhash(shingle_set: set[str]) -> list[int]:
    if not shingle_set:
        return [0] * _NUM_HASH_FUNCTIONS
    sig = []
    for seed in _MINHASH_SEEDS:
        min_val = min(hash(s + str(seed)) & 0x7FFFFFFF for s in shingle_set)
        sig.append(min_val)
    return sig


def _jaccard_approx(sig_a: list[int], sig_b: list[int]) -> float:
    matches = sum(1 for a, b in zip(sig_a, sig_b) if a == b)
    return matches / len(sig_a)


def compress(text: str, opts: dict) -> str:
    """Remove near-duplicate sentences using MinHash + Jaccard similarity."""
    if not text:
        return text

    threshold = opts.get("similarity_threshold", 0.85)
    unit = opts.get("unit", "sentence")  # "sentence" or "paragraph"

    if unit == "paragraph":
        units = re.split(r"\n{2,}", text)
        separator = "\n\n"
    else:
        units = _split_sentences(text)
        separator = " "

    seen_sigs: list[list[int]] = []
    kept: list[str] = []

    for unit_text in units:
        stripped = unit_text.strip()
        if not stripped:
            kept.append(unit_text)
            continue
        content_words = [w for w in re.findall(r"\b\w+\b", stripped.lower()) if w not in _STOPWORDS]
        if len(content_words) < 5:
            kept.append(unit_text)
            continue

        shingle_set = _shingles(stripped)
        sig = _minhash(shingle_set)

        is_dup = any(_jaccard_approx(sig, seen) >= threshold for seen in seen_sigs)
        if not is_dup:
            seen_sigs.append(sig)
            kept.append(unit_text)

    if unit == "paragraph":
        return separator.join(k for k in kept if k.strip())
    else:
        return _rejoin_sentences(kept)


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(\n+)", text)
    result = []
    for part in parts:
        if "\n" in part:
            result.append(part)
        else:
            sentences = _SENTENCE_SPLIT.split(part)
            result.extend(sentences)
    return result


def _rejoin_sentences(parts: list[str]) -> str:
    result = []
    for part in parts:
        if not part:
            continue
        if result and not result[-1].endswith("\n") and not part.startswith("\n"):
            result.append(" ")
        result.append(part)
    return "".join(result).strip()
