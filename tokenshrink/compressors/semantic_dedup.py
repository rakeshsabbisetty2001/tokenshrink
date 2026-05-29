from __future__ import annotations

import re
import random
from math import log


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")

_STOPWORDS_PROSE = frozenset(
    "a an the is are was were be been being have has had do does did "
    "will would could should may might shall can i you he she it we they "
    "this that these those of in on at to for with by from as into about "
    "and or but not so yet nor both either neither just also only even "
    "all any each few more most other some such no nor not only same so "
    "than too very s t just don don't doesn doesn't didn didn't isn isn't "
    "aren aren't wasn wasn't weren weren't won won't wouldn wouldn't".split()
)

# Common keywords across Python, JavaScript, TypeScript, SQL — treated as
# stopwords so code-aware dedup focuses on identifiers and literals instead.
_STOPWORDS_CODE = frozenset(
    # Python
    "def class return import from as if elif else for while with try except "
    "finally raise pass break continue lambda yield async await True False None "
    "and or not in is del global nonlocal assert "
    # JavaScript / TypeScript
    "function var let const export default new typeof instanceof void "
    "switch case this super extends implements interface enum abstract "
    "public private protected readonly static "
    # SQL
    "select from where join inner outer left right on group by order having "
    "insert into values update set delete create table drop alter index view "
    "distinct count sum avg min max between like exists union all ".split()
)

# Detect whether text looks like code (heuristic: starts with keywords or
# contains assignment / function-call patterns).
_CODE_SIGNAL = re.compile(
    r"^\s*(def |class |import |from |function |const |let |var |select |insert |update )",
    re.IGNORECASE | re.MULTILINE,
)

_STOPWORDS = _STOPWORDS_PROSE  # default; overridden per-call based on language opt


def _choose_stopwords(text: str, language: str) -> frozenset:
    if language == "code":
        return _STOPWORDS_PROSE | _STOPWORDS_CODE
    if language == "prose":
        return _STOPWORDS_PROSE
    # "auto": peek at the text
    if _CODE_SIGNAL.search(text):
        return _STOPWORDS_PROSE | _STOPWORDS_CODE
    return _STOPWORDS_PROSE

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
    seen_sigs: list[list[int]] = []
    return compress_with_seen(text, seen_sigs, opts)


def compress_with_seen(text: str, seen_sigs: list[list[int]], opts: dict) -> str:
    """Like compress(), but shares seen_sigs with the caller for cross-text deduplication."""
    if not isinstance(text, str):
        return str(text) if text is not None else ""
    if not text:
        return text

    threshold = opts.get("similarity_threshold", 0.85)
    unit = opts.get("unit", "sentence")  # "sentence" or "paragraph"
    language = opts.get("language", "auto")  # "auto", "prose", or "code"
    stopwords = _choose_stopwords(text, language)

    if unit == "paragraph":
        units = re.split(r"\n{2,}", text)
        separator = "\n\n"
    else:
        units = _split_sentences(text)
        separator = " "

    kept: list[str] = []

    for unit_text in units:
        stripped = unit_text.strip()
        if not stripped:
            kept.append(unit_text)
            continue
        content_words = [w for w in re.findall(r"\b\w+\b", stripped.lower()) if w not in stopwords]
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
