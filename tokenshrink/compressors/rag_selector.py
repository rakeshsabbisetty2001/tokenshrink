from __future__ import annotations

import math
import re
from collections import Counter


_STOPWORDS = frozenset(
    "a an the is are was were be been being have has had do does did "
    "will would could should may might shall can i you he she it we they "
    "this that these those of in on at to for with by from as into about "
    "and or but not so yet nor both either neither just also".split()
)


def _tokenize(text: str) -> list[str]:
    return [w for w in re.findall(r"\b\w+\b", text.lower()) if w not in _STOPWORDS]


def _bm25_score(query_terms: list[str], doc_terms: list[str], avgdl: float, n_docs: int, df: dict[str, int], k1: float, b: float) -> float:
    tf = Counter(doc_terms)
    dl = len(doc_terms)
    score = 0.0
    for term in set(query_terms):
        if term not in tf:
            continue
        idf = math.log((n_docs - df.get(term, 0) + 0.5) / (df.get(term, 0) + 0.5) + 1)
        freq = tf[term]
        score += idf * (freq * (k1 + 1)) / (freq + k1 * (1 - b + b * dl / max(avgdl, 1)))
    return score


def select_chunks(chunks: list[str], query: str, max_tokens: int, counter, opts: dict) -> list[str]:
    """Score chunks with BM25 and greedily include the highest-scoring ones within the token budget."""
    if not chunks or not query:
        return chunks

    min_score = opts.get("min_score", 0.0)
    k1 = opts.get("bm25_k1", 1.5)
    b = opts.get("bm25_b", 0.75)
    query_terms = _tokenize(query)

    # Tokenize all docs
    doc_tokens = [_tokenize(c) for c in chunks]
    n_docs = len(chunks)
    avgdl = sum(len(d) for d in doc_tokens) / max(n_docs, 1)

    # Build document frequency table
    df: dict[str, int] = {}
    for doc in doc_tokens:
        for term in set(doc):
            df[term] = df.get(term, 0) + 1

    # Score each chunk
    scored = []
    for i, (chunk, doc) in enumerate(zip(chunks, doc_tokens)):
        score = _bm25_score(query_terms, doc, avgdl, n_docs, df, k1, b)
        scored.append((score, i, chunk))

    # Sort by score descending
    scored.sort(key=lambda x: x[0], reverse=True)

    # Greedily fill token budget
    selected: list[tuple[int, str]] = []
    used_tokens = 0
    for score, idx, chunk in scored:
        if score < min_score:
            continue
        chunk_tokens = counter.count(chunk)
        if used_tokens + chunk_tokens <= max_tokens:
            selected.append((idx, chunk))
            used_tokens += chunk_tokens
        if used_tokens >= max_tokens:
            break

    # Return in original order
    selected.sort(key=lambda x: x[0])
    return [c for _, c in selected]
