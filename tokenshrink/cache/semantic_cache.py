"""Semantic response cache backed by SQLite with SimHash similarity detection."""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
from pathlib import Path
from typing import Any


# ── Pure-Python SimHash ───────────────────────────────────────────────────────

_BITS = 64
_MASK = (1 << _BITS) - 1


def _shingles(text: str, k: int = 3) -> list[str]:
    tokens = re.findall(r"\b\w+\b", text.lower())
    return [" ".join(tokens[i : i + k]) for i in range(max(len(tokens) - k + 1, 1))]


def simhash(text: str) -> int:
    """Return a signed 64-bit SimHash of *text* (safe for SQLite INTEGER storage)."""
    v = [0] * _BITS
    for shingle in _shingles(text):
        h = int(hashlib.md5(shingle.encode()).hexdigest(), 16) & _MASK
        for i in range(_BITS):
            if h & (1 << i):
                v[i] += 1
            else:
                v[i] -= 1
    result = 0
    for i in range(_BITS):
        if v[i] > 0:
            result |= 1 << i
    # Convert unsigned 64-bit to signed 64-bit for SQLite compatibility
    if result >= (1 << 63):
        result -= (1 << 64)
    return result


def hamming_similarity(a: int, b: int) -> float:
    """Return similarity in [0, 1] based on Hamming distance of two SimHashes."""
    # Mask to unsigned to ensure correct bit counting regardless of sign
    xor = (a ^ b) & _MASK
    distance = bin(xor).count("1")
    return 1.0 - distance / _BITS


# ── Schema ────────────────────────────────────────────────────────────────────

_DDL = """
CREATE TABLE IF NOT EXISTS cache_entries (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    simhash     INTEGER NOT NULL,
    model       TEXT    NOT NULL,
    prompt_hash TEXT    NOT NULL,
    response    TEXT    NOT NULL,
    created_at  REAL    NOT NULL,
    ttl_seconds INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_model_simhash ON cache_entries (model, simhash);
"""

_DEFAULT_DB = Path.home() / ".tokenshrink" / "cache.db"


# ── SemanticCache ─────────────────────────────────────────────────────────────

class SemanticCache:
    """
    Caches LLM responses keyed by SimHash of the (compressed) prompt text.

    Args:
        backend: Only "sqlite" is supported currently.
        path:    SQLite database path. Defaults to ~/.tokenshrink/cache.db.
        ttl:     Time-to-live in seconds for cache entries. Default 3600.
        threshold: Minimum SimHash similarity to count as a cache hit [0, 1].
                   Default 0.95 (very conservative).
    """

    def __init__(
        self,
        backend: str = "sqlite",
        path: Path | str | None = None,
        ttl: int = 3600,
        threshold: float = 0.95,
    ) -> None:
        if backend != "sqlite":
            raise ValueError(f"Unsupported cache backend: {backend!r}")
        db_path = Path(path) if path else _DEFAULT_DB
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db_path = db_path
        self._ttl = ttl
        self._threshold = threshold
        self._hits = 0
        self._misses = 0
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.executescript(_DDL)
        self._conn.commit()

    # ── Public API ─────────────────────────────────────────────────────────────

    def get(self, prompt_text: str, model: str) -> Any | None:
        """Return a cached response dict/object, or None on a miss."""
        sh = simhash(prompt_text)
        now = time.time()
        cur = self._conn.execute(
            "SELECT simhash, response, created_at, ttl_seconds "
            "FROM cache_entries WHERE model = ? ORDER BY created_at DESC LIMIT 200",
            (model,),
        )
        for row_sh, response_json, created_at, ttl in cur:
            if now - created_at > ttl:
                continue
            if hamming_similarity(sh, row_sh) >= self._threshold:
                self._hits += 1
                return json.loads(response_json)
        self._misses += 1
        return None

    def set(self, prompt_text: str, model: str, response: Any) -> None:
        """Store a response in the cache."""
        sh = simhash(prompt_text)
        prompt_hash = hashlib.sha256(prompt_text.encode()).hexdigest()
        self._conn.execute(
            "INSERT INTO cache_entries (simhash, model, prompt_hash, response, created_at, ttl_seconds) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (sh, model, prompt_hash, json.dumps(response), time.time(), self._ttl),
        )
        self._conn.commit()

    def stats(self) -> dict:
        """Return hit rate, total entries, and approximate tokens saved."""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total else 0.0
        row = self._conn.execute("SELECT COUNT(*) FROM cache_entries").fetchone()
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(hit_rate, 4),
            "total_entries": row[0],
            "ttl_seconds": self._ttl,
            "threshold": self._threshold,
        }

    def clear_expired(self) -> int:
        """Delete expired entries; return count removed."""
        cur = self._conn.execute(
            "DELETE FROM cache_entries WHERE (? - created_at) > ttl_seconds",
            (time.time(),),
        )
        self._conn.commit()
        return cur.rowcount

    def close(self) -> None:
        self._conn.close()

    def __repr__(self) -> str:
        s = self.stats()
        return (
            f"SemanticCache(db={self._db_path}, entries={s['total_entries']}, "
            f"hit_rate={s['hit_rate']:.1%}, threshold={self._threshold})"
        )
