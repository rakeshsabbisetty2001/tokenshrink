"""Shared session state helpers for TokenShrink Claude Code hooks."""
from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path


def _persistent_base() -> Path | None:
    """
    Return the persistent state directory, or None if it cannot be created.

    Priority:
    1. TOKENSHRINK_STATE_DIR env var (explicit override)
    2. ~/.claude/tokenshrink/  (default persistent location)
    3. None → caller falls back to OS temp dir
    """
    override = os.environ.get("TOKENSHRINK_STATE_DIR")
    if override:
        try:
            p = Path(override).expanduser()
            p.mkdir(parents=True, exist_ok=True)
            return p
        except OSError:
            return None

    default = Path.home() / ".claude" / "tokenshrink"
    try:
        default.mkdir(parents=True, exist_ok=True)
        return default
    except OSError:
        return None


def session_dir(session_id: str) -> Path:
    """
    Return (and create) the per-session state directory.

    Tries the persistent location first (~/.claude/tokenshrink/<hash>).
    Falls back to the OS temp dir if the persistent location is unavailable,
    preserving the original behaviour for read-only home directories.
    """
    key = hashlib.md5(session_id.encode()).hexdigest()[:8]
    base = _persistent_base()
    if base is not None:
        d = base / key
    else:
        d = Path(tempfile.gettempdir()) / f"tokenshrink_{key}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def approx_tokens(text: str) -> int:
    return math.ceil(len(text.split()) * 1.3)


@dataclass
class SessionState:
    output_tokens: int = 0
    input_tokens: int = 0
    turns: int = 0
    fill_fraction: float = 0.0
    turn_number: int = 0

    @classmethod
    def load(cls, session_id: str) -> "SessionState":
        budget_file = session_dir(session_id) / "budget.json"
        if budget_file.exists():
            try:
                raw = json.loads(budget_file.read_text())
                return cls(
                    output_tokens=int(raw.get("output_tokens", 0)),
                    input_tokens=int(raw.get("input_tokens", 0)),
                    turns=int(raw.get("turns", 0)),
                    fill_fraction=float(raw.get("fill_fraction", 0.0)),
                    turn_number=int(raw.get("turn_number", 0)),
                )
            except Exception:
                pass
        return cls()

    def save(self, session_id: str) -> None:
        budget_file = session_dir(session_id) / "budget.json"
        budget_file.write_text(json.dumps(asdict(self)))
