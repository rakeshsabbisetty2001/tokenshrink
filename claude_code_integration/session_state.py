"""Shared session state helpers for TokenShrink Claude Code hooks."""
from __future__ import annotations

import hashlib
import json
import math
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path


def session_dir(session_id: str) -> Path:
    key = hashlib.md5(session_id.encode()).hexdigest()[:8]
    d = Path(tempfile.gettempdir()) / f"tokenshrink_{key}"
    d.mkdir(exist_ok=True)
    return d


def approx_tokens(text: str) -> int:
    return math.ceil(len(text.split()) * 1.3)


@dataclass
class SessionState:
    output_tokens: int = 0
    input_tokens: int = 0
    turns: int = 0
    fill_fraction: float = 0.0

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
                )
            except Exception:
                pass
        return cls()

    def save(self, session_id: str) -> None:
        budget_file = session_dir(session_id) / "budget.json"
        budget_file.write_text(json.dumps(asdict(self)))
