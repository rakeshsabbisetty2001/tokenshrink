"""Per-tool compression profiles for the compress_tool_output hook."""
from __future__ import annotations

import json
import os
from typing import NamedTuple


class ToolProfile(NamedTuple):
    techniques: list[str]
    min_tokens: int
    max_code_block_lines: int


# Tuned defaults per tool type:
#   Bash    — omit format (output rarely has XML/boilerplate), add semantic_dedup
#   Read    — conservative; large code files should stay mostly intact
#   WebFetch/WebSearch — full pipeline; HTML/JSON responses benefit from format optimizer
#   Grep    — whitespace + dedup only; context snippets are short and already compact
DEFAULT_PROFILES: dict[str, ToolProfile] = {
    "Bash": ToolProfile(
        techniques=["whitespace", "deduplication", "semantic_dedup"],
        min_tokens=150,
        max_code_block_lines=60,
    ),
    "Read": ToolProfile(
        techniques=["whitespace", "deduplication"],
        min_tokens=300,
        max_code_block_lines=80,
    ),
    "WebFetch": ToolProfile(
        techniques=["whitespace", "deduplication", "format", "semantic_dedup"],
        min_tokens=100,
        max_code_block_lines=40,
    ),
    "WebSearch": ToolProfile(
        techniques=["whitespace", "deduplication", "format", "semantic_dedup"],
        min_tokens=100,
        max_code_block_lines=40,
    ),
    "Grep": ToolProfile(
        techniques=["whitespace", "deduplication"],
        min_tokens=200,
        max_code_block_lines=60,
    ),
}

_FALLBACK = ToolProfile(
    techniques=["whitespace", "deduplication"],
    min_tokens=200,
    max_code_block_lines=60,
)


def load_profiles() -> dict[str, ToolProfile]:
    """Return profiles, optionally overridden by TOKENSHRINK_TOOL_PROFILES (path to JSON)."""
    profiles: dict[str, ToolProfile] = dict(DEFAULT_PROFILES)
    override_path = os.environ.get("TOKENSHRINK_TOOL_PROFILES")
    if override_path:
        try:
            with open(override_path) as f:
                overrides = json.load(f)
            for tool_name, cfg in overrides.items():
                profiles[tool_name] = ToolProfile(
                    techniques=cfg.get("techniques", _FALLBACK.techniques),
                    min_tokens=int(cfg.get("min_tokens", _FALLBACK.min_tokens)),
                    max_code_block_lines=int(cfg.get("max_code_block_lines", _FALLBACK.max_code_block_lines)),
                )
        except Exception:
            pass
    return profiles


def get_profile(tool_name: str, profiles: dict[str, ToolProfile] | None = None) -> ToolProfile:
    if profiles is None:
        profiles = load_profiles()
    return profiles.get(tool_name, _FALLBACK)


def apply_fill_escalation(profile: ToolProfile, fill: float) -> ToolProfile:
    """Upgrade a profile's aggressiveness based on context fill fraction."""
    if fill < 0.40:
        return profile
    if fill < 0.70:
        techniques = list(profile.techniques)
        if "semantic_dedup" not in techniques:
            techniques.append("semantic_dedup")
        return ToolProfile(
            techniques=techniques,
            min_tokens=max(50, profile.min_tokens // 2),
            max_code_block_lines=profile.max_code_block_lines,
        )
    # >= 70%: full pipeline, aggressive thresholds
    return ToolProfile(
        techniques=["whitespace", "deduplication", "format", "semantic_dedup"],
        min_tokens=50,
        max_code_block_lines=min(40, profile.max_code_block_lines),
    )
