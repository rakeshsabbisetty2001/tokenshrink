"""Rule table for the PreToolUse command rewriter hook."""
from __future__ import annotations

import re
from typing import NamedTuple


class RewriteRule(NamedTuple):
    pattern: re.Pattern
    replacement: str
    min_fill: float  # only apply when fill_fraction >= this value


# Rules applied in order; first match wins for each command.
# min_fill=0.0 means "always apply".
REWRITE_RULES: list[RewriteRule] = [
    # ── Always ──────────────────────────────────────────────────────────────────
    RewriteRule(
        re.compile(r"^cat\s+(\S+)\s*$"),
        r"head -100 \1",
        0.0,
    ),
    RewriteRule(
        re.compile(r"^pip\s+list\s*$"),
        "pip list | head -30",
        0.0,
    ),
    RewriteRule(
        re.compile(r"^pip3\s+list\s*$"),
        "pip3 list | head -30",
        0.0,
    ),
    # ── fill > 0.40 ─────────────────────────────────────────────────────────────
    RewriteRule(
        re.compile(r"^(find\s+\S+\s+-name\s+\"[^\"]+\")\s*$"),
        r"\1 | head -50",
        0.40,
    ),
    RewriteRule(
        re.compile(r"^(find\s+\S+\s+-name\s+'[^']+')\s*$"),
        r"\1 | head -50",
        0.40,
    ),
    RewriteRule(
        re.compile(r"^ls\s+-la?\s*$"),
        "ls -la | head -40",
        0.40,
    ),
    RewriteRule(
        re.compile(r"^ls\s+-al?\s*$"),
        "ls -al | head -40",
        0.40,
    ),
    # ── fill > 0.70 ─────────────────────────────────────────────────────────────
    RewriteRule(
        re.compile(r"^(npm\s+test)\s*$"),
        r"\1 -- --reporter=min",
        0.70,
    ),
    RewriteRule(
        re.compile(r"^(npm\s+test\s+--)(?!\s+--reporter)(.*)$"),
        r"\1 --reporter=min\2",
        0.70,
    ),
    RewriteRule(
        re.compile(r"^(cargo\s+test)\s*$"),
        r"\1 2>&1 | tail -50",
        0.70,
    ),
]


def rewrite_command(command: str, fill_fraction: float) -> tuple[str, str | None]:
    """
    Apply the first matching rule for this fill fraction.

    Returns (rewritten_command, note) where note is a human-readable
    description of the rewrite, or (original_command, None) if no rule matched.
    """
    cmd = command.strip()
    for rule in REWRITE_RULES:
        if fill_fraction < rule.min_fill:
            continue
        if rule.pattern.match(cmd):
            rewritten = rule.pattern.sub(rule.replacement, cmd)
            if rewritten != cmd:
                note = f"rewritten for context efficiency: `{cmd}` → `{rewritten}`"
                return rewritten, note
    return command, None
