from __future__ import annotations

import re


def compress(text: str, opts: dict) -> str:
    """Normalize whitespace without touching code blocks or quoted strings."""
    if not text:
        return text

    # Preserve fenced code blocks verbatim
    parts = re.split(r"(```[\s\S]*?```)", text)
    result = []
    for i, part in enumerate(parts):
        if i % 2 == 1:
            # Inside fenced code block — preserve exactly
            result.append(part)
        else:
            result.append(_compress_prose(part, opts))
    return "".join(result)


def _compress_prose(text: str, opts: dict) -> str:
    # Normalize Windows/Mac line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Strip trailing whitespace on every line
    text = re.sub(r"[ \t]+$", "", text, flags=re.MULTILINE)
    # Strip leading whitespace on lines that are otherwise blank
    text = re.sub(r"^[ \t]+$", "", text, flags=re.MULTILINE)
    # Collapse 3+ consecutive blank lines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse multiple spaces within a line (but not leading indentation)
    text = re.sub(r"(?<=\S) {2,}", " ", text)
    # Convert tab-only indentation to 2 spaces (configurable)
    indent_spaces = opts.get("indent_spaces", 2) if isinstance(opts, dict) else 2
    text = re.sub(r"^(\t+)", lambda m: " " * indent_spaces * len(m.group(1)), text, flags=re.MULTILINE)
    # Strip leading/trailing blank lines from the whole document
    text = text.strip("\n")
    return text
