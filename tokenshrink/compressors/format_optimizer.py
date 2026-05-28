from __future__ import annotations

import re


# Each entry: (compiled_pattern, replacement, min_occurrences_to_fire)
_PATTERNS = [
    # LangChain / LlamaIndex XML chunk wrappers
    (re.compile(r"<document(?:\s[^>]*)?>", re.IGNORECASE), "---\n", 1),
    (re.compile(r"</document>", re.IGNORECASE), "", 1),
    (re.compile(r"<source>([^<]*)</source>\s*", re.IGNORECASE), r"[\1] ", 1),
    (re.compile(r"<content>\s*", re.IGNORECASE), "", 1),
    (re.compile(r"\s*</content>", re.IGNORECASE), "", 1),
    # Generic XML tag pairs that wrap single blocks of text (not inline)
    (re.compile(r"^<(?:chunk|passage|context|result|item|entry)(?:\s[^>]*)?>[ \t]*\n", re.MULTILINE | re.IGNORECASE), "", 2),
    (re.compile(r"^</(?:chunk|passage|context|result|item|entry)>[ \t]*\n?", re.MULTILINE | re.IGNORECASE), "", 2),
    # Verbose Markdown heading decorators (underline style) → ATX style already compact
    (re.compile(r"^={3,}\s*$", re.MULTILINE), "", 2),
    (re.compile(r"^-{3,}\s*$", re.MULTILINE), "---", 2),
    # HTML entities → Unicode
    (re.compile(r"&amp;"), "&", 1),
    (re.compile(r"&lt;"), "<", 1),
    (re.compile(r"&gt;"), ">", 1),
    (re.compile(r"&quot;"), '"', 1),
    (re.compile(r"&apos;"), "'", 1),
    (re.compile(r"&#(\d+);"), lambda m: chr(int(m.group(1))), 1),
    # Repeated boilerplate delimiters (4+ equals or dashes on their own line → single ---)
    (re.compile(r"^[=\-]{4,}\s*$", re.MULTILINE), "---", 3),
    # Extra blank lines after structural substitutions
    (re.compile(r"\n{3,}"), "\n\n", 1),
]


def compress(text: str, opts: dict) -> str:
    if not text:
        return text

    # Preserve fenced code blocks
    parts = re.split(r"(```[\s\S]*?```)", text)
    result = []
    for i, part in enumerate(parts):
        if i % 2 == 1:
            result.append(part)
        else:
            result.append(_apply_patterns(part))
    return "".join(result)


def _apply_patterns(text: str) -> str:
    for pattern, replacement, min_count in _PATTERNS:
        if callable(replacement):
            text = pattern.sub(replacement, text)
        else:
            if min_count > 1:
                if len(pattern.findall(text)) < min_count:
                    continue
            text = pattern.sub(replacement, text)
    return text
