#!/usr/bin/env python3
"""
PostToolUse hook for Claude Code.

Compresses tool outputs before they are fed back to Claude, using both
tool-specific filters (Bash command patterns, Grep deduplication, Read
caching) and generic TokenShrink compression.

Input:  JSON on stdin  — tool_name, tool_input, output, session_id, ...
Output: JSON on stdout — hookSpecificOutput.updatedToolOutput (if compressed)
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from tokenshrink import TokenShrink

MIN_TOKENS = int(os.environ.get("TOKENSHRINK_MIN_TOKENS", "200"))
COMPRESS_TOOLS = {"Bash", "Read", "WebFetch", "WebSearch", "Grep"}


# ── Session cache ─────────────────────────────────────────────────────────────

def _session_dir(session_id: str) -> Path:
    key = hashlib.md5(session_id.encode()).hexdigest()[:8]
    d = Path(tempfile.gettempdir()) / f"tokenshrink_{key}"
    d.mkdir(exist_ok=True)
    return d


# ── Bash command-pattern filters ──────────────────────────────────────────────

def _filter_pytest(output: str) -> str:
    """Keep only failures, errors, and the final summary line."""
    lines = output.splitlines()
    result: list[str] = []
    in_relevant = False

    for line in lines:
        # Final summary (=== N failed, M passed ===)
        if re.match(r"^=+ .*(passed|failed|error).* =+$", line, re.I):
            result.append(line)
            continue
        # FAILURES / ERRORS / short test summary sections
        if re.match(r"^=+ (FAILURES|ERRORS|short test summary)", line, re.I):
            in_relevant = True
            result.append(line)
            continue
        if in_relevant:
            result.append(line)
        elif re.search(r"\bwarning\b", line, re.I):
            result.append(line)

    return "\n".join(result) if result else output


def _filter_install(output: str) -> str:
    """Keep only meaningful lines from npm/pip/yarn install output."""
    keep = re.compile(
        r"error|warn|successfully installed|added \d+|updated \d+|"
        r"removed \d+|packages? (installed|updated)|requirement already|"
        r"collecting |^\+ |done\.",
        re.I,
    )
    skip = re.compile(r"^\s*[█▓░▒]+|^\s*\d+%|^npm http|node_modules", re.I)

    result = [l for l in output.splitlines() if not skip.search(l) and keep.search(l)]
    return "\n".join(result) if len(result) >= 3 else output


def _filter_git_log(output: str) -> str:
    """Strip Author/Date header lines; collapse consecutive blank lines."""
    result: list[str] = []
    for line in output.splitlines():
        if re.match(r"^(Author|Date):\s+", line):
            continue
        if line.strip() == "" and result and result[-1].strip() == "":
            continue
        result.append(line)
    return "\n".join(result)


def _filter_traceback(output: str) -> str:
    """For Python tracebacks: keep first 3 + last 3 frames per block."""
    KEEP = 3
    lines = output.splitlines()
    tb_starts = [i for i, l in enumerate(lines) if "Traceback (most recent call last)" in l]
    if not tb_starts:
        return output

    result: list[str] = []
    prev_end = 0

    for start in tb_starts:
        result.extend(lines[prev_end:start])
        result.append(lines[start])

        frames: list[tuple[str, str]] = []
        error_line: str | None = None
        i = start + 1
        while i < len(lines):
            line = lines[i]
            if line.startswith("  File ") and i + 1 < len(lines):
                frames.append((line, lines[i + 1]))
                i += 2
            elif frames and not line.startswith(" "):
                error_line = line
                i += 1
                break
            else:
                i += 1

        if len(frames) > KEEP * 2:
            for a, b in frames[:KEEP]:
                result += [a, b]
            result.append(f"  ... ({len(frames) - KEEP * 2} frames omitted) ...")
            for a, b in frames[-KEEP:]:
                result += [a, b]
        else:
            for a, b in frames:
                result += [a, b]

        if error_line:
            result.append(error_line)
        prev_end = i

    result.extend(lines[prev_end:])
    return "\n".join(result)


def _filter_directory(output: str) -> str:
    """Strip permissions/owner/timestamps from ls -la / find output."""
    # ls -la: drwxr-xr-x  2 user group 4096 Jan  1 12:00 name
    pat = re.compile(
        r"^([d\-lbcsp][rwx\-]{9})\s+\d+\s+\S+\s+\S+\s+(\d+)\s+\S+\s+\S+\s+\S+\s+(.+)$"
    )
    result: list[str] = []
    for line in output.splitlines():
        m = pat.match(line)
        if m:
            kind = "d" if m.group(1).startswith("d") else "-"
            result.append(f"{kind} {m.group(3):<40} {int(m.group(2)):>10}")
        else:
            result.append(line)
    return "\n".join(result)


def apply_bash_filter(command: str, output: str) -> str:
    """Route command output through the appropriate filter."""
    cmd = (command or "").strip()

    if re.search(r"\bpytest\b|\bpython\s+-m\s+(pytest|unittest)\b", cmd):
        return _filter_pytest(output)

    if re.search(r"\b(npm\s+(install|i)|pip3?\s+install|yarn\s+add)\b", cmd):
        return _filter_install(output)

    if re.search(r"\bgit\s+log\b", cmd):
        return _filter_git_log(output)

    if re.search(r"\bls\s+-\S*[la]\S*|\bfind\s+", cmd):
        return _filter_directory(output)

    # Pattern-free: detect Python tracebacks in any output
    if "Traceback (most recent call last)" in output:
        return _filter_traceback(output)

    return output


# ── Grep output deduplication ─────────────────────────────────────────────────

def deduplicate_grep(output: str) -> str:
    """Remove duplicate context blocks separated by '--' in grep output."""
    blocks = re.split(r"\n--\n", output)
    seen: set[str] = set()
    unique: list[str] = []

    for block in blocks:
        # Strip file:line: prefixes before hashing so identical content matches
        normalized = re.sub(r"^[^\n:]+[:\-]\d+[:\-]", "", block, flags=re.MULTILINE).strip()
        key = hashlib.md5(normalized.encode()).hexdigest()
        if key not in seen:
            seen.add(key)
            unique.append(block)

    if len(unique) == len(blocks):
        return output

    removed = len(blocks) - len(unique)
    suffix = f"\n[TokenShrink: {removed} duplicate context block(s) omitted]"
    return "\n--\n".join(unique) + suffix


# ── Read-tool session cache ───────────────────────────────────────────────────

def check_read_cache(session_id: str, file_path: str, content: str) -> tuple[bool, str]:
    """
    Returns (cache_hit, text).  On a hit the text is a short note; on a miss
    the cache entry is written and the original content is returned unchanged.
    """
    sdir = _session_dir(session_id)
    cache_file = sdir / f"read_{hashlib.md5(file_path.encode()).hexdigest()}.json"
    content_hash = hashlib.md5(content.encode()).hexdigest()

    if cache_file.exists():
        try:
            cached = json.loads(cache_file.read_text())
            if cached.get("hash") == content_hash:
                lines = len(content.splitlines())
                return True, (
                    f"[TokenShrink: '{os.path.basename(file_path)}' unchanged since last read "
                    f"— {lines} lines, {len(content)} chars. Request a line range if you need content.]"
                )
        except Exception:
            pass

    cache_file.write_text(json.dumps({"hash": content_hash, "path": file_path}))
    return False, content


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    try:
        data = json.loads(sys.stdin.read())
    except (json.JSONDecodeError, ValueError):
        sys.exit(0)

    tool_name = data.get("tool_name", "")
    if tool_name not in COMPRESS_TOOLS:
        sys.exit(0)

    output = data.get("output", "")
    if not isinstance(output, str) or not output.strip():
        sys.exit(0)

    session_id = data.get("session_id", "default")
    tool_input = data.get("tool_input") or {}

    processed = output

    # ── Tool-specific pre-processing ──────────────────────────────────────────
    if tool_name == "Bash":
        processed = apply_bash_filter(tool_input.get("command", ""), processed)

    elif tool_name == "Read":
        file_path = tool_input.get("file_path", "")
        if file_path:
            hit, cache_result = check_read_cache(session_id, file_path, processed)
            if hit:
                print(json.dumps({"hookSpecificOutput": {"updatedToolOutput": cache_result}}))
                sys.exit(0)

    elif tool_name == "Grep":
        processed = deduplicate_grep(processed)

    # ── Generic TokenShrink compression ──────────────────────────────────────
    ts = TokenShrink()
    original_tokens = ts.count_tokens(output)
    current_tokens = ts.count_tokens(processed)

    if current_tokens >= MIN_TOKENS:
        result = ts.compress_prompt(processed)
        if result.compressed_tokens < current_tokens:
            processed = result.compressed_text
            current_tokens = result.compressed_tokens

    if current_tokens >= original_tokens:
        sys.exit(0)

    response: dict = {"hookSpecificOutput": {"updatedToolOutput": processed}}

    if os.environ.get("TOKENSHRINK_DEBUG"):
        saved = original_tokens - current_tokens
        pct = saved / original_tokens * 100 if original_tokens else 0
        response["systemMessage"] = (
            f"[TokenShrink] {tool_name}: {original_tokens} → {current_tokens} tokens "
            f"({pct:.0f}% reduction, {saved} saved)"
        )

    print(json.dumps(response))


if __name__ == "__main__":
    main()
