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

import difflib
import hashlib
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from tokenshrink import TokenShrink
from claude_code_integration.session_state import SessionState, session_dir
from claude_code_integration.tool_profiles import apply_fill_escalation, get_profile, load_profiles

_HEAD_TAIL = 20
COMPRESS_TOOLS = {"Bash", "Read", "WebFetch", "WebSearch", "Grep"}
_PROFILES = load_profiles()


# ── Code block truncation ─────────────────────────────────────────────────────

_CODE_BLOCK_RE = re.compile(r"(```[^\n]*\n)([\s\S]*?)(```)", re.DOTALL)


def _truncate_large_code_blocks(text: str, max_lines: int) -> str:
    """Truncate fenced code blocks over max_lines to head+tail snippets."""
    def _maybe_truncate(m: re.Match) -> str:
        opener, content, closer = m.group(1), m.group(2), m.group(3)
        lines = content.splitlines()
        if len(lines) <= max_lines:
            return m.group(0)
        omitted = len(lines) - _HEAD_TAIL * 2
        kept = lines[:_HEAD_TAIL] + [f"... ({omitted} lines omitted) ..."] + lines[-_HEAD_TAIL:]
        return opener + "\n".join(kept) + "\n" + closer

    return _CODE_BLOCK_RE.sub(_maybe_truncate, text)


# ── Session cache ─────────────────────────────────────────────────────────────


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


# ── Build log structurer ──────────────────────────────────────────────────────

def _filter_tsc(output: str) -> str:
    """Group TypeScript compiler errors by file."""
    errors_by_file: dict[str, list[str]] = {}
    other_errors: list[str] = []

    for line in output.splitlines():
        m = re.match(r"^(.+\.tsx?)\((\d+),(\d+)\): (error|warning) (TS\d+): (.+)$", line)
        if m:
            filepath, row, col, severity, code, msg = m.groups()
            if severity == "error":
                errors_by_file.setdefault(filepath, []).append(f"  ({row},{col}) {code}: {msg}")
        elif re.search(r"\berror\b", line, re.I):
            other_errors.append(line)

    if not errors_by_file and not other_errors:
        return output

    total = sum(len(v) for v in errors_by_file.values()) + len(other_errors)
    result = [f"[tsc: {total} error(s) in {len(errors_by_file)} file(s)]"]
    for filepath, errs in sorted(errors_by_file.items()):
        result.append(f"{filepath}: {len(errs)} error(s)")
        result.extend(errs[:3])
        if len(errs) > 3:
            result.append(f"  ... {len(errs) - 3} more")
    result.extend(other_errors[:5])
    return "\n".join(result)


def _filter_build_log(command: str, output: str) -> str:
    """Structure build tool output: errors, warning counts, last 5 lines."""
    if re.search(r"\btsc\b", command):
        return _filter_tsc(output)

    lines = output.splitlines()
    error_lines: list[str] = []
    warning_counts: dict[str, int] = {}

    for line in lines:
        if re.search(r"\berror\b|\bERROR\b|Error:", line):
            error_lines.append(line)
        elif re.search(r"\bwarning\b|\bWARN\b", line, re.I):
            m = re.search(r"(?:warning|warn)[:\s]+([A-Za-z][\w\-/]+)", line, re.I)
            key = m.group(1) if m else "general"
            warning_counts[key] = warning_counts.get(key, 0) + 1

    if not error_lines and not warning_counts:
        # Build succeeded — just show tail
        return "\n".join(lines[-5:]) if len(lines) > 5 else output

    result: list[str] = []
    if error_lines:
        result.append(f"[Errors: {len(error_lines)}]")
        result.extend(error_lines[:20])
        if len(error_lines) > 20:
            result.append(f"  ... {len(error_lines) - 20} more error(s)")
    if warning_counts:
        warn_parts = ", ".join(f"{k}:{v}" for k, v in sorted(warning_counts.items()))
        result.append(f"[Warnings] {warn_parts}")
    result.append("[Last lines]")
    result.extend(lines[-5:])
    return "\n".join(result)


# ── Linter noise filter ───────────────────────────────────────────────────────

def _filter_eslint_json(data: list) -> str:
    """Compact ESLint JSON-format output."""
    result: list[str] = []
    seen_warn_rules: set[str] = set()
    total_errors = total_warnings = 0

    for file_result in data:
        filepath = file_result.get("filePath", "")
        short = filepath.replace("\\", "/").split("/")[-1]
        messages = file_result.get("messages", [])
        file_lines: list[str] = []

        for msg in messages:
            sev = msg.get("severity", 0)
            rule = msg.get("ruleId") or "unknown"
            text = msg.get("message", "")
            loc = f"{msg.get('line',0)}:{msg.get('column',0)}"
            if sev == 2:
                total_errors += 1
                file_lines.append(f"  {loc} error  {text} ({rule})")
            elif sev == 1:
                total_warnings += 1
                if rule not in seen_warn_rules:
                    seen_warn_rules.add(rule)
                    file_lines.append(f"  {loc} warn   {text} ({rule})")

        if file_lines:
            result.append(f"{short}:")
            result.extend(file_lines)

    summary = f"[ESLint: {total_errors} errors, {total_warnings} warnings ({len(seen_warn_rules)} unique warning rules shown)]"
    return ("\n".join(result) + "\n" + summary) if result else ""


def _filter_ruff_text(output: str) -> str:
    """Deduplicate ruff warning rules; keep all errors."""
    seen_rules: set[str] = set()
    result: list[str] = []
    for line in output.splitlines():
        m = re.search(r":\s+([A-Z]\d+)\s+", line)
        if m:
            rule = m.group(1)
            if rule.startswith("W"):
                if rule not in seen_rules:
                    seen_rules.add(rule)
                    result.append(line)
                # else: drop duplicate warning rule
            else:
                result.append(line)  # errors always kept
        else:
            result.append(line)
    return "\n".join(result)


def _filter_linter_text(output: str) -> str:
    """Generic line-based linter filter: deduplicate warnings by extracted rule name."""
    seen_rules: set[str] = set()
    result: list[str] = []
    for line in output.splitlines():
        lower = line.lower()
        if "error" in lower:
            result.append(line)
        elif "warning" in lower or "warn" in lower:
            # Try quoted rule, parenthesized code, or trailing rule name (eslint text format)
            m = re.search(
                r"['\"]([a-z][\w\-]{2,})['\"]"
                r"|\(([A-Z]\d+)\)"
                r"|[ \t]([a-z][\w\-]{3,})[ \t]*$",
                line,
            )
            rule = next((g for g in (m.groups() if m else []) if g), line[:50])
            if rule not in seen_rules:
                seen_rules.add(rule)
                result.append(line)
        else:
            result.append(line)
    return "\n".join(result)


def _filter_linter(command: str, output: str) -> str:
    """Route linter output to the right filter."""
    if re.search(r"\beslint\b", command):
        try:
            data = json.loads(output)
            if isinstance(data, list):
                filtered = _filter_eslint_json(data)
                return filtered if filtered else output
        except (json.JSONDecodeError, ValueError):
            pass
        return _filter_linter_text(output)
    if re.search(r"\bruff\b", command):
        return _filter_ruff_text(output)
    return _filter_linter_text(output)


# ── Jest / Vitest filter ──────────────────────────────────────────────────────

def _filter_jest(output: str) -> str:
    """Collapse passing suites; keep failures and summary."""
    lines = output.splitlines()
    result: list[str] = []
    pass_count = 0
    in_failure = False

    for line in lines:
        # Suite result lines
        if re.match(r"^\s*(PASS|✓)\s+", line):
            pass_count += 1
            continue
        if re.match(r"^\s*(FAIL|✕|×)\s+", line):
            in_failure = True
            result.append(line)
            continue
        # Summary line
        if re.match(r"^(Tests?|Test Suites?|Suites?):", line):
            in_failure = False
            result.append(line)
            continue
        if in_failure:
            result.append(line)

    prefix = [f"[{pass_count} suite(s) passed -- omitted]"] if pass_count else []
    return "\n".join(prefix + result) if (prefix or result) else output


def apply_bash_filter(command: str, output: str) -> str:
    """Route command output through the appropriate filter."""
    cmd = (command or "").strip()

    if re.search(r"\bpytest\b|\bpython\s+-m\s+(pytest|unittest)\b", cmd):
        return _filter_pytest(output)

    if re.search(r"\b(jest|vitest)\b", cmd):
        return _filter_jest(output)

    if re.search(r"\b(npm\s+(install|i)|pip3?\s+install|yarn\s+add)\b", cmd):
        return _filter_install(output)

    if re.search(r"\bgit\s+log\b", cmd):
        return _filter_git_log(output)

    if re.search(r"\bls\s+-\S*[la]\S*|\bfind\s+", cmd):
        return _filter_directory(output)

    if re.search(r"\b(tsc\b|webpack\b|cargo\s+build|make\b|gradle\b|next\s+build)", cmd):
        return _filter_build_log(cmd, output)

    if re.search(r"\b(eslint|ruff\s+check|flake8|pylint)\b", cmd):
        return _filter_linter(cmd, output)

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

_MAX_CACHED_BYTES = 200 * 1024  # 200 KB per file


def check_read_cache(session_id: str, file_path: str, content: str) -> tuple[bool, str]:
    """
    Returns (cache_hit, text).

    - Identical content: returns a short "unchanged" note.
    - Changed content (diff < 30% of new size): returns the unified diff.
    - Changed content (large diff) or first read: stores content and returns unchanged.

    Content is capped at _MAX_CACHED_BYTES to limit session dir growth.
    """
    sdir = session_dir(session_id)
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

            # Content changed — attempt delta compression
            prev_content = cached.get("content")
            prev_turn = cached.get("turn", 0)
            if prev_content is not None:
                diff_lines = list(difflib.unified_diff(
                    prev_content.splitlines(keepends=True),
                    content.splitlines(keepends=True),
                    fromfile=f"{os.path.basename(file_path)} (turn {prev_turn})",
                    tofile=f"{os.path.basename(file_path)} (current)",
                    lineterm="",
                ))
                diff_text = "".join(diff_lines)
                if len(diff_text) < len(content) * 0.30:
                    # Store updated content for next diff
                    _write_cache(cache_file, content_hash, file_path, content, session_id)
                    return True, (
                        f"[TokenShrink: showing diff vs turn {prev_turn}]\n{diff_text}"
                    )
        except Exception:
            pass

    _write_cache(cache_file, content_hash, file_path, content, session_id)
    return False, content


def _write_cache(cache_file: Path, content_hash: str, file_path: str, content: str, session_id: str) -> None:
    from claude_code_integration.session_state import SessionState
    turn = SessionState.load(session_id).turn_number
    payload: dict = {"hash": content_hash, "path": file_path, "turn": turn}
    if len(content.encode()) <= _MAX_CACHED_BYTES:
        payload["content"] = content
    cache_file.write_text(json.dumps(payload))


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

    # ── Per-tool profile + adaptive fill escalation ───────────────────────────
    fill = SessionState.load(session_id).fill_fraction
    profile = apply_fill_escalation(get_profile(tool_name, _PROFILES), fill)

    # ── Code block truncation ─────────────────────────────────────────────────
    processed = _truncate_large_code_blocks(processed, profile.max_code_block_lines)

    # ── Generic TokenShrink compression ──────────────────────────────────────
    ts = TokenShrink(techniques=profile.techniques)
    original_tokens = ts.count_tokens(output)
    current_tokens = ts.count_tokens(processed)

    if current_tokens >= profile.min_tokens:
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
