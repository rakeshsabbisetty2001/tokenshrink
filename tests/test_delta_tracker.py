"""Tests for the File-Content Delta Tracker in check_read_cache."""
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Patch session_dir to use a temp directory for isolation
import claude_code_integration.hooks.compress_tool_output as cto_module
import claude_code_integration.session_state as ss_module


def _make_session(tmp_path: Path) -> str:
    """Return a unique session ID backed by tmp_path."""
    sid = f"test_{tmp_path.name}"
    original = ss_module._persistent_base

    def _fake_base():
        tmp_path.mkdir(parents=True, exist_ok=True)
        return tmp_path

    ss_module._persistent_base = _fake_base
    return sid


class TestCheckReadCache:
    def test_first_read_returns_content(self, tmp_path):
        sid = _make_session(tmp_path)
        hit, result = cto_module.check_read_cache(sid, "/fake/file.py", "hello world\n")
        assert not hit
        assert result == "hello world\n"

    def test_same_content_cache_hit(self, tmp_path):
        sid = _make_session(tmp_path)
        content = "line one\nline two\n"
        cto_module.check_read_cache(sid, "/fake/file.py", content)
        hit, result = cto_module.check_read_cache(sid, "/fake/file.py", content)
        assert hit
        assert "unchanged" in result
        assert "file.py" in result

    def test_changed_content_large_diff_no_delta(self, tmp_path):
        sid = _make_session(tmp_path)
        original = "\n".join(f"line {i}" for i in range(50))
        changed = "\n".join(f"new {i}" for i in range(50))  # ~100% diff
        cto_module.check_read_cache(sid, "/fake/big.py", original)
        hit, result = cto_module.check_read_cache(sid, "/fake/big.py", changed)
        # Diff is large — should NOT return a hit with delta
        assert not hit
        assert result == changed

    def test_changed_content_small_diff_returns_delta(self, tmp_path):
        sid = _make_session(tmp_path)
        # 100 lines → ~700 bytes; changing one line gives a tiny diff relative to file size
        lines = [f"stable content line number {i} with some padding text here\n" for i in range(100)]
        original = "".join(lines)
        modified = lines[:]
        modified[50] = "line 50 was CHANGED to something different\n"
        changed = "".join(modified)

        cto_module.check_read_cache(sid, "/fake/small.py", original)
        hit, result = cto_module.check_read_cache(sid, "/fake/small.py", changed)
        assert hit
        assert "diff" in result.lower() or "@@" in result

    def test_different_files_independent(self, tmp_path):
        sid = _make_session(tmp_path)
        content_a = "file A content"
        content_b = "file B content"
        cto_module.check_read_cache(sid, "/a.py", content_a)
        cto_module.check_read_cache(sid, "/b.py", content_b)
        hit_a, _ = cto_module.check_read_cache(sid, "/a.py", content_a)
        hit_b, _ = cto_module.check_read_cache(sid, "/b.py", content_b)
        assert hit_a
        assert hit_b

    def test_large_file_content_not_stored(self, tmp_path):
        sid = _make_session(tmp_path)
        # ~210 KB — over the 200 KB cap
        big_content = "x" * (210 * 1024)
        cto_module.check_read_cache(sid, "/big_file.bin", big_content)
        # Read the cache JSON and confirm "content" key is absent
        import hashlib
        sdir = ss_module.session_dir(sid)
        cache_file = sdir / f"read_{hashlib.md5('/big_file.bin'.encode()).hexdigest()}.json"
        cached = json.loads(cache_file.read_text())
        assert "content" not in cached
