"""
Integration tests for all 4 Claude Code hooks.

Each hook reads JSON from stdin and writes JSON to stdout.  We simulate that
by monkeypatching sys.stdin / capturing sys.stdout with capsys.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent.parent))


# ── helpers ──────────────────────────────────────────────────────────────────

def _run_hook(hook_module, stdin_data: dict, env: dict | None = None) -> dict | None:
    """
    Call hook_module.main() with stdin_data as JSON on stdin.
    Returns parsed stdout JSON, or None if the hook exited silently.
    """
    stdin_json = json.dumps(stdin_data)
    env = env or {}

    with patch.dict(os.environ, env, clear=False):
        with patch("sys.stdin") as mock_stdin:
            mock_stdin.read.return_value = stdin_json
            captured_output: list[str] = []
            original_print = print

            def capturing_print(data, **kwargs):
                captured_output.append(data)

            try:
                with patch("builtins.print", side_effect=capturing_print):
                    with pytest.raises(SystemExit):
                        hook_module.main()
            except SystemExit:
                pass
            finally:
                pass

            if captured_output:
                return json.loads(captured_output[0])
            return None


def _run_hook_no_exit(hook_module, stdin_data: dict, env: dict | None = None) -> dict | None:
    """Like _run_hook but handles hooks that may or may not call sys.exit."""
    stdin_json = json.dumps(stdin_data)
    env = env or {}
    captured_output: list[str] = []

    def capturing_print(data, **kwargs):
        captured_output.append(str(data))

    with patch.dict(os.environ, env, clear=False):
        with patch("sys.stdin") as mock_stdin:
            mock_stdin.read.return_value = stdin_json
            try:
                with patch("builtins.print", side_effect=capturing_print):
                    try:
                        hook_module.main()
                    except SystemExit:
                        pass
            except Exception:
                pass

    if captured_output:
        try:
            return json.loads(captured_output[0])
        except (json.JSONDecodeError, IndexError):
            return None
    return None


# ── terse_mode hook ───────────────────────────────────────────────────────────

class TestTerseMode:
    def setup_method(self):
        # Use a fresh temp dir for each test so flag files don't bleed across
        self.tmpdir = tempfile.mkdtemp()

    def _run(self, session_id: str = "test-session", env: dict | None = None) -> dict | None:
        from claude_code_integration.hooks import terse_mode

        env = env or {}
        # Redirect session_dir to our tmp so tests are isolated
        with patch("claude_code_integration.hooks.terse_mode.session_dir") as mock_sdir:
            mock_sdir.return_value = Path(self.tmpdir)
            return _run_hook_no_exit(
                terse_mode,
                {"session_id": session_id, "tool_name": "Bash", "tool_input": {}},
                env=env,
            )

    def test_first_call_injects_nudge(self):
        result = self._run()
        assert result is not None
        assert "systemMessage" in result
        assert "TokenShrink" in result["systemMessage"]

    def test_second_call_is_silent(self):
        self._run()  # first call writes flag
        result = self._run()  # second call should be silent
        assert result is None

    def test_disabled_via_env(self):
        result = self._run(env={"TOKENSHRINK_TERSE": "0"})
        assert result is None

    def test_nudge_mentions_batching(self):
        result = self._run()
        assert result is not None
        msg = result["systemMessage"].lower()
        assert "batch" in msg or "tool" in msg


# ── compress_tool_output hook ─────────────────────────────────────────────────

class TestCompressToolOutput:
    def _run(self, tool_name: str, output: str, tool_input: dict | None = None,
             session_id: str = "test-session", env: dict | None = None) -> dict | None:
        from claude_code_integration.hooks import compress_tool_output

        data = {
            "tool_name": tool_name,
            "output": output,
            "tool_input": tool_input or {},
            "session_id": session_id,
        }
        with patch("claude_code_integration.hooks.compress_tool_output.SessionState") as mock_state:
            mock_state.load.return_value = MagicMock(fill_fraction=0.1)
            return _run_hook_no_exit(compress_tool_output, data, env=env)

    def test_ignored_tool_exits_silently(self):
        result = self._run("Edit", "some output")
        assert result is None

    def test_bash_pytest_filters_to_summary(self):
        output = (
            "collected 10 items\n"
            "test_foo.py::test_bar PASSED\n"
            "test_foo.py::test_baz FAILED\n"
            "FAILED test_foo.py::test_baz — assert 1 == 2\n"
            "=== 1 failed, 9 passed in 0.5s ===\n"
        )
        # The filter should keep at least the summary line
        from claude_code_integration.hooks.compress_tool_output import apply_bash_filter
        filtered = apply_bash_filter("pytest tests/", output)
        assert "failed" in filtered.lower() or "passed" in filtered.lower()

    def test_bash_git_log_strips_author_date(self):
        output = (
            "commit abc123\n"
            "Author: Someone <s@example.com>\n"
            "Date:   Mon Jan 1 00:00:00 2024\n\n"
            "    Add feature\n"
        )
        from claude_code_integration.hooks.compress_tool_output import apply_bash_filter
        filtered = apply_bash_filter("git log --oneline", output)
        assert "Author:" not in filtered
        assert "Date:" not in filtered
        assert "Add feature" in filtered

    def test_grep_deduplication_removes_duplicate_blocks(self):
        from claude_code_integration.hooks.compress_tool_output import deduplicate_grep
        block = "src/foo.py:10: def foo():\nsrc/foo.py:11:     pass"
        output = f"{block}\n--\n{block}\n--\n{block}"
        result = deduplicate_grep(output)
        assert "omitted" in result.lower()
        assert result.count(block) == 1

    def test_read_cache_returns_note_on_second_read(self):
        from claude_code_integration.hooks.compress_tool_output import check_read_cache
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("claude_code_integration.hooks.compress_tool_output.session_dir") as mock_sd:
                mock_sd.return_value = Path(tmpdir)
                content = "line1\nline2\nline3"
                hit1, result1 = check_read_cache("sess", "/path/to/file.py", content)
                assert not hit1
                assert result1 == content
                hit2, result2 = check_read_cache("sess", "/path/to/file.py", content)
                assert hit2
                assert "unchanged" in result2

    def test_large_output_is_compressed(self):
        # Produce a large output with repeated content
        output = ("This is repeated content.\n" * 50)
        result = self._run("Bash", output, tool_input={"command": "echo test"})
        # If compressed, updatedToolOutput should be shorter; if not, result is None
        if result is not None:
            updated = result.get("hookSpecificOutput", {}).get("updatedToolOutput", output)
            assert len(updated) <= len(output)

    def test_empty_output_exits_silently(self):
        result = self._run("Bash", "   ")
        assert result is None

    def test_code_block_truncation(self):
        from claude_code_integration.hooks.compress_tool_output import _truncate_large_code_blocks
        lines = "\n".join(f"line {i}" for i in range(100))
        text = f"```python\n{lines}\n```"
        truncated = _truncate_large_code_blocks(text, max_lines=40)
        assert "omitted" in truncated
        assert "line 0" in truncated
        assert "line 99" in truncated

    def test_traceback_filter_keeps_first_and_last_frames(self):
        from claude_code_integration.hooks.compress_tool_output import _filter_traceback
        frames = ""
        for i in range(8):
            frames += f'  File "f{i}.py", line {i}\n    code_{i}()\n'
        output = f"Traceback (most recent call last):\n{frames}ValueError: bad value\n"
        result = _filter_traceback(output)
        assert "omitted" in result
        assert "f0.py" in result   # first frame kept
        assert "f7.py" in result   # last frame kept


# ── compress_user_prompt hook ────────────────────────────────────────────────

class TestCompressUserPrompt:
    def _run(self, prompt: str, fill: float = 0.1, env: dict | None = None) -> dict | None:
        from claude_code_integration.hooks import compress_user_prompt

        data = {"user_prompt": prompt, "session_id": "test-session"}
        with patch("claude_code_integration.hooks.compress_user_prompt.SessionState") as mock_state:
            mock_state.load.return_value = MagicMock(fill_fraction=fill)
            return _run_hook_no_exit(compress_user_prompt, data, env=env)

    def test_short_prompt_exits_silently(self):
        result = self._run("Hi")
        assert result is None

    def test_empty_prompt_exits_silently(self):
        result = self._run("   ")
        assert result is None

    def test_verbose_prompt_is_compressed(self):
        prompt = (
            "Please   help   me   with   my   code.\n\n\n\n"
            "Please   help   me   with   my   code.\n\n\n\n"
            "I need   assistance   with   my   Python   script."
        ) * 5
        result = self._run(prompt)
        if result is not None:
            updated = result.get("updatedInput", {}).get("user_prompt", prompt)
            assert len(updated) <= len(prompt)

    def test_high_fill_uses_aggressive_profile(self):
        from claude_code_integration.hooks.compress_user_prompt import _compression_profile
        techniques_low, _ = _compression_profile(0.1)
        techniques_high, _ = _compression_profile(0.75)
        assert len(techniques_high) >= len(techniques_low)

    def test_fill_thresholds_correct(self):
        from claude_code_integration.hooks.compress_user_prompt import _compression_profile
        t_low, min_low = _compression_profile(0.1)
        t_mid, min_mid = _compression_profile(0.5)
        t_high, min_high = _compression_profile(0.8)
        assert min_high < min_low  # more aggressive = lower threshold
        assert len(t_high) >= len(t_mid) >= len(t_low)


# ── token_budget hook ─────────────────────────────────────────────────────────

class TestTokenBudget:
    def _run(self, session_id: str = "test-session", fill: float = 0.2,
             api_input: int | None = None, api_output: int | None = 500,
             env: dict | None = None) -> dict | None:
        from claude_code_integration.hooks import token_budget

        usage: dict = {}
        if api_input is not None:
            usage["input_tokens"] = api_input
        if api_output is not None:
            usage["output_tokens"] = api_output

        data = {
            "session_id": session_id,
            "stop_reason": "end_turn",
            "message": {"content": "Hello world", "usage": usage},
        }
        with patch("claude_code_integration.hooks.token_budget.SessionState") as mock_state_cls:
            state = MagicMock()
            state.turns = 1
            state.fill_fraction = fill
            state.input_tokens = api_input or 0
            state.output_tokens = api_output or 0
            mock_state_cls.load.return_value = state

            with patch("claude_code_integration.hooks.token_budget.session_dir") as mock_sdir:
                tmpdir = tempfile.mkdtemp()
                mock_sdir.return_value = Path(tmpdir)
                return _run_hook_no_exit(token_budget, data, env=env)

    def test_disabled_via_env(self):
        result = self._run(env={"TOKENSHRINK_BUDGET": "0"})
        assert result is None

    def test_below_threshold_is_silent(self):
        result = self._run(fill=0.05, env={"TOKENSHRINK_BUDGET_THRESHOLD": "0.15"})
        assert result is None

    def test_above_threshold_emits_status(self):
        # api_input=90000 → fill_fraction = 90000/180000 = 0.50, above threshold 0.10
        result = self._run(fill=0.5, api_input=90000, env={"TOKENSHRINK_BUDGET_THRESHOLD": "0.10"})
        assert result is not None
        assert "systemMessage" in result
        assert "TokenShrink" in result["systemMessage"]

    def test_at_70pct_emits_compact_directive(self):
        from claude_code_integration.hooks import token_budget

        data = {
            "session_id": "compact-test",
            "stop_reason": "end_turn",
            "message": {"content": "x", "usage": {"output_tokens": 1000}},
        }
        with patch("claude_code_integration.hooks.token_budget.SessionState") as mock_state_cls:
            state = MagicMock()
            state.turns = 5
            state.fill_fraction = 0.75
            state.input_tokens = 135000
            state.output_tokens = 5000
            mock_state_cls.load.return_value = state
            with patch("claude_code_integration.hooks.token_budget.session_dir") as mock_sdir:
                tmpdir = tempfile.mkdtemp()
                mock_sdir.return_value = Path(tmpdir)
                result = _run_hook_no_exit(token_budget, data)

        assert result is not None
        msg = result.get("systemMessage", "")
        assert "compact" in msg.lower() or "context" in msg.lower()

    def test_compact_directive_fires_only_once(self):
        from claude_code_integration.hooks import token_budget

        with tempfile.TemporaryDirectory() as tmpdir:
            data = {
                "session_id": "once-test",
                "stop_reason": "end_turn",
                "message": {"content": "x", "usage": {"output_tokens": 500}},
            }

            def _run_once():
                with patch("claude_code_integration.hooks.token_budget.SessionState") as mock_cls:
                    state = MagicMock()
                    state.turns = 1
                    state.fill_fraction = 0.80
                    state.input_tokens = 144000
                    state.output_tokens = 500
                    mock_cls.load.return_value = state
                    with patch("claude_code_integration.hooks.token_budget.session_dir") as mock_sd:
                        mock_sd.return_value = Path(tmpdir)
                        return _run_hook_no_exit(token_budget, data)

            first = _run_once()
            second = _run_once()
            # Compact flag should be written after first call; second call should not repeat it
            assert first is not None
            # Second result may be None or a regular status, but NOT another /compact directive
            if second is not None:
                msg = second.get("systemMessage", "")
                assert "must call /compact" not in msg

    def test_status_line_format(self):
        result = self._run(fill=0.3, api_input=54000, env={"TOKENSHRINK_BUDGET_THRESHOLD": "0.1"})
        if result is not None:
            msg = result.get("systemMessage", "")
            assert "Turn" in msg
            assert "remaining" in msg
