"""Tests for the PreToolUse command rewriter."""
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from claude_code_integration.command_rules import rewrite_command


# ── rewrite_command unit tests ────────────────────────────────────────────────

class TestRewriteAlways:
    def test_cat_rewritten(self):
        cmd, note = rewrite_command("cat foo.py", 0.0)
        assert cmd == "head -100 foo.py"
        assert note is not None

    def test_cat_with_path(self):
        cmd, note = rewrite_command("cat /some/long/path/file.txt", 0.0)
        assert cmd.startswith("head -100")
        assert note is not None

    def test_pip_list(self):
        cmd, note = rewrite_command("pip list", 0.0)
        assert "head -30" in cmd
        assert note is not None

    def test_pip3_list(self):
        cmd, note = rewrite_command("pip3 list", 0.0)
        assert "head -30" in cmd

    def test_no_match_unchanged(self):
        cmd, note = rewrite_command("python script.py", 0.0)
        assert cmd == "python script.py"
        assert note is None


class TestRewriteFill40:
    def test_find_below_threshold_no_rewrite(self):
        cmd, note = rewrite_command('find . -name "*.py"', 0.30)
        assert note is None

    def test_find_at_threshold_rewritten(self):
        cmd, note = rewrite_command('find . -name "*.py"', 0.40)
        assert "head -50" in cmd
        assert note is not None

    def test_find_single_quotes(self):
        cmd, note = rewrite_command("find . -name '*.js'", 0.45)
        assert "head -50" in cmd

    def test_ls_la_below_threshold(self):
        cmd, note = rewrite_command("ls -la", 0.30)
        assert note is None

    def test_ls_la_at_threshold(self):
        cmd, note = rewrite_command("ls -la", 0.40)
        assert "head -40" in cmd

    def test_ls_al(self):
        cmd, note = rewrite_command("ls -al", 0.55)
        assert "head -40" in cmd


class TestRewriteFill70:
    def test_npm_test_below_threshold(self):
        cmd, note = rewrite_command("npm test", 0.60)
        assert note is None

    def test_npm_test_at_threshold(self):
        cmd, note = rewrite_command("npm test", 0.70)
        assert "--reporter=min" in cmd

    def test_cargo_test(self):
        cmd, note = rewrite_command("cargo test", 0.80)
        assert "tail -50" in cmd

    def test_cargo_test_below_threshold(self):
        cmd, note = rewrite_command("cargo test", 0.65)
        assert note is None


# ── Hook stdin/stdout integration ────────────────────────────────────────────

def _run_hook(payload: dict) -> dict | None:
    import subprocess
    hook = os.path.join(os.path.dirname(__file__), "..", "claude_code_integration", "hooks", "command_rewriter.py")
    result = subprocess.run(
        [sys.executable, hook],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
    )
    if not result.stdout.strip():
        return None
    return json.loads(result.stdout)


class TestCommandRewriterHook:
    def test_non_bash_tool_silent(self):
        out = _run_hook({"tool_name": "Read", "tool_input": {"file_path": "x.py"}, "session_id": "test"})
        assert out is None

    def test_bash_no_match_silent(self):
        out = _run_hook({"tool_name": "Bash", "tool_input": {"command": "python foo.py"}, "session_id": "test"})
        assert out is None

    def test_cat_rewritten(self):
        out = _run_hook({"tool_name": "Bash", "tool_input": {"command": "cat README.md"}, "session_id": "test"})
        assert out is not None
        updated_cmd = out["hookSpecificOutput"]["updatedInput"]["command"]
        assert updated_cmd.startswith("head -100")
        assert "[TokenShrink]" in out["systemMessage"]

    def test_pip_list_rewritten(self):
        out = _run_hook({"tool_name": "Bash", "tool_input": {"command": "pip list"}, "session_id": "test"})
        assert out is not None
        assert "head -30" in out["hookSpecificOutput"]["updatedInput"]["command"]
