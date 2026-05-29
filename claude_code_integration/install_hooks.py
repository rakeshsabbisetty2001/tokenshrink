#!/usr/bin/env python3
"""
Install TokenShrink hooks into Claude Code's settings.local.json.

Usage:
    python install_hooks.py [--settings PATH] [--uninstall] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HOOK_ROOT = Path(__file__).parent / "hooks"
DEFAULT_SETTINGS = Path(__file__).parent.parent.parent / ".claude" / "settings.local.json"


def load_settings(path: Path) -> dict:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def save_settings(path: Path, data: dict, dry_run: bool = False):
    pretty = json.dumps(data, indent=2)
    if dry_run:
        print(f"[dry-run] Would write to {path}:")
        print(pretty)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(pretty + "\n")
        print(f"Wrote {path}")


def python_cmd() -> str:
    return sys.executable


def make_hook_config(settings_path: Path) -> dict:
    tool_output_hook = str(HOOK_ROOT / "compress_tool_output.py")
    user_prompt_hook = str(HOOK_ROOT / "compress_user_prompt.py")
    terse_mode_hook = str(HOOK_ROOT / "terse_mode.py")
    token_budget_hook = str(HOOK_ROOT / "token_budget.py")
    command_rewriter_hook = str(HOOK_ROOT / "command_rewriter.py")

    py = python_cmd()
    return {
        "PreToolUse": [
            {
                "hooks": [
                    {
                        "type": "command",
                        "command": f'"{py}" "{terse_mode_hook}"',
                    },
                    {
                        "type": "command",
                        "command": f'"{py}" "{command_rewriter_hook}"',
                        "matcher": "Bash",
                    },
                ]
            }
        ],
        "PostToolUse": [
            {
                "hooks": [
                    {
                        "type": "command",
                        "command": f'"{py}" "{tool_output_hook}"',
                        "matcher": "Bash|Read|Grep|WebFetch|WebSearch",
                    }
                ]
            }
        ],
        "UserPromptSubmit": [
            {
                "hooks": [
                    {
                        "type": "command",
                        "command": f'"{py}" "{user_prompt_hook}"',
                    }
                ]
            }
        ],
        "Stop": [
            {
                "hooks": [
                    {
                        "type": "command",
                        "command": f'"{py}" "{token_budget_hook}"',
                    }
                ]
            }
        ],
    }


def _is_tokenshrink_hook(hook: dict) -> bool:
    cmd = hook.get("command", "")
    return any(
        name in cmd
        for name in (
            "compress_tool_output",
            "compress_user_prompt",
            "terse_mode",
            "token_budget",
            "command_rewriter",
        )
    )


def install(settings_path: Path, dry_run: bool):
    data = load_settings(settings_path)
    new_hooks = make_hook_config(settings_path)
    hooks = data.setdefault("hooks", {})

    for event, entries in new_hooks.items():
        event_hooks = hooks.setdefault(event, [])
        # Flatten all new hooks for this event into one merged entry
        for entry in event_hooks:
            entry["hooks"] = [h for h in entry.get("hooks", []) if not _is_tokenshrink_hook(h)]
        # Add fresh entries
        event_hooks.extend(entries)
        hooks[event] = [e for e in event_hooks if e.get("hooks")]

    save_settings(settings_path, data, dry_run)
    print("TokenShrink hooks installed (PreToolUse ×2, PostToolUse, UserPromptSubmit, Stop).")
    print("  TOKENSHRINK_DEBUG=1       — show per-tool compression stats")
    print("  TOKENSHRINK_TERSE=0       — disable brevity nudge")
    print("  TOKENSHRINK_BUDGET=0      — disable token budget display")
    print("  TOKENSHRINK_CONTEXT_WINDOW=N — set assumed context size (default 180000)")


def uninstall(settings_path: Path, dry_run: bool):
    data = load_settings(settings_path)
    hooks = data.get("hooks", {})

    for event in list(hooks.keys()):
        for entry in hooks[event]:
            entry["hooks"] = [h for h in entry.get("hooks", []) if not _is_tokenshrink_hook(h)]
        hooks[event] = [e for e in hooks[event] if e.get("hooks")]
        if not hooks[event]:
            del hooks[event]

    save_settings(settings_path, data, dry_run)
    print("TokenShrink hooks removed.")


def main():
    parser = argparse.ArgumentParser(description="Install TokenShrink hooks into Claude Code")
    parser.add_argument("--settings", default=str(DEFAULT_SETTINGS), help="Path to settings.local.json")
    parser.add_argument("--uninstall", action="store_true", help="Remove TokenShrink hooks")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without writing")
    args = parser.parse_args()

    path = Path(args.settings)
    if args.uninstall:
        uninstall(path, args.dry_run)
    else:
        install(path, args.dry_run)


if __name__ == "__main__":
    main()
