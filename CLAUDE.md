# TokenShrink — Codebase Guide

## What this project does
Lossless token compression library for LLM prompts, conversations, and RAG context. Also ships Claude Code IDE hooks that compress tool outputs and track context budget in real time.

## Key commands
```bash
pytest tests/ -v                          # full test suite
pytest tests/test_benchmark_regression.py # regression thresholds only
tokenshrink compress-prompt "text..."     # CLI compression
tokenshrink benchmark file.txt            # per-technique breakdown
tokenshrink install-hooks                 # install Claude Code hooks
python claude_code_integration/install_hooks.py  # alternative installer
```

## Project layout
```
tokenshrink/                 # installable Python package
  core.py                    # TokenShrink class — public API entry point
  cli.py                     # Click CLI (compress-prompt, benchmark, install-hooks, ...)
  pipeline.py                # Pipeline: runs ordered compressors, tracks per-stage tokens
  counter.py                 # TokenCounter: tiktoken → Anthropic API → heuristic fallback
  streaming.py               # StreamingCompressor: chunk-based generator API
  compressors/
    base.py                  # Compressor protocol + FunctionCompressor adapter
    whitespace.py            # Normalize EOL, trailing spaces, blank lines (lossless)
    deduplication.py         # SHA-256 paragraph dedup (lossless)
    format_optimizer.py      # JSON minify, XML/MD boilerplate strip (lossless)
    semantic_dedup.py        # MinHash + Jaccard near-duplicate removal (lossless)
    rag_selector.py          # BM25 relevance scoring + greedy chunk selection
    conversation.py          # Cross-turn dedup + optional TF-IDF summarization
  wrappers/
    anthropic_wrapper.py     # Transparent Anthropic SDK proxy
    openai_wrapper.py        # Transparent OpenAI SDK proxy
    base_wrapper.py          # Fallback generic proxy

claude_code_integration/     # Claude Code IDE hooks (not imported by the library)
  install_hooks.py           # Hook installer CLI (--uninstall, --dry-run)
  session_state.py           # SessionState dataclass; persists to ~/.claude/tokenshrink/
  tool_profiles.py           # ToolProfile namedtuple + fill-based escalation
  hooks/
    terse_mode.py            # PreToolUse: one-shot brevity nudge per session
    compress_tool_output.py  # PostToolUse: per-tool filters + pipeline (largest hook)
    compress_user_prompt.py  # UserPromptSubmit: compress prompts above min-token threshold
    token_budget.py          # Stop: context usage tracking + /compact directive at 70%

tests/
  test_whitespace.py / test_deduplication.py / test_format_optimizer.py
  test_semantic_dedup.py / test_rag_selector.py / test_conversation.py
  test_integration.py        # end-to-end with realistic fixtures
  test_benchmark_regression.py  # min reduction % thresholds per technique
  test_hooks.py              # hook integration tests (stdin/stdout simulation)
  test_counter.py            # TokenCounter backend + cache tests
  test_edge_cases.py         # malformed input, empty strings, broken JSON
  fixtures/
    benchmark_corpus.py      # BenchmarkCase dataclass + 6 parameterized cases
    verbose_prompts.py       # system prompt, LangChain RAG, RAG chunks/query
    long_conversations.py    # 18-turn conversation fixture
```

## Architecture patterns

### Adding a new compressor
1. Create `tokenshrink/compressors/my_technique.py` exporting `compress(text: str, opts: dict) -> str`
2. Register it in `TokenShrink._build_pipeline()` in `core.py`
3. Add it to `_techniques` default set
4. Write `tests/test_my_technique.py` with idempotency + reduction threshold tests

### Pipeline flow
`TokenShrink.compress_prompt(text)` → `Pipeline.run(text)` → each compressor in order → `CompressionResult` with `compressed_text`, token counts, per-stage `StageResult` list, and `quality_score`.

### Hook data flow
Each hook reads JSON from stdin, optionally mutates the relevant field, and writes JSON to stdout. Silent hooks exit 0 with no output.
- PostToolUse: `hookSpecificOutput.updatedToolOutput`
- UserPromptSubmit: `updatedInput.user_prompt`
- PreToolUse / Stop: `systemMessage`

### Adaptive compression
`SessionState.fill_fraction` drives escalation at 40% and 70% context thresholds. Hooks read this from `~/.claude/tokenshrink/{session_hash}/budget.json` (falls back to OS temp dir).

## Environment variables
| Variable | Default | Purpose |
|----------|---------|---------|
| `TOKENSHRINK_DEBUG` | off | Show per-tool compression stats |
| `TOKENSHRINK_TERSE` | 1 | Set to 0 to disable brevity nudge |
| `TOKENSHRINK_BUDGET` | 1 | Set to 0 to disable token budget display |
| `TOKENSHRINK_CONTEXT_WINDOW` | 180000 | Assumed context window size |
| `TOKENSHRINK_BUDGET_THRESHOLD` | 0.15 | Fill fraction before showing budget |
| `TOKENSHRINK_NO_REMOTE_COUNT` | off | Force heuristic token counting |
| `TOKENSHRINK_STATE_DIR` | ~/.claude/tokenshrink | Override persistent state directory |
| `TOKENSHRINK_MAX_CODE_BLOCK_LINES` | 60 | Head+tail threshold for code block truncation |
| `TOKENSHRINK_TOOL_PROFILES` | — | Path to custom per-tool profiles JSON |
| `TOKENSHRINK_SUMMARIZE` | off | Enable extractive summarization in conversation |
