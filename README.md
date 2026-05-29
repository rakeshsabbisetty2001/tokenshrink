# TokenShrink

Reduce LLM token usage by up to **70%** with a lossless compression pipeline. Works with Anthropic, OpenAI, or any LLM API — no meaning is lost, just waste.

Built specifically to keep **Claude Code dev sessions** from hitting usage limits, but usable anywhere tokens matter.

## How it works

TokenShrink runs text through a configurable pipeline of compression techniques before it reaches the model:

| Technique | What it removes |
|---|---|
| `whitespace` | Trailing spaces, redundant blank lines, over-indentation |
| `deduplication` | Exact duplicate paragraphs and blocks |
| `format` | Verbose XML/markdown boilerplate and pretty-printed JSON (minified in place) |
| `semantic_dedup` | Near-duplicate sentences using MinHash + Jaccard similarity (prose and code-aware) |
| `rag` | Low-relevance RAG chunks via BM25 scoring |
| `conversation` | Redundant content in old conversation turns; cross-turn near-duplicates are also removed |

Token counting auto-detects the best available backend: `tiktoken` → Anthropic `count_tokens` API (with local LRU cache) → character approximation.

---

## Installation

```bash
# Basic (no tokenizer — uses approximation)
pip install -e .

# With accurate token counting
pip install -e ".[anthropic]"   # Anthropic projects
pip install -e ".[openai]"      # OpenAI / tiktoken projects
pip install -e ".[all]"         # Everything

# Development
pip install -e ".[dev]"
```

**Requirements:** Python 3.10+

---

## Usage

### Python API

**Compress a prompt:**
```python
from tokenshrink import TokenShrink

ts = TokenShrink()
result = ts.compress_prompt("Your long, verbose prompt text here...")

print(result.compressed_text)
print(f"Reduced by {result.reduction_pct:.1f}%")
print(f"Quality score: {result.quality_score:.3f}")  # word-overlap Jaccard [0,1]
# result.original_tokens, result.compressed_tokens also available
```

**Compress conversation history:**
```python
messages = [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."},
    # ... many more turns
]

# Lossless: compresses older turns, keeps the last 5 verbatim
compressed = ts.compress_conversation(messages, keep_recent=5)

# Lossy (opt-in): replace old turns with extractive TF-IDF summaries
#   Each old turn is reduced to its top 1/3 sentences in original order,
#   prefixed with [Summary: N->K sentences].
compressed = ts.compress_conversation(messages, keep_recent=5, summarize=True, summarize_ratio=3)
```

**Select relevant RAG chunks within a token budget:**
```python
chunks = ["chunk about topic A...", "chunk about topic B...", ...]

selected = ts.select_context(
    chunks,
    query="What is the deployment process?",
    max_tokens=2000,
)
```

**Auto-compress every API call (SDK wrapper):**
```python
import anthropic
from tokenshrink import TokenShrink

ts = TokenShrink()
client = anthropic.Anthropic()

with ts.wrap(client) as c:
    response = c.messages.create(
        model="claude-opus-4-8",
        max_tokens=1024,
        messages=[{"role": "user", "content": "your prompt"}],
    )
```

Works with both Anthropic and OpenAI clients.

**Semantic response cache (avoid re-calling the API for near-identical prompts):**
```python
from tokenshrink import TokenShrink, SemanticCache
from tokenshrink.wrappers.anthropic_wrapper import AnthropicWrapper

ts = TokenShrink()
cache = SemanticCache(
    path="~/.tokenshrink/cache.db",  # default
    ttl=3600,                         # seconds before entries expire
    threshold=0.95,                   # SimHash similarity required for a hit
)

wrapper = AnthropicWrapper(client, ts, cache=cache)
response = wrapper.messages.create(model="claude-sonnet-4-6", messages=[...])

# Subsequent calls with near-identical prompts hit the cache instead of the API
print(cache.stats())
# {'hits': 1, 'misses': 1, 'hit_rate': 0.5, 'total_entries': 1, ...}

cache.clear_expired()   # evict stale entries
cache.close()
```

**Trim an OpenAPI or GraphQL schema to what's relevant for a task:**
```python
from tokenshrink.compressors.schema_trimmer import trim_schema

# Pass raw JSON/YAML OpenAPI or GraphQL SDL text
trimmed = trim_schema(schema_text, task="create a new user account", top_k=10)
```

**Streaming compression (for Anthropic streaming API):**
```python
from tokenshrink import TokenShrink

ts = TokenShrink()

# Generator-based: compresses as chunks arrive, yields when a paragraph completes
with client.messages.stream(...) as stream:
    for compressed_chunk in ts.stream_compress(stream.text_stream):
        sys.stdout.write(compressed_chunk)

# Or use StreamingCompressor directly for more control
from tokenshrink import StreamingCompressor

sc = StreamingCompressor(language="auto")  # "auto", "prose", or "code"
for raw_chunk in stream.text_stream:
    output = sc.feed(raw_chunk)
    if output:
        process(output)
remainder = sc.flush()
if remainder:
    process(remainder)
```

**Configure which techniques run:**
```python
ts = TokenShrink(
    techniques=["whitespace", "deduplication"],  # only these two
    similarity_threshold=0.85,   # semantic dedup sensitivity (0–1)
    rag_max_tokens=2000,         # token budget for RAG selection
    keep_recent=5,               # conversation turns to keep verbatim
    indent_spaces=2,             # tab -> N spaces
    bm25_k1=1.5,                 # BM25 term saturation (lower for code-heavy corpora)
    bm25_b=0.75,                 # BM25 length normalization
)
```

**Build a custom pipeline with your own compressors:**
```python
from tokenshrink.pipeline import Pipeline
from tokenshrink.compressors.base import FunctionCompressor
from tokenshrink.counter import TokenCounter

def my_compressor(text: str, opts: dict) -> str:
    return text.replace("  ", " ")   # example

counter = TokenCounter()
p = Pipeline(counter)
p.add_compressor(FunctionCompressor("my_step", my_compressor, lossless=True))
result = p.run("some text with  extra  spaces")
print(f"Quality: {result.quality_score:.3f}")  # word-overlap Jaccard
```

**Pipeline fault isolation:** if any compressor raises an exception, it is silently skipped and the pipeline continues with the previous output. Compressors that return `None` are also handled gracefully.

---

### CLI

```bash
# Compress a prompt string
tokenshrink compress-prompt "your verbose prompt" --stats

# Compress from file / stdin
cat prompt.txt | tokenshrink compress-prompt - --stats
tokenshrink compress-prompt prompt.txt --output json

# Benchmark each technique individually
tokenshrink benchmark prompt.txt

# Compress a conversation (JSON array of {role, content} messages)
tokenshrink compress-conversation messages.json --keep-recent 5 --stats

# Select top RAG chunks within a token budget
tokenshrink select-context chunks.txt --query "your query" --max-tokens 2000 --stats
tokenshrink select-context chunks.txt --query "code query" --bm25-k1 1.2 --bm25-b 0.5

# Trim an OpenAPI / GraphQL schema to relevant operations
tokenshrink trim-schema openapi.json --task "create a new user" --top-k 10
tokenshrink trim-schema schema.graphql --task "fetch user profile" --output json

# Install / remove Claude Code hooks
tokenshrink install-hooks
tokenshrink install-hooks --settings /path/to/settings.local.json --dry-run
tokenshrink uninstall-hooks
```

**CLI flags:**

| Flag | Description |
|---|---|
| `--stats` | Print token counts and reduction % |
| `--output json` | Output as JSON |
| `--model` | Model name for accurate token counting (e.g. `gpt-4o`, `claude-3-5-sonnet`) |
| `--techniques` | Comma-separated list of techniques to apply |
| `--keep-recent` | Number of recent conversation turns to keep verbatim (default: 5) |
| `--max-tokens` | Token budget for RAG chunk selection (default: 2000) |
| `--chunk-sep` | Separator between RAG chunks in input file (default: `---`) |
| `--bm25-k1` | BM25 term saturation parameter (default: 1.5; try 1.2 for code) |
| `--bm25-b` | BM25 length normalization parameter (default: 0.75; try 0.5 for code) |
| `--task` | Task description for schema trimming (trim-schema) |
| `--top-k` | Max operations to keep when trimming a schema (default: 10) |
| `--settings` | Path to Claude Code settings.local.json (install/uninstall-hooks) |
| `--dry-run` | Preview hook changes without writing (install/uninstall-hooks) |

---

### Claude Code Integration

TokenShrink installs five hooks into Claude Code that together attack token consumption from every angle:

```bash
# Install all hooks (CLI — preferred)
tokenshrink install-hooks

# Or run the installer script directly
python claude_code_integration/install_hooks.py

# Preview changes without writing
tokenshrink install-hooks --dry-run

# Point to a custom settings file
tokenshrink install-hooks --settings /path/to/settings.local.json

# Remove all hooks
tokenshrink uninstall-hooks
```

#### What each hook does

| Hook event | File | What it does |
|---|---|---|
| `PreToolUse` | `terse_mode.py` | Injects a brevity nudge once per session so Claude skips narration and batches tool calls |
| `PreToolUse` | `command_rewriter.py` | Rewrites expensive Bash commands to cheaper equivalents based on context fill level |
| `PostToolUse` | `compress_tool_output.py` | Compresses `Bash`, `Read`, `Grep`, `WebFetch`, `WebSearch` outputs using per-tool profiles |
| `UserPromptSubmit` | `compress_user_prompt.py` | Compresses your prompts before they reach the model |
| `Stop` | `token_budget.py` | Tracks session context usage (input + output tokens) and injects a budget status line |

#### Adaptive compression

As the session context fills up, TokenShrink automatically escalates compression aggressiveness:

| Context fill | Techniques applied | Min-tokens threshold |
|---|---|---|
| < 40% | whitespace + deduplication | 200 tokens |
| 40–70% | + semantic_dedup | 100 tokens |
| > 70% | full pipeline + code truncation | 50 tokens |

When context first crosses 70%, a one-shot imperative message fires:
> `[TokenShrink] IMPORTANT: Context is at 72%. You must call /compact before taking any other action.`

#### Session state persistence

Token budget state persists across shell restarts in `~/.claude/tokenshrink/`. Override the location with `TOKENSHRINK_STATE_DIR`. Falls back to the OS temp dir if the home directory is read-only.

#### Per-tool compression profiles

Each tool gets a tuned default profile. Override with `TOKENSHRINK_TOOL_PROFILES=/path/to/profiles.json`:

```json
{
  "Bash":      {"techniques": ["whitespace","deduplication","semantic_dedup"], "min_tokens": 150, "max_code_block_lines": 60},
  "Read":      {"techniques": ["whitespace","deduplication"],                  "min_tokens": 300, "max_code_block_lines": 80},
  "WebFetch":  {"techniques": ["whitespace","deduplication","format","semantic_dedup"], "min_tokens": 100, "max_code_block_lines": 40},
  "WebSearch": {"techniques": ["whitespace","deduplication","format","semantic_dedup"], "min_tokens": 100, "max_code_block_lines": 40},
  "Grep":      {"techniques": ["whitespace","deduplication"],                  "min_tokens": 200, "max_code_block_lines": 60}
}
```

Large fenced code blocks in tool output are automatically truncated to the first + last 20 lines with an omission marker (configurable via `TOKENSHRINK_MAX_CODE_BLOCK_LINES`).

#### PreToolUse command rewriter

The `command_rewriter` hook intercepts Bash commands before they run and rewrites expensive ones to cheaper equivalents. Rules activate based on context fill fraction:

| Fill level | Original command | Rewritten to |
|---|---|---|
| Always | `cat <file>` | `head -100 <file>` |
| Always | `pip list` | `pip list \| head -30` |
| ≥ 40% | `find . -name "*.py"` | `find . -name "*.py" \| head -50` |
| ≥ 40% | `ls -la` | `ls -la \| head -40` |
| ≥ 70% | `npm test` | `npm test -- --reporter=min` |
| ≥ 70% | `cargo test` | `cargo test 2>&1 \| tail -50` |

A `systemMessage` is emitted so Claude knows the command was rewritten.

#### PostToolUse smart filters

The `PostToolUse` hook applies tool-specific logic before generic compression:

**Bash — command-pattern filters**

| Command | What gets stripped |
|---|---|
| `pytest` / `python -m unittest` | All passing test lines; keeps only failures + final summary |
| `jest` / `vitest` | Passing suite lines collapsed to a count; keeps failures + summary |
| `npm install` / `pip install` / `yarn add` | Progress bars, HTTP logs; keeps errors + final summary |
| `git log` | Author and Date header lines; keeps commit message + stats |
| `ls -la` / `find` | Permissions, owner, group, timestamps; keeps name + size |
| `tsc` | Errors grouped by file with location; warning counts summarised |
| `webpack` / `cargo build` / `make` | Errors listed, warnings summarised by rule, last 5 lines kept |
| `eslint` / `ruff` / `flake8` | Duplicate warning rules deduplicated; all errors kept |
| Any output with a Python traceback | Middle frames; keeps first 3 + last 3 frames per traceback |

**Read — session cache with delta compression**

When Claude reads the same file a second time without it changing, the hook replaces the full file content with a one-line note:

```
[TokenShrink: 'core.py' unchanged since last read — 312 lines, 8,450 chars. Request a line range if you need content.]
```

When the file *has* changed, the hook computes a unified diff and emits just the diff if it is less than 30% of the new file size:

```
[TokenShrink: showing diff vs turn 4]
--- core.py (turn 4)
+++ core.py (current)
@@ -12,7 +12,7 @@
...
```

File content is cached up to 200 KB per file to limit session directory growth.

**Grep — context deduplication**

Grep results with `-C` context flags often repeat the same surrounding lines across multiple matches. The hook deduplicates identical context blocks and reports how many were removed.

#### Environment variables

| Variable | Default | Effect |
|---|---|---|
| `TOKENSHRINK_DEBUG` | off | Show per-tool compression stats as system messages |
| `TOKENSHRINK_TERSE` | `1` | Set to `0` to disable the brevity nudge |
| `TOKENSHRINK_BUDGET` | `1` | Set to `0` to disable the token budget display |
| `TOKENSHRINK_CONTEXT_WINDOW` | `180000` | Assumed context window size for budget calculations |
| `TOKENSHRINK_BUDGET_THRESHOLD` | `0.15` | Fraction of context used before budget status appears |
| `TOKENSHRINK_MAX_CODE_BLOCK_LINES` | `60` | Lines threshold before code blocks are head+tail truncated |
| `TOKENSHRINK_TOOL_PROFILES` | — | Path to a JSON file overriding per-tool compression profiles |
| `TOKENSHRINK_NO_REMOTE_COUNT` | off | Set to `1` to force heuristic token counting (no API calls) |
| `TOKENSHRINK_SUMMARIZE` | off | Set to `1` to enable extractive summarization of old conversation turns |
| `TOKENSHRINK_STATE_DIR` | `~/.claude/tokenshrink` | Override persistent session state directory |

---

## Compression Quality

Every `CompressionResult` and `PipelineResult` exposes a `quality_score` — a word-overlap Jaccard similarity between the original and compressed text. It measures vocabulary preservation on a [0, 1] scale; a well-tuned lossless compressor on typical text scores above 0.85.

```python
result = ts.compress_prompt(text)
print(f"{result.reduction_pct:.1f}% smaller, {result.quality_score:.3f} quality")
```

The benchmark regression suite enforces a minimum `quality_score ≥ 0.5` per technique and `≥ 0.6` for the full pipeline, catching regressions that shrink tokens at the cost of content.

---

## Code-Aware Semantic Deduplication

The `semantic_dedup` technique supports a `language` option that adjusts its stopword set to avoid treating code keywords as content-bearing tokens:

```python
ts = TokenShrink()
result = ts.compress_prompt(code_heavy_text)           # auto-detects code

# Or explicitly:
from tokenshrink.compressors import semantic_dedup
result = semantic_dedup.compress(text, {"language": "code"})   # Python/JS/SQL keywords as stopwords
result = semantic_dedup.compress(text, {"language": "prose"})  # prose-only stopwords
```

In `"auto"` mode (default) the technique peeks at the text for common code patterns (`def`, `class`, `function`, `SELECT`, etc.) and selects the appropriate stopword set automatically.

---

## Running Tests

```bash
pip install -e ".[dev]"
pytest
pytest --cov=tokenshrink              # with coverage
pytest tests/test_benchmark_regression.py -v   # benchmark regression suite
```

The test suite has **192 tests** across 15 modules:

| Module | What it covers |
|---|---|
| `test_whitespace.py` | Whitespace normalization, code block preservation, idempotency |
| `test_deduplication.py` | Exact paragraph dedup |
| `test_format_optimizer.py` | JSON minification, XML/MD boilerplate stripping |
| `test_semantic_dedup.py` | MinHash near-duplicate detection |
| `test_rag_selector.py` | BM25 relevance scoring and token budget |
| `test_conversation.py` | Cross-turn compression and extractive summarization |
| `test_integration.py` | End-to-end compression ratios on realistic fixtures |
| `test_benchmark_regression.py` | Minimum reduction % and quality score thresholds per technique |
| `test_hooks.py` | All Claude Code hooks (stdin/stdout simulation) |
| `test_counter.py` | TokenCounter backends, LRU cache, env var override |
| `test_edge_cases.py` | Empty inputs, malformed JSON, None inputs, broken code blocks, pipeline fault isolation, Unicode |
| `test_command_rewriter.py` | Fill-fraction rule table, hook stdin/stdout |
| `test_delta_tracker.py` | Read cache hit/miss, unified diff emission, 200 KB cap |
| `test_schema_trimmer.py` | OpenAPI operation scoring/stripping, GraphQL docstring trimming, auto-detect |
| `test_semantic_cache.py` | SimHash, SQLite backend, TTL, threshold, AnthropicWrapper integration |

---

## Project Structure

```
tokenshrink/
├── tokenshrink/
│   ├── core.py              # TokenShrink main class
│   ├── cli.py               # Click CLI (compress-prompt, benchmark, trim-schema, install-hooks, ...)
│   ├── counter.py           # Auto-detecting token counter (tiktoken / Anthropic API / approx)
│   ├── pipeline.py          # Compression pipeline runner with quality_score metrics
│   ├── streaming.py         # StreamingCompressor — chunk-based generator for streaming APIs
│   ├── cache/
│   │   └── semantic_cache.py    # SimHash + SQLite semantic response cache
│   ├── compressors/
│   │   ├── base.py              # Compressor protocol + FunctionCompressor adapter
│   │   ├── whitespace.py        # Whitespace normalization
│   │   ├── deduplication.py     # Exact duplicate removal
│   │   ├── format_optimizer.py  # XML/markdown boilerplate + JSON minification
│   │   ├── semantic_dedup.py    # MinHash near-duplicate removal (cross-turn + code-aware)
│   │   ├── rag_selector.py      # BM25 chunk selection (configurable k1/b)
│   │   ├── conversation.py      # Conversation history compression + extractive summarization
│   │   └── schema_trimmer.py    # OpenAPI / GraphQL schema trimmer (BM25-scored)
│   └── wrappers/
│       ├── anthropic_wrapper.py  # Transparent proxy with optional SemanticCache
│       ├── openai_wrapper.py
│       └── base_wrapper.py
├── claude_code_integration/
│   ├── install_hooks.py          # Hook installer / uninstaller
│   ├── session_state.py          # Persistent session state (~/.claude/tokenshrink/)
│   ├── tool_profiles.py          # Per-tool compression profiles + fill escalation
│   ├── command_rules.py          # Fill-fraction rule table for command rewriting
│   └── hooks/
│       ├── terse_mode.py            # PreToolUse  — brevity nudge
│       ├── command_rewriter.py      # PreToolUse  — Bash command rewriter
│       ├── compress_tool_output.py  # PostToolUse — tool output compression + delta tracking
│       ├── compress_user_prompt.py  # UserPromptSubmit — prompt compression
│       └── token_budget.py          # Stop        — session budget tracker + /compact directive
└── tests/
    ├── fixtures/
    │   ├── benchmark_corpus.py   # Representative inputs for regression benchmarks
    │   ├── verbose_prompts.py    # System prompts and RAG output fixtures
    │   └── long_conversations.py # Multi-turn conversation fixture
    ├── test_benchmark_regression.py  # Compressor regression + quality score suite
    ├── test_hooks.py             # Claude Code hook integration tests
    ├── test_counter.py           # TokenCounter unit tests
    ├── test_edge_cases.py        # Robustness: empty, malformed, Unicode, fault isolation
    ├── test_command_rewriter.py  # PreToolUse command rewriter
    ├── test_delta_tracker.py     # Read cache delta compression
    ├── test_schema_trimmer.py    # OpenAPI / GraphQL schema trimmer
    └── test_semantic_cache.py    # SemanticCache unit + wrapper integration
```

---

## License

MIT
