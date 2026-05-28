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
| `semantic_dedup` | Near-duplicate sentences using MinHash + Jaccard similarity |
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
```

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
```

**CLI flags:**

| Flag | Description |
|---|---|
| `--stats` | Print token counts and reduction % |
| `--output json` | Output as JSON (compress-prompt only) |
| `--model` | Model name for accurate token counting (e.g. `gpt-4o`, `claude-3-5-sonnet`) |
| `--techniques` | Comma-separated list of techniques to apply |
| `--keep-recent` | Number of recent conversation turns to keep verbatim (default: 5) |
| `--max-tokens` | Token budget for RAG chunk selection (default: 2000) |
| `--chunk-sep` | Separator between RAG chunks in input file (default: `---`) |
| `--bm25-k1` | BM25 term saturation parameter (default: 1.5; try 1.2 for code) |
| `--bm25-b` | BM25 length normalization parameter (default: 0.75; try 0.5 for code) |

---

### Claude Code Integration

TokenShrink installs four hooks into Claude Code that together attack token consumption from every angle:

```bash
# Install all hooks into Claude Code's settings.local.json
python claude_code_integration/install_hooks.py

# Preview changes without writing
python claude_code_integration/install_hooks.py --dry-run

# Point to a custom settings file
python claude_code_integration/install_hooks.py --settings /path/to/settings.local.json

# Remove all hooks
python claude_code_integration/install_hooks.py --uninstall
```

#### What each hook does

| Hook event | File | What it does |
|---|---|---|
| `PreToolUse` | `terse_mode.py` | Injects a brevity nudge once per session so Claude skips narration and batches tool calls |
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

#### PostToolUse smart filters

The `PostToolUse` hook applies tool-specific logic before generic compression:

**Bash — command-pattern filters**

| Command | What gets stripped |
|---|---|
| `pytest` / `python -m unittest` | All passing test lines; keeps only failures + final summary |
| `npm install` / `pip install` / `yarn add` | Progress bars, HTTP logs; keeps errors + final summary |
| `git log` | Author and Date header lines; keeps commit message + stats |
| `ls -la` / `find` | Permissions, owner, group, timestamps; keeps name + size |
| Any output with a Python traceback | Middle frames; keeps first 3 + last 3 frames per traceback |

**Read — session cache**

When Claude reads the same file a second time without it changing, the hook replaces the full file content with a one-line note:

```
[TokenShrink: 'core.py' unchanged since last read — 312 lines, 8,450 chars. Request a line range if you need content.]
```

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

---

## Running Tests

```bash
pip install -e ".[dev]"
pytest
pytest --cov=tokenshrink              # with coverage
pytest tests/test_benchmark_regression.py -v   # benchmark regression suite
```

The benchmark regression suite (`tests/test_benchmark_regression.py`) validates that each compressor achieves a minimum token reduction on representative real-world inputs (JSON API responses, repeated log errors, markdown boilerplate, duplicate RAG chunks, etc.). Run it after any compressor change to catch regressions.

---

## Project Structure

```
tokenshrink/
├── tokenshrink/
│   ├── core.py              # TokenShrink main class
│   ├── cli.py               # Click CLI commands
│   ├── counter.py           # Auto-detecting token counter (tiktoken / Anthropic API / approx)
│   ├── pipeline.py          # Compression pipeline runner with Compressor protocol support
│   ├── compressors/
│   │   ├── base.py              # Compressor protocol + FunctionCompressor adapter
│   │   ├── whitespace.py        # Whitespace normalization
│   │   ├── deduplication.py     # Exact duplicate removal
│   │   ├── format_optimizer.py  # XML/markdown boilerplate + JSON minification
│   │   ├── semantic_dedup.py    # MinHash near-duplicate removal (cross-turn aware)
│   │   ├── rag_selector.py      # BM25 chunk selection (configurable k1/b)
│   │   └── conversation.py      # Conversation history compression + extractive summarization
│   └── wrappers/
│       ├── anthropic_wrapper.py
│       ├── openai_wrapper.py
│       └── base_wrapper.py
├── claude_code_integration/
│   ├── install_hooks.py          # Hook installer / uninstaller
│   ├── session_state.py          # Shared session state (fill fraction, token tallies)
│   ├── tool_profiles.py          # Per-tool compression profiles + fill escalation
│   └── hooks/
│       ├── terse_mode.py         # PreToolUse  — brevity nudge
│       ├── compress_tool_output.py  # PostToolUse — tool output compression
│       ├── compress_user_prompt.py  # UserPromptSubmit — prompt compression
│       └── token_budget.py       # Stop        — session budget tracker + /compact directive
└── tests/
    ├── fixtures/
    │   ├── benchmark_corpus.py   # Representative inputs for regression benchmarks
    │   └── ...
    └── test_benchmark_regression.py  # Compressor regression suite
```

---

## License

MIT
