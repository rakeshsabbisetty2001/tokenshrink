# TokenShrink

Reduce LLM token usage by up to **70%** with a lossless compression pipeline. Works with Anthropic, OpenAI, or any LLM API — no meaning is lost, just waste.

Built specifically to keep **Claude Code dev sessions** from hitting usage limits, but usable anywhere tokens matter.

## How it works

TokenShrink runs text through a configurable pipeline of compression techniques before it reaches the model:

| Technique | What it removes |
|---|---|
| `whitespace` | Trailing spaces, redundant blank lines, over-indentation |
| `deduplication` | Exact duplicate lines and paragraphs |
| `format` | Verbose JSON/markdown formatting |
| `semantic_dedup` | Near-duplicate sentences using MinHash + Jaccard similarity |
| `rag` | Low-relevance RAG chunks via BM25 scoring |
| `conversation` | Redundant content in old conversation turns (keeps recent turns verbatim) |

Token counting auto-detects the best available backend: `tiktoken` → Anthropic SDK → character approximation.

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

# Compresses older turns, keeps the last 5 verbatim
compressed = ts.compress_conversation(messages, keep_recent=5)
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

**Count tokens:**
```python
n = ts.count_tokens("some text")
```

**Configure which techniques run:**
```python
ts = TokenShrink(
    techniques=["whitespace", "deduplication"],  # only these two
    similarity_threshold=0.85,   # semantic dedup sensitivity (0–1)
    rag_max_tokens=2000,         # token budget for RAG selection
    keep_recent=5,               # conversation turns to keep verbatim
    indent_spaces=2,             # tab → N spaces
)
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
tokenshrink select-context chunks.txt --query "your query" --max-tokens 2000 --stats --chunk-sep "---"
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
| `PostToolUse` | `compress_tool_output.py` | Compresses `Bash`, `Read`, `Grep`, `WebFetch`, `WebSearch` outputs |
| `UserPromptSubmit` | `compress_user_prompt.py` | Compresses your prompts before they reach the model |
| `Stop` | `token_budget.py` | Tracks output tokens per session and injects a budget status line |

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

This eliminates one of the single biggest sources of redundant tokens in refactoring sessions.

**Grep — context deduplication**

Grep results with `-C` context flags often repeat the same surrounding lines across multiple matches. The hook deduplicates identical context blocks and reports how many were removed.

#### Environment variables

| Variable | Default | Effect |
|---|---|---|
| `TOKENSHRINK_DEBUG` | off | Show per-tool compression stats as system messages |
| `TOKENSHRINK_TERSE` | `1` | Set to `0` to disable the brevity nudge |
| `TOKENSHRINK_BUDGET` | `1` | Set to `0` to disable the token budget display |
| `TOKENSHRINK_CONTEXT_WINDOW` | `180000` | Assumed context window size for budget calculations |
| `TOKENSHRINK_MIN_TOKENS` | `200` | Minimum tool output tokens before compression runs |
| `TOKENSHRINK_PROMPT_MIN_TOKENS` | `100` | Minimum prompt tokens before compression runs |
| `TOKENSHRINK_BUDGET_THRESHOLD` | `0.15` | Fraction of context used before budget status appears |

---

## Running Tests

```bash
pip install -e ".[dev]"
pytest
pytest --cov=tokenshrink   # with coverage
```

---

## Project Structure

```
tokenshrink/
├── tokenshrink/
│   ├── core.py              # TokenShrink main class
│   ├── cli.py               # Click CLI commands
│   ├── counter.py           # Auto-detecting token counter
│   ├── pipeline.py          # Compression pipeline runner
│   ├── compressors/
│   │   ├── whitespace.py        # Whitespace normalization
│   │   ├── deduplication.py     # Exact duplicate removal
│   │   ├── format_optimizer.py  # JSON/markdown format compression
│   │   ├── semantic_dedup.py    # MinHash near-duplicate removal
│   │   ├── rag_selector.py      # BM25 chunk selection
│   │   └── conversation.py      # Conversation history compression
│   └── wrappers/
│       ├── anthropic_wrapper.py
│       ├── openai_wrapper.py
│       └── base_wrapper.py
├── claude_code_integration/
│   ├── install_hooks.py          # Hook installer / uninstaller
│   └── hooks/
│       ├── terse_mode.py         # PreToolUse  — brevity nudge
│       ├── compress_tool_output.py  # PostToolUse — tool output compression
│       ├── compress_user_prompt.py  # UserPromptSubmit — prompt compression
│       └── token_budget.py       # Stop        — session budget tracker
└── tests/
```

---

## License

MIT
