# TokenShrink

Reduce LLM token usage by up to **70%** with a lossless compression pipeline. Works with Anthropic, OpenAI, or any LLM API — no meaning is lost, just waste.

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

Auto-compress tool outputs and user prompts inside Claude Code sessions:

```bash
# Install hooks into Claude Code's settings.local.json
python claude_code_integration/install_hooks.py

# Preview changes without writing
python claude_code_integration/install_hooks.py --dry-run

# Point to a custom settings file
python claude_code_integration/install_hooks.py --settings /path/to/settings.local.json

# Remove hooks
python claude_code_integration/install_hooks.py --uninstall
```

Set `TOKENSHRINK_DEBUG=1` to print compression stats to stderr during Claude Code sessions.

The hooks fire on:
- **PostToolUse** — compresses output from `Bash`, `Read`, `WebFetch`, `WebSearch`
- **UserPromptSubmit** — compresses the user's prompt before it reaches the model

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
│   │   ├── whitespace.py    # Whitespace normalization
│   │   ├── deduplication.py # Exact duplicate removal
│   │   ├── format_optimizer.py  # JSON/markdown format compression
│   │   ├── semantic_dedup.py    # MinHash near-duplicate removal
│   │   ├── rag_selector.py      # BM25 chunk selection
│   │   └── conversation.py      # Conversation history compression
│   └── wrappers/
│       ├── anthropic_wrapper.py
│       ├── openai_wrapper.py
│       └── base_wrapper.py
├── claude_code_integration/
│   ├── install_hooks.py
│   └── hooks/
│       ├── compress_tool_output.py
│       └── compress_user_prompt.py
└── tests/
```

---

## License

MIT
