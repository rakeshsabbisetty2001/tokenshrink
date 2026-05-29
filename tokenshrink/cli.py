from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from .core import TokenShrink


@click.group()
def main():
    """TokenShrink — reduce LLM token usage by up to 70% with lossless compression."""


@main.command("compress-prompt")
@click.argument("text", default="-")
@click.option("--model", default=None, help="Model name for token counting (e.g. gpt-4o, claude-3-5-sonnet).")
@click.option("--stats", is_flag=True, help="Print token counts and compression ratio.")
@click.option("--output", type=click.Choice(["text", "json"]), default="text", help="Output format.")
@click.option("--techniques", default=None, help="Comma-separated techniques to apply (whitespace,deduplication,format,semantic_dedup).")
def compress_prompt(text, model, stats, output, techniques):
    """Compress a prompt text. Pass text as argument or pipe via stdin (-)."""
    if text == "-":
        text = sys.stdin.read()

    tech = [t.strip() for t in techniques.split(",")] if techniques else None
    ts = TokenShrink(model=model, techniques=tech)
    result = ts.compress_prompt(text)

    if output == "json":
        out = {"compressed_text": result.compressed_text}
        if stats:
            out["original_tokens"] = result.original_tokens
            out["compressed_tokens"] = result.compressed_tokens
            out["reduction_pct"] = round(result.reduction_pct, 1)
        click.echo(json.dumps(out, indent=2))
    else:
        click.echo(result.compressed_text)
        if stats:
            _print_stats(result.original_tokens, result.compressed_tokens, result.stages)


@main.command("compress-conversation")
@click.argument("file", default="-")
@click.option("--keep-recent", default=5, show_default=True, help="Number of recent turns to keep verbatim.")
@click.option("--model", default=None, help="Model name for token counting.")
@click.option("--stats", is_flag=True, help="Print token counts before/after.")
def compress_conversation(file, keep_recent, model, stats):
    """Compress a conversation JSON array (list of {role, content} messages)."""
    if file == "-":
        raw = sys.stdin.read()
    else:
        with open(file) as f:
            raw = f.read()

    messages = json.loads(raw)
    ts = TokenShrink(model=model)

    if stats:
        before = ts.count_tokens(" ".join(
            (m.get("content") or "") if isinstance(m.get("content"), str) else
            " ".join(b.get("text", "") for b in m.get("content", []) if isinstance(b, dict))
            for m in messages
        ))

    compressed = ts.compress_conversation(messages, keep_recent=keep_recent)
    click.echo(json.dumps(compressed, indent=2))

    if stats:
        after = ts.count_tokens(" ".join(
            (m.get("content") or "") if isinstance(m.get("content"), str) else
            " ".join(b.get("text", "") for b in m.get("content", []) if isinstance(b, dict))
            for m in compressed
        ))
        click.echo(f"\nTokens: {before} ->{after}  ({_pct(before, after):.1f}% reduction)", err=True)


@main.command("select-context")
@click.argument("file", default="-")
@click.option("--query", required=True, help="Query to score chunks against.")
@click.option("--max-tokens", default=2000, show_default=True, help="Token budget for selected chunks.")
@click.option("--chunk-sep", default="---", show_default=True, help="Separator between chunks in input.")
@click.option("--model", default=None, help="Model name for token counting.")
@click.option("--bm25-k1", default=1.5, show_default=True, type=float, help="BM25 k1 parameter (term saturation).")
@click.option("--bm25-b", default=0.75, show_default=True, type=float, help="BM25 b parameter (length normalization).")
@click.option("--stats", is_flag=True, help="Print selection stats.")
def select_context(file, query, max_tokens, chunk_sep, model, bm25_k1, bm25_b, stats):
    """Select the most relevant RAG chunks within a token budget using BM25 scoring."""
    if file == "-":
        raw = sys.stdin.read()
    else:
        with open(file) as f:
            raw = f.read()

    chunks = [c.strip() for c in raw.split(chunk_sep) if c.strip()]
    ts = TokenShrink(model=model, rag_max_tokens=max_tokens, bm25_k1=bm25_k1, bm25_b=bm25_b)
    selected = ts.select_context(chunks, query=query, max_tokens=max_tokens)

    click.echo(f"\n{chunk_sep}\n".join(selected))

    if stats:
        click.echo(f"\nChunks: {len(chunks)} ->{len(selected)}  ({len(selected)/max(len(chunks),1)*100:.0f}% kept)", err=True)


@main.command("benchmark")
@click.argument("file", default="-")
@click.option("--model", default=None, help="Model name for token counting.")
@click.option("--techniques", default=None, help="Comma-separated techniques to benchmark individually.")
def benchmark(file, model, techniques):
    """Show per-technique token reduction stats for a text file."""
    if file == "-":
        text = sys.stdin.read()
    else:
        with open(file) as f:
            text = f.read()

    tech_list = [t.strip() for t in techniques.split(",")] if techniques else [
        "whitespace", "deduplication", "format", "semantic_dedup"
    ]

    ts = TokenShrink(model=model, techniques=tech_list)
    result = ts.compress_prompt(text)

    click.echo(f"\n{'Technique':<20} {'Before':>8} {'After':>8} {'Saved':>8} {'%':>6}")
    click.echo("-" * 56)
    for stage in result.stages:
        click.echo(f"{stage.name:<20} {stage.tokens_before:>8} {stage.tokens_after:>8} {stage.saved:>8} {stage.ratio*100:>5.1f}%")
    click.echo("-" * 56)
    click.echo(f"{'TOTAL':<20} {result.original_tokens:>8} {result.compressed_tokens:>8} {result.original_tokens - result.compressed_tokens:>8} {result.reduction_pct:>5.1f}%")


@main.command("estimate-cost")
@click.argument("text", default="-")
@click.option("--model", default="claude-sonnet-4-6", show_default=True, help="Model ID to price against.")
@click.option("--no-compression", is_flag=True, help="Skip compression estimate; show raw cost only.")
@click.option("--output", type=click.Choice(["text", "json"]), default="text", help="Output format.")
def estimate_cost(text, model, no_compression, output):
    """Estimate API cost for a prompt before sending. Reads from stdin if text is '-'."""
    if text == "-":
        text = sys.stdin.read()

    ts = TokenShrink()
    estimate = ts.estimate_cost(text, model=model, include_compression=not no_compression)

    if output == "json":
        click.echo(json.dumps({
            "model": estimate.model,
            "raw_tokens": estimate.raw_tokens,
            "compressed_tokens": estimate.compressed_tokens,
            "raw_usd": round(estimate.raw_usd, 6),
            "compressed_usd": round(estimate.compressed_usd, 6),
            "savings_usd": round(estimate.savings_usd, 6),
            "savings_pct": round(estimate.savings_pct, 1),
        }, indent=2))
    else:
        click.echo(str(estimate))


@main.command("trim-schema")
@click.argument("file", default="-")
@click.option("--task", required=True, help="Task description to score operations against.")
@click.option("--top-k", default=10, show_default=True, type=int, help="Number of operations to keep.")
@click.option("--output", type=click.Choice(["text", "json"]), default="text", help="Output format (text=schema, json=schema+stats).")
def trim_schema(file, task, top_k, output):
    """Trim an OpenAPI or GraphQL schema to operations relevant to a task."""
    from .compressors.schema_trimmer import trim_schema as _trim

    if file == "-":
        text = sys.stdin.read()
    else:
        with open(file) as f:
            text = f.read()

    trimmed = _trim(text, task=task, top_k=top_k)

    if output == "json":
        click.echo(json.dumps({
            "trimmed": trimmed,
            "original_chars": len(text),
            "trimmed_chars": len(trimmed),
            "reduction_pct": round((len(text) - len(trimmed)) / max(len(text), 1) * 100, 1),
        }, indent=2))
    else:
        click.echo(trimmed)


@main.command("install-hooks")
@click.option(
    "--settings",
    default=None,
    help="Path to Claude Code settings.local.json (auto-detected if omitted).",
)
@click.option("--dry-run", is_flag=True, help="Preview changes without writing.")
def install_hooks(settings, dry_run):
    """Install TokenShrink hooks into Claude Code's settings.local.json."""
    try:
        from claude_code_integration.install_hooks import install, DEFAULT_SETTINGS
    except ImportError:
        click.echo(
            "Error: claude_code_integration not found. "
            "Run this command from the TokenShrink project root.",
            err=True,
        )
        sys.exit(1)

    path = Path(settings) if settings else DEFAULT_SETTINGS
    install(path, dry_run)


@main.command("uninstall-hooks")
@click.option(
    "--settings",
    default=None,
    help="Path to Claude Code settings.local.json (auto-detected if omitted).",
)
@click.option("--dry-run", is_flag=True, help="Preview changes without writing.")
def uninstall_hooks(settings, dry_run):
    """Remove TokenShrink hooks from Claude Code's settings.local.json."""
    try:
        from claude_code_integration.install_hooks import uninstall, DEFAULT_SETTINGS
    except ImportError:
        click.echo(
            "Error: claude_code_integration not found. "
            "Run this command from the TokenShrink project root.",
            err=True,
        )
        sys.exit(1)

    path = Path(settings) if settings else DEFAULT_SETTINGS
    uninstall(path, dry_run)


def _print_stats(original: int, compressed: int, stages):
    click.echo(f"\n--- TokenShrink Stats ---", err=True)
    for stage in stages:
        click.echo(f"  {stage.name}: {stage.tokens_before} ->{stage.tokens_after} ({stage.ratio*100:.1f}% saved)", err=True)
    click.echo(f"  TOTAL: {original} ->{compressed} ({_pct(original, compressed):.1f}% reduction)", err=True)


def _pct(before: int, after: int) -> float:
    if before == 0:
        return 0.0
    return (before - after) / before * 100
