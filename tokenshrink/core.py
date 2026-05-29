from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator, Iterable

from .counter import TokenCounter
from .pipeline import Pipeline, PipelineResult
from .compressors import whitespace, deduplication, format_optimizer, semantic_dedup
from .compressors import rag_selector, conversation as conv_compressor

_PRICING_FILE = Path(__file__).parent / "pricing.json"


def _load_pricing() -> dict:
    try:
        return json.loads(_PRICING_FILE.read_text())
    except Exception:
        return {"models": {}}


@dataclass
class CostEstimate:
    model: str
    raw_tokens: int
    compressed_tokens: int
    raw_usd: float
    compressed_usd: float
    savings_usd: float
    savings_pct: float

    def __str__(self) -> str:
        return (
            f"Model: {self.model}\n"
            f"Tokens: {self.raw_tokens:,} raw -> {self.compressed_tokens:,} compressed\n"
            f"Cost:   ${self.raw_usd:.4f} raw -> ${self.compressed_usd:.4f} compressed\n"
            f"Saved:  ${self.savings_usd:.4f} ({self.savings_pct:.1f}%)"
        )


@dataclass
class CompressionResult:
    compressed_text: str
    original_tokens: int
    compressed_tokens: int
    stages: list = field(default_factory=list)
    _original_text: str = field(default="", repr=False)

    @property
    def ratio(self) -> float:
        if self.original_tokens == 0:
            return 0.0
        return (self.original_tokens - self.compressed_tokens) / self.original_tokens

    @property
    def reduction_pct(self) -> float:
        return self.ratio * 100

    @property
    def quality_score(self) -> float:
        """
        Word-overlap Jaccard similarity between the original and compressed text.
        Range [0, 1].  Higher means more vocabulary preserved.
        """
        from .pipeline import _word_jaccard
        return _word_jaccard(self._original_text, self.compressed_text)


class TokenShrink:
    """Orchestrates lossless token compression across prompts, conversations, and RAG context."""

    def __init__(
        self,
        tokenizer: str = "auto",
        model: str | None = None,
        techniques: list[str] | None = None,
        similarity_threshold: float = 0.85,
        rag_min_score: float = 0.0,
        rag_max_tokens: int = 2000,
        keep_recent: int = 5,
        indent_spaces: int = 2,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
    ):
        self._counter = TokenCounter(backend=tokenizer, model=model)
        self._techniques = set(techniques) if techniques else {
            "whitespace", "deduplication", "format", "semantic_dedup", "rag", "conversation"
        }
        self._opts = {
            "similarity_threshold": similarity_threshold,
            "min_score": rag_min_score,
            "indent_spaces": indent_spaces,
            "bm25_k1": bm25_k1,
            "bm25_b": bm25_b,
        }
        self._rag_max_tokens = rag_max_tokens
        self._keep_recent = keep_recent

    def _build_pipeline(self) -> Pipeline:
        p = Pipeline(self._counter)
        if "whitespace" in self._techniques:
            p.add("whitespace", whitespace.compress, self._opts)
        if "deduplication" in self._techniques:
            p.add("deduplication", deduplication.compress, self._opts)
        if "format" in self._techniques:
            p.add("format", format_optimizer.compress, self._opts)
        if "semantic_dedup" in self._techniques:
            p.add("semantic_dedup", semantic_dedup.compress, self._opts)
        return p

    def compress_prompt(self, text: str) -> CompressionResult:
        result: PipelineResult = self._build_pipeline().run(text)
        return CompressionResult(
            compressed_text=result.compressed_text,
            original_tokens=result.original_tokens,
            compressed_tokens=result.compressed_tokens,
            stages=result.stages,
            _original_text=text,
        )

    def compress_conversation(
        self,
        messages: list[dict],
        keep_recent: int | None = None,
        summarize: bool = False,
        summarize_ratio: int = 3,
    ) -> list[dict]:
        n = keep_recent if keep_recent is not None else self._keep_recent
        opts = {**self._opts, "summarize_old_turns": summarize, "summarize_ratio": summarize_ratio}
        return conv_compressor.compress_messages(messages, n, self._counter, opts)

    def select_context(
        self,
        chunks: list[str],
        query: str,
        max_tokens: int | None = None,
    ) -> list[str]:
        budget = max_tokens if max_tokens is not None else self._rag_max_tokens
        return rag_selector.select_chunks(chunks, query, budget, self._counter, self._opts)

    @contextmanager
    def wrap(self, client):
        wrapped = _detect_and_wrap(client, self)
        try:
            yield wrapped
        finally:
            pass

    def stream_compress(
        self,
        chunks: Iterable[str],
        language: str = "auto",
    ) -> Generator[str, None, None]:
        """
        Generator that compresses a stream of text chunks, yielding compressed
        output as complete paragraphs are accumulated.

        Suitable for Anthropic streaming API: feed each text_stream chunk,
        collect the generator output, then drain the remainder automatically.

        Example::

            for compressed_chunk in ts.stream_compress(stream.text_stream):
                sys.stdout.write(compressed_chunk)
        """
        from .streaming import StreamingCompressor
        sc = StreamingCompressor(
            techniques=list(self._techniques),
            similarity_threshold=self._opts.get("similarity_threshold", 0.85),
            language=language,
        )
        yield from sc.compress_stream(chunks)

    def count_tokens(self, text: str) -> int:
        return self._counter.count(text)

    def estimate_cost(
        self,
        text_or_messages: "str | list[dict]",
        model: str = "claude-sonnet-4-6",
        include_compression: bool = True,
    ) -> CostEstimate:
        """Estimate API cost before sending. No network calls made.

        Args:
            text_or_messages: A prompt string or list of {role, content} messages.
            model: Model ID to price against (see tokenshrink/pricing.json).
            include_compression: If True, also compute cost after compression.

        Returns:
            CostEstimate with raw/compressed token counts and USD costs.
        """
        pricing = _load_pricing()
        rates = pricing.get("models", {}).get(model)
        if rates is None:
            # Unknown model — fall back to Sonnet pricing as a reasonable default
            rates = {"input_per_mtok": 3.0, "output_per_mtok": 15.0}

        input_rate = rates["input_per_mtok"] / 1_000_000

        if isinstance(text_or_messages, list):
            raw_text = " ".join(
                (m.get("content") or "") if isinstance(m.get("content"), str)
                else " ".join(
                    b.get("text", "") for b in (m.get("content") or [])
                    if isinstance(b, dict) and b.get("type") == "text"
                )
                for m in text_or_messages
            )
        else:
            raw_text = text_or_messages

        raw_tokens = self._counter.count(raw_text)

        if include_compression and raw_text.strip():
            compressed_tokens = self._build_pipeline().run(raw_text).compressed_tokens
        else:
            compressed_tokens = raw_tokens

        raw_usd = raw_tokens * input_rate
        compressed_usd = compressed_tokens * input_rate
        savings_usd = raw_usd - compressed_usd
        savings_pct = (savings_usd / raw_usd * 100) if raw_usd > 0 else 0.0

        return CostEstimate(
            model=model,
            raw_tokens=raw_tokens,
            compressed_tokens=compressed_tokens,
            raw_usd=raw_usd,
            compressed_usd=compressed_usd,
            savings_usd=savings_usd,
            savings_pct=savings_pct,
        )


def _detect_and_wrap(client, ts: TokenShrink):
    module = type(client).__module__
    if "anthropic" in module:
        from .wrappers.anthropic_wrapper import AnthropicWrapper
        return AnthropicWrapper(client, ts)
    if "openai" in module:
        from .wrappers.openai_wrapper import OpenAIWrapper
        return OpenAIWrapper(client, ts)
    from .wrappers.base_wrapper import GenericWrapper
    return GenericWrapper(client, ts)
