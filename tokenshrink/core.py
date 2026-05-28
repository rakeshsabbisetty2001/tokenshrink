from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field

from .counter import TokenCounter
from .pipeline import Pipeline, PipelineResult
from .compressors import whitespace, deduplication, format_optimizer, semantic_dedup
from .compressors import rag_selector, conversation as conv_compressor


@dataclass
class CompressionResult:
    compressed_text: str
    original_tokens: int
    compressed_tokens: int
    stages: list = field(default_factory=list)

    @property
    def ratio(self) -> float:
        if self.original_tokens == 0:
            return 0.0
        return (self.original_tokens - self.compressed_tokens) / self.original_tokens

    @property
    def reduction_pct(self) -> float:
        return self.ratio * 100


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

    def count_tokens(self, text: str) -> int:
        return self._counter.count(text)


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
