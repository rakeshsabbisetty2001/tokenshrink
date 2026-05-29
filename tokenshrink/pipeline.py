from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from tokenshrink.compressors.base import Compressor, FunctionCompressor


@dataclass
class CompressorConfig:
    enabled: bool = True
    options: dict = field(default_factory=dict)


@dataclass
class StageResult:
    name: str
    tokens_before: int
    tokens_after: int

    @property
    def saved(self) -> int:
        return self.tokens_before - self.tokens_after

    @property
    def ratio(self) -> float:
        if self.tokens_before == 0:
            return 0.0
        return self.saved / self.tokens_before


CompressorFn = Callable[[str, dict], str]


class Pipeline:
    """Runs an ordered list of Compressor objects, tracking per-stage token counts."""

    def __init__(self, counter):
        self._counter = counter
        self._stages: list[tuple[Compressor, dict]] = []

    def add(self, name: str, fn: CompressorFn, options: dict | None = None) -> "Pipeline":
        """Add a bare function as a lossless compressor (backward-compatible)."""
        return self.add_compressor(FunctionCompressor(name, fn, lossless=True), options)

    def add_compressor(self, compressor: Compressor, options: dict | None = None) -> "Pipeline":
        """Add a Compressor object directly."""
        self._stages.append((compressor, options or {}))
        return self

    def run(self, text: str, skip_lossy: bool = False) -> "PipelineResult":
        current = text
        stages: list[StageResult] = []
        for compressor, opts in self._stages:
            if skip_lossy and not compressor.lossless:
                continue
            before_tokens = self._counter.count(current)
            try:
                result = compressor.compress(current, opts)
                # Guard against compressors returning None or non-str
                current = result if isinstance(result, str) else current
            except Exception:
                pass  # compressor failure: keep current text, record stage as no-op
            after_tokens = self._counter.count(current)
            stages.append(StageResult(compressor.name, before_tokens, after_tokens))
        original_tokens = stages[0].tokens_before if stages else self._counter.count(text)
        final_tokens = self._counter.count(current)
        return PipelineResult(
            original_text=text,
            compressed_text=current,
            original_tokens=original_tokens,
            compressed_tokens=final_tokens,
            stages=stages,
        )


def _word_jaccard(original: str, compressed: str) -> float:
    """
    Estimate semantic preservation via word-level Jaccard similarity.
    Returns a float in [0, 1] where 1.0 means all words in the compressed
    text also appeared in the original.  Empty inputs return 1.0.
    """
    import re
    orig_words = set(re.findall(r"\b\w+\b", original.lower()))
    comp_words = set(re.findall(r"\b\w+\b", compressed.lower()))
    if not comp_words:
        return 1.0
    intersection = orig_words & comp_words
    union = orig_words | comp_words
    if not union:
        return 1.0
    return len(intersection) / len(union)


@dataclass
class PipelineResult:
    original_text: str
    compressed_text: str
    original_tokens: int
    compressed_tokens: int
    stages: list[StageResult]

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
        Word-overlap Jaccard similarity between original and compressed text.
        Range [0, 1] — higher is better semantic preservation.
        A lossless compressor on typical text should score > 0.85.
        """
        return _word_jaccard(self.original_text, self.compressed_text)
