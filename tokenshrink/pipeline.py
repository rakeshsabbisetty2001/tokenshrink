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
            current = compressor.compress(current, opts)
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
