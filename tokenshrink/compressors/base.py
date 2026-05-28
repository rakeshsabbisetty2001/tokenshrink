from __future__ import annotations

from typing import Callable, Protocol, runtime_checkable


@runtime_checkable
class Compressor(Protocol):
    """Protocol that all compressor objects must satisfy."""
    name: str
    lossless: bool

    def compress(self, text: str, opts: dict) -> str: ...


class FunctionCompressor:
    """Adapts a plain compress(text, opts) -> str function into a Compressor."""

    def __init__(self, name: str, fn: Callable[[str, dict], str], lossless: bool = True):
        self.name = name
        self.lossless = lossless
        self._fn = fn

    def compress(self, text: str, opts: dict) -> str:
        return self._fn(text, opts)
