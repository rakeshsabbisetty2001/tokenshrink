from __future__ import annotations

import math


class TokenCounter:
    """Auto-detecting token counter. Falls back gracefully when SDKs are absent."""

    def __init__(self, backend: str = "auto", model: str | None = None):
        self._backend = backend
        self._model = model
        self._encoder = None
        self._backend_name = self._resolve_backend(backend, model)

    def _resolve_backend(self, backend: str, model: str | None) -> str:
        if backend == "tiktoken" or (backend == "auto" and self._try_tiktoken(model)):
            return "tiktoken"
        if backend == "anthropic" or (backend == "auto" and self._try_anthropic()):
            return "anthropic"
        return "approx"

    def _try_tiktoken(self, model: str | None) -> bool:
        try:
            import tiktoken
            enc_name = model or "gpt-4o"
            try:
                self._encoder = tiktoken.encoding_for_model(enc_name)
            except KeyError:
                self._encoder = tiktoken.get_encoding("cl100k_base")
            return True
        except ImportError:
            return False

    def _try_anthropic(self) -> bool:
        try:
            import anthropic  # noqa: F401
            return True
        except ImportError:
            return False

    def count(self, text: str) -> int:
        if not text:
            return 0
        if self._backend_name == "tiktoken":
            return len(self._encoder.encode(text))
        if self._backend_name == "anthropic":
            return self._anthropic_count(text)
        return self._approx_count(text)

    def _anthropic_count(self, text: str) -> int:
        # Anthropic SDK doesn't expose a standalone tokenizer; use approximation
        # that matches their ~4 chars/token heuristic for English prose.
        return math.ceil(len(text) / 4)

    def _approx_count(self, text: str) -> int:
        return math.ceil(len(text.split()) * 1.3)

    def count_messages(self, messages: list[dict]) -> int:
        total = 0
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                total += self.count(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        total += self.count(block.get("text", ""))
        return total
