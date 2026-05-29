from __future__ import annotations

from .base_wrapper import _MessagesProxy


class AnthropicWrapper:
    """Wraps anthropic.Anthropic — handles top-level system= and messages= params."""

    def __init__(self, client, ts, cache=None):
        self._client = client
        self._ts = ts
        self._cache = cache

    @property
    def messages(self) -> "_AnthropicMessagesProxy":
        return _AnthropicMessagesProxy(self._client.messages, self._ts, self._cache)

    def __getattr__(self, name):
        return getattr(self._client, name)


class _AnthropicMessagesProxy(_MessagesProxy):
    def __init__(self, messages_obj, ts, cache=None):
        super().__init__(messages_obj, ts)
        self._cache = cache

    def _compress_kwargs(self, kwargs: dict) -> dict:
        # Anthropic: system is a top-level string param
        if "system" in kwargs and isinstance(kwargs["system"], str):
            kwargs = {**kwargs, "system": self._ts.compress_prompt(kwargs["system"]).compressed_text}
        # Compress message history (excluding any system-role messages in the list)
        if "messages" in kwargs and isinstance(kwargs["messages"], list):
            kwargs = {**kwargs, "messages": self._ts.compress_conversation(kwargs["messages"])}
        return kwargs

    def _prompt_text(self, kwargs: dict) -> str:
        """Extract a single string representation of the prompt for cache keying."""
        parts = []
        if "system" in kwargs:
            parts.append(str(kwargs["system"]))
        for msg in kwargs.get("messages", []):
            content = msg.get("content", "")
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        parts.append(block.get("text", ""))
        return "\n".join(parts)

    def create(self, **kwargs):
        kwargs = self._compress_kwargs(kwargs)
        if self._cache is not None:
            model = kwargs.get("model", "")
            prompt_text = self._prompt_text(kwargs)
            cached = self._cache.get(prompt_text, model)
            if cached is not None:
                return cached
            response = self._messages.create(**kwargs)
            self._cache.set(prompt_text, model, response)
            return response
        return self._messages.create(**kwargs)

    async def create_async(self, **kwargs):
        kwargs = self._compress_kwargs(kwargs)
        if self._cache is not None:
            model = kwargs.get("model", "")
            prompt_text = self._prompt_text(kwargs)
            cached = self._cache.get(prompt_text, model)
            if cached is not None:
                return cached
            response = await self._messages.create(**kwargs)
            self._cache.set(prompt_text, model, response)
            return response
        return await self._messages.create(**kwargs)

    def stream(self, **kwargs):
        return self._messages.stream(**self._compress_kwargs(kwargs))

    def __getattr__(self, name):
        return getattr(self._messages, name)
