from __future__ import annotations

from .base_wrapper import _MessagesProxy


class AnthropicWrapper:
    """Wraps anthropic.Anthropic — handles top-level system= and messages= params."""

    def __init__(self, client, ts):
        self._client = client
        self._ts = ts

    @property
    def messages(self) -> "_AnthropicMessagesProxy":
        return _AnthropicMessagesProxy(self._client.messages, self._ts)

    def __getattr__(self, name):
        return getattr(self._client, name)


class _AnthropicMessagesProxy(_MessagesProxy):
    def _compress_kwargs(self, kwargs: dict) -> dict:
        # Anthropic: system is a top-level string param
        if "system" in kwargs and isinstance(kwargs["system"], str):
            kwargs = {**kwargs, "system": self._ts.compress_prompt(kwargs["system"]).compressed_text}
        # Compress message history (excluding any system-role messages in the list)
        if "messages" in kwargs and isinstance(kwargs["messages"], list):
            kwargs = {**kwargs, "messages": self._ts.compress_conversation(kwargs["messages"])}
        return kwargs

    def stream(self, **kwargs):
        return self._messages.stream(**self._compress_kwargs(kwargs))

    def __getattr__(self, name):
        return getattr(self._messages, name)
