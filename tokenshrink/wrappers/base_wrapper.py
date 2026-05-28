from __future__ import annotations


class _MessagesProxy:
    """Intercepts messages.create() to compress inputs before sending."""

    def __init__(self, messages_obj, ts):
        self._messages = messages_obj
        self._ts = ts

    def _compress_kwargs(self, kwargs: dict) -> dict:
        if "system" in kwargs and isinstance(kwargs["system"], str):
            kwargs = {**kwargs, "system": self._ts.compress_prompt(kwargs["system"]).compressed_text}
        if "messages" in kwargs and isinstance(kwargs["messages"], list):
            kwargs = {**kwargs, "messages": self._ts.compress_conversation(kwargs["messages"])}
        return kwargs

    def create(self, **kwargs):
        return self._messages.create(**self._compress_kwargs(kwargs))

    async def create_async(self, **kwargs):
        return await self._messages.create(**self._compress_kwargs(kwargs))

    def __getattr__(self, name):
        return getattr(self._messages, name)


class GenericWrapper:
    """Fallback proxy for unknown SDK clients. Intercepts .messages and .chat."""

    def __init__(self, client, ts):
        self._client = client
        self._ts = ts

    @property
    def messages(self):
        return _MessagesProxy(self._client.messages, self._ts)

    def __getattr__(self, name):
        return getattr(self._client, name)
