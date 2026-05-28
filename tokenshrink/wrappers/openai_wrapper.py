from __future__ import annotations


class OpenAIWrapper:
    """Wraps openai.OpenAI — handles chat.completions.create() message list."""

    def __init__(self, client, ts):
        self._client = client
        self._ts = ts

    @property
    def chat(self) -> "_ChatProxy":
        return _ChatProxy(self._client.chat, self._ts)

    @property
    def messages(self):
        # Some code uses client.messages — proxy to chat.completions
        return _ChatCompletionsProxy(self._client.chat.completions, self._ts)

    def __getattr__(self, name):
        return getattr(self._client, name)


class _ChatProxy:
    def __init__(self, chat_obj, ts):
        self._chat = chat_obj
        self._ts = ts

    @property
    def completions(self) -> "_ChatCompletionsProxy":
        return _ChatCompletionsProxy(self._chat.completions, self._ts)

    def __getattr__(self, name):
        return getattr(self._chat, name)


class _ChatCompletionsProxy:
    def __init__(self, completions_obj, ts):
        self._completions = completions_obj
        self._ts = ts

    def _compress_messages(self, messages: list[dict]) -> list[dict]:
        """OpenAI messages list: system role is inside the list, not a separate param."""
        # Compress system message content
        compressed = []
        for msg in messages:
            if msg.get("role") == "system" and isinstance(msg.get("content"), str):
                compressed_text = self._ts.compress_prompt(msg["content"]).compressed_text
                compressed.append({**msg, "content": compressed_text})
            else:
                compressed.append(msg)
        # Now compress conversation history (non-system messages)
        non_system = [m for m in compressed if m.get("role") != "system"]
        system_msgs = [m for m in compressed if m.get("role") == "system"]
        compressed_history = self._ts.compress_conversation(non_system)
        return system_msgs + compressed_history

    def create(self, **kwargs):
        if "messages" in kwargs and isinstance(kwargs["messages"], list):
            kwargs = {**kwargs, "messages": self._compress_messages(kwargs["messages"])}
        return self._completions.create(**kwargs)

    def __getattr__(self, name):
        return getattr(self._completions, name)
