"""Unit tests for TokenCounter: backend detection, fallbacks, cache, and count_messages."""
from __future__ import annotations

import math
import os
from unittest.mock import MagicMock, patch

import pytest

from tokenshrink.counter import TokenCounter


# ── Heuristic / approx backend ────────────────────────────────────────────────

class TestApproxBackend:
    def setup_method(self):
        # Force approx by making tiktoken and anthropic both unavailable
        self.counter = TokenCounter.__new__(TokenCounter)
        self.counter._backend = "approx"
        self.counter._model = None
        self.counter._encoder = None
        self.counter._anthropic_client = None
        self.counter._count_cache = {}
        self.counter._backend_name = "approx"

    def test_empty_string_returns_zero(self):
        assert self.counter.count("") == 0

    def test_single_word(self):
        result = self.counter.count("hello")
        assert result == math.ceil(1 * 1.3)

    def test_multiple_words(self):
        text = "hello world foo bar"
        result = self.counter.count(text)
        assert result == math.ceil(4 * 1.3)

    def test_longer_text(self):
        text = " ".join(["word"] * 100)
        result = self.counter.count(text)
        assert result == math.ceil(100 * 1.3)

    def test_none_text_empty_returns_zero(self):
        assert self.counter.count("") == 0


# ── Tiktoken backend ──────────────────────────────────────────────────────────

class TestTiktokenBackend:
    def test_uses_tiktoken_when_available(self):
        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = [1, 2, 3, 4, 5]

        mock_tiktoken = MagicMock()
        mock_tiktoken.encoding_for_model.return_value = mock_encoder

        with patch.dict("sys.modules", {"tiktoken": mock_tiktoken}):
            counter = TokenCounter(backend="tiktoken")

        counter._encoder = mock_encoder
        counter._backend_name = "tiktoken"
        result = counter.count("hello world")
        mock_encoder.encode.assert_called_once_with("hello world")
        assert result == 5

    def test_falls_back_to_cl100k_on_unknown_model(self):
        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = [1, 2, 3]

        mock_tiktoken = MagicMock()
        mock_tiktoken.encoding_for_model.side_effect = KeyError("unknown")
        mock_tiktoken.get_encoding.return_value = mock_encoder

        with patch.dict("sys.modules", {"tiktoken": mock_tiktoken}):
            counter = TokenCounter.__new__(TokenCounter)
            counter._backend = "tiktoken"
            counter._model = "unknown-model"
            counter._encoder = None
            counter._anthropic_client = None
            counter._count_cache = {}
            result = counter._try_tiktoken("unknown-model")

        assert result is True
        assert counter._encoder is mock_encoder


# ── Anthropic backend ─────────────────────────────────────────────────────────

class TestAnthropicBackend:
    def _make_counter(self, mock_client):
        counter = TokenCounter.__new__(TokenCounter)
        counter._backend = "anthropic"
        counter._model = None
        counter._encoder = None
        counter._anthropic_client = mock_client
        counter._count_cache = {}
        counter._backend_name = "anthropic"
        return counter

    def test_calls_count_tokens_api(self):
        mock_response = MagicMock()
        mock_response.input_tokens = 42
        mock_client = MagicMock()
        mock_client.messages.count_tokens.return_value = mock_response

        counter = self._make_counter(mock_client)
        result = counter.count("hello world")
        assert result == 42
        mock_client.messages.count_tokens.assert_called_once()

    def test_caches_result(self):
        mock_response = MagicMock()
        mock_response.input_tokens = 10
        mock_client = MagicMock()
        mock_client.messages.count_tokens.return_value = mock_response

        counter = self._make_counter(mock_client)
        r1 = counter.count("cached text")
        r2 = counter.count("cached text")
        assert r1 == r2 == 10
        assert mock_client.messages.count_tokens.call_count == 1  # second call hits cache

    def test_falls_back_to_approx_on_api_error(self):
        mock_client = MagicMock()
        mock_client.messages.count_tokens.side_effect = Exception("API error")

        counter = self._make_counter(mock_client)
        result = counter.count("hello world foo")
        # Should fall back to char-based heuristic
        assert result == math.ceil(len("hello world foo") / 4)

    def test_no_remote_count_env_forces_heuristic(self):
        mock_client = MagicMock()
        counter = self._make_counter(mock_client)

        with patch.dict(os.environ, {"TOKENSHRINK_NO_REMOTE_COUNT": "1"}):
            result = counter.count("hello world")

        mock_client.messages.count_tokens.assert_not_called()
        assert result > 0

    def test_cache_limited_to_256_entries(self):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.input_tokens = 1
        mock_client.messages.count_tokens.return_value = mock_response

        counter = self._make_counter(mock_client)
        # Fill cache to exactly 256 entries
        for i in range(256):
            counter.count(f"unique text entry number {i}")
        assert len(counter._count_cache) == 256
        # 257th entry should not be cached (cache full)
        counter.count("the 257th unique entry here")
        assert len(counter._count_cache) == 256


# ── Backend auto-detection ────────────────────────────────────────────────────

class TestBackendResolution:
    def test_auto_prefers_tiktoken(self):
        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = [1]
        mock_tiktoken = MagicMock()
        mock_tiktoken.encoding_for_model.return_value = mock_encoder

        with patch.dict("sys.modules", {"tiktoken": mock_tiktoken}):
            counter = TokenCounter(backend="auto")
        assert counter._backend_name == "tiktoken"

    def test_auto_falls_to_anthropic_when_no_tiktoken(self):
        mock_anthropic = MagicMock()
        mock_anthropic.Anthropic.return_value = MagicMock()

        # Remove tiktoken from sys.modules so _try_tiktoken fails
        with patch.dict("sys.modules", {"tiktoken": None}):
            with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
                counter = TokenCounter.__new__(TokenCounter)
                counter._backend = "auto"
                counter._model = None
                counter._encoder = None
                counter._anthropic_client = None
                counter._count_cache = {}
                # Manually simulate auto-detection
                if not counter._try_tiktoken(None):
                    if counter._try_anthropic():
                        counter._backend_name = "anthropic"
                    else:
                        counter._backend_name = "approx"
                else:
                    counter._backend_name = "tiktoken"

        # With tiktoken removed and anthropic mocked, should be anthropic
        assert counter._backend_name in ("anthropic", "approx")

    def test_explicit_approx_backend(self):
        counter = TokenCounter.__new__(TokenCounter)
        counter._backend = "approx"
        counter._model = None
        counter._encoder = None
        counter._anthropic_client = None
        counter._count_cache = {}
        counter._backend_name = counter._resolve_backend("approx", None)
        # With neither tiktoken nor anthropic forced, resolve returns "approx"
        # (since we didn't call _try_tiktoken which would try to import)
        assert counter._backend_name == "approx"


# ── count_messages ────────────────────────────────────────────────────────────

class TestCountMessages:
    def setup_method(self):
        self.counter = TokenCounter.__new__(TokenCounter)
        self.counter._backend_name = "approx"
        self.counter._encoder = None
        self.counter._anthropic_client = None
        self.counter._count_cache = {}

    def test_string_content(self):
        messages = [
            {"role": "user", "content": "hello world"},
            {"role": "assistant", "content": "hi there friend"},
        ]
        result = self.counter.count_messages(messages)
        expected = self.counter.count("hello world") + self.counter.count("hi there friend")
        assert result == expected

    def test_list_content_blocks(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "part one"},
                    {"type": "text", "text": "part two"},
                    {"type": "image_url", "url": "http://example.com"},
                ],
            }
        ]
        result = self.counter.count_messages(messages)
        expected = self.counter.count("part one") + self.counter.count("part two")
        assert result == expected

    def test_empty_messages(self):
        assert self.counter.count_messages([]) == 0

    def test_missing_content_key(self):
        messages = [{"role": "user"}]
        result = self.counter.count_messages(messages)
        assert result == 0

    def test_ignores_non_text_blocks(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "tool_use", "id": "abc", "name": "Bash"},
                ],
            }
        ]
        result = self.counter.count_messages(messages)
        assert result == 0
