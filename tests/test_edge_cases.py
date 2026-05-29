"""
Edge case and robustness tests: malformed inputs, empty strings, broken JSON,
non-string types, and pipeline fault isolation.
"""
from __future__ import annotations

import pytest

from tokenshrink import TokenShrink
from tokenshrink.compressors import (
    whitespace,
    deduplication,
    format_optimizer,
    semantic_dedup,
)
from tokenshrink.pipeline import Pipeline
from tokenshrink.counter import TokenCounter


ts = TokenShrink()
counter = TokenCounter()


# ── Empty / whitespace-only inputs ───────────────────────────────────────────

class TestEmptyInputs:
    def test_whitespace_empty(self):
        assert whitespace.compress("", {}) == ""

    def test_whitespace_only_spaces(self):
        result = whitespace.compress("   \n  \n  ", {})
        assert result.strip() == ""

    def test_dedup_empty(self):
        assert deduplication.compress("", {}) == ""

    def test_format_empty(self):
        assert format_optimizer.compress("", {}) == ""

    def test_semantic_dedup_empty(self):
        assert semantic_dedup.compress("", {}) == ""

    def test_tokenshrink_empty_prompt(self):
        result = ts.compress_prompt("")
        assert result.compressed_text == ""
        assert result.original_tokens == 0
        assert result.compressed_tokens == 0

    def test_tokenshrink_whitespace_only(self):
        result = ts.compress_prompt("   \n  \n  ")
        assert isinstance(result.compressed_text, str)

    def test_compress_conversation_empty_list(self):
        result = ts.compress_conversation([])
        assert result == []

    def test_compress_conversation_fewer_than_keep_recent(self):
        messages = [{"role": "user", "content": "hi"}]
        result = ts.compress_conversation(messages, keep_recent=5)
        assert result == messages


# ── Single character / minimal inputs ────────────────────────────────────────

class TestMinimalInputs:
    def test_single_char(self):
        result = whitespace.compress("x", {})
        assert result == "x"

    def test_single_newline(self):
        result = whitespace.compress("\n", {})
        assert isinstance(result, str)

    def test_dedup_single_word(self):
        result = deduplication.compress("word", {})
        assert "word" in result

    def test_semantic_dedup_short_text(self):
        # Text with < 5 content words per sentence should be kept as-is
        result = semantic_dedup.compress("Hi.", {})
        assert "Hi" in result


# ── Malformed JSON ───────────────────────────────────────────────────────────

class TestMalformedJSON:
    def test_truncated_json_not_minified(self):
        bad = '{"key": "value", "nested": {'
        result = format_optimizer.compress(bad, {})
        assert isinstance(result, str)
        # Should not raise; returns original or partial result

    def test_json_with_trailing_garbage(self):
        bad = '{"a": 1}extra garbage here'
        result = format_optimizer.compress(bad, {})
        assert isinstance(result, str)

    def test_partial_array_json(self):
        bad = "[1, 2, 3"
        result = format_optimizer.compress(bad, {})
        assert isinstance(result, str)

    def test_valid_json_is_minified(self):
        good = '{\n  "key": "value",\n  "num": 42\n}'
        result = format_optimizer.compress(good, {})
        assert isinstance(result, str)
        assert "\n" not in result or len(result) < len(good)


# ── Non-string type inputs ────────────────────────────────────────────────────

class TestNonStringInputs:
    def test_whitespace_handles_none_gracefully(self):
        result = whitespace.compress(None, {})  # type: ignore[arg-type]
        assert isinstance(result, str)

    def test_dedup_handles_none_gracefully(self):
        result = deduplication.compress(None, {})  # type: ignore[arg-type]
        assert isinstance(result, str)

    def test_format_handles_none_gracefully(self):
        result = format_optimizer.compress(None, {})  # type: ignore[arg-type]
        assert isinstance(result, str)

    def test_semantic_dedup_handles_none_gracefully(self):
        result = semantic_dedup.compress(None, {})  # type: ignore[arg-type]
        assert isinstance(result, str)


# ── Broken code blocks ────────────────────────────────────────────────────────

class TestBrokenCodeBlocks:
    def test_unclosed_code_block_does_not_hang(self):
        text = "Before\n```python\ndef foo():\n    pass\n"  # no closing ```
        result = whitespace.compress(text, {})
        assert isinstance(result, str)
        assert "foo" in result

    def test_nested_backticks_do_not_crash(self):
        text = "```\nouter\n```inner```\n```"
        result = whitespace.compress(text, {})
        assert isinstance(result, str)

    def test_code_block_with_no_language(self):
        text = "```\nsome code\n```"
        result = whitespace.compress(text, {})
        assert "some code" in result


# ── Very large inputs ─────────────────────────────────────────────────────────

class TestLargeInputs:
    def test_large_repeated_text(self):
        text = ("This is a sentence that repeats. " * 500)
        result = ts.compress_prompt(text)
        assert isinstance(result.compressed_text, str)
        assert result.compressed_tokens <= result.original_tokens

    def test_large_json_blob(self):
        import json
        data = {"items": [{"id": i, "value": f"entry_{i}"} for i in range(200)]}
        text = json.dumps(data, indent=2)
        result = format_optimizer.compress(text, {})
        assert isinstance(result, str)
        # Minified should be shorter
        assert len(result) < len(text)

    def test_large_conversation(self):
        messages = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"Message number {i}. " * 20}
            for i in range(50)
        ]
        result = ts.compress_conversation(messages, keep_recent=5)
        assert isinstance(result, list)
        assert len(result) <= len(messages)


# ── Pipeline fault isolation ──────────────────────────────────────────────────

class TestPipelineFaultIsolation:
    def test_broken_compressor_does_not_crash_pipeline(self):
        def exploding_compressor(text: str, opts: dict) -> str:
            raise RuntimeError("I am broken")

        p = Pipeline(counter)
        p.add("whitespace", whitespace.compress, {})
        p.add("broken", exploding_compressor, {})
        p.add("dedup", deduplication.compress, {})

        text = "Hello   world.\n\n\nGoodbye   world."
        result = p.run(text)
        assert isinstance(result.compressed_text, str)
        assert "Hello" in result.compressed_text

    def test_compressor_returning_none_does_not_crash(self):
        def null_compressor(text: str, opts: dict):
            return None  # type: ignore[return-value]

        p = Pipeline(counter)
        p.add("null", null_compressor, {})

        result = p.run("some text")
        assert result.compressed_text == "some text"

    def test_empty_pipeline_returns_original(self):
        p = Pipeline(counter)
        result = p.run("original text")
        assert result.compressed_text == "original text"


# ── Unicode and special characters ───────────────────────────────────────────

class TestUnicodeInputs:
    def test_unicode_text(self):
        text = "Héllo Wörld — this is a tëst.\n\n\nHéllo Wörld — this is a tëst."
        result = ts.compress_prompt(text)
        assert isinstance(result.compressed_text, str)

    def test_emoji_in_text(self):
        text = "Hello 🌍 world 🎉!\n\n\nHello 🌍 world 🎉!"
        result = whitespace.compress(text, {})
        assert "Hello" in result

    def test_chinese_characters(self):
        text = "你好世界。\n\n\n你好世界。"
        result = deduplication.compress(text, {})
        assert isinstance(result, str)

    def test_null_bytes_in_text(self):
        text = "Hello\x00World"
        result = whitespace.compress(text, {})
        assert isinstance(result, str)


# ── Conversation edge cases ───────────────────────────────────────────────────

class TestConversationEdgeCases:
    def test_messages_with_none_content(self):
        messages = [
            {"role": "user", "content": None},
            {"role": "assistant", "content": "response"},
        ]
        result = ts.compress_conversation(messages, keep_recent=1)
        assert isinstance(result, list)

    def test_messages_with_empty_string_content(self):
        messages = [
            {"role": "user", "content": ""},
            {"role": "assistant", "content": "response"},
        ]
        result = ts.compress_conversation(messages, keep_recent=1)
        assert isinstance(result, list)

    def test_messages_missing_role(self):
        messages = [{"content": "no role here"}, {"role": "user", "content": "hi"}]
        result = ts.compress_conversation(messages, keep_recent=1)
        assert isinstance(result, list)

    def test_multi_block_content(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello world. " * 10},
                    {"type": "image_url", "url": "http://example.com/img.png"},
                ],
            },
            {"role": "assistant", "content": "Got it."},
        ]
        result = ts.compress_conversation(messages, keep_recent=1)
        assert isinstance(result, list)
        # Image block should be preserved untouched
        user_msg = result[0]
        if isinstance(user_msg.get("content"), list):
            types = [b.get("type") for b in user_msg["content"] if isinstance(b, dict)]
            assert "image_url" in types
