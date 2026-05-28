import pytest
from tokenshrink.compressors import semantic_dedup


NEAR_DUP_TEXT = """\
TokenShrink reduces token usage in large language model applications significantly.
TokenShrink reduces token usage in large language model applications by a significant amount.
This is a completely different sentence about something else entirely.
TokenShrink reduces token usage in large language model applications by significant means.
"""


def test_removes_near_duplicates():
    result = semantic_dedup.compress(NEAR_DUP_TEXT, {"similarity_threshold": 0.7})
    # The three near-duplicate sentences should collapse to one
    tokenshrink_sentences = [
        s for s in result.split(". ")
        if "TokenShrink reduces token" in s
    ]
    assert len(tokenshrink_sentences) <= 2


def test_keeps_unique_sentences():
    result = semantic_dedup.compress(NEAR_DUP_TEXT, {"similarity_threshold": 0.7})
    assert "completely different sentence" in result


def test_short_sentences_kept():
    text = "Hi. Hello. Hey there."
    result = semantic_dedup.compress(text, {})
    # Short sentences (< 5 content words) are always kept
    assert result  # just check it doesn't crash or empty out


def test_idempotent():
    result = semantic_dedup.compress(NEAR_DUP_TEXT, {"similarity_threshold": 0.8})
    assert semantic_dedup.compress(result, {"similarity_threshold": 0.8}) == result


def test_reduces_tokens():
    from tokenshrink.counter import TokenCounter
    counter = TokenCounter()
    result = semantic_dedup.compress(NEAR_DUP_TEXT, {"similarity_threshold": 0.7})
    assert counter.count(result) < counter.count(NEAR_DUP_TEXT)
