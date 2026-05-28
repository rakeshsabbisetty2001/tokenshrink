import pytest
from tokenshrink.compressors import whitespace


def test_collapses_blank_lines():
    text = "Hello\n\n\n\nWorld"
    result = whitespace.compress(text, {})
    assert "\n\n\n" not in result
    assert "Hello" in result
    assert "World" in result


def test_strips_trailing_spaces():
    text = "line one   \nline two  \n"
    result = whitespace.compress(text, {})
    for line in result.split("\n"):
        assert not line.endswith(" ")


def test_collapses_inline_spaces():
    text = "Hello   world,  how   are  you?"
    result = whitespace.compress(text, {})
    assert "  " not in result


def test_preserves_code_blocks():
    text = "Before\n```\nif x:    \n    pass\n\n\n```\nAfter"
    result = whitespace.compress(text, {})
    assert "if x:    " in result  # whitespace inside code block preserved


def test_idempotent():
    text = "Hello\n\n\n\nWorld   this is  a test.\n\n"
    once = whitespace.compress(text, {})
    twice = whitespace.compress(once, {})
    assert once == twice


def test_reduces_tokens():
    from tokenshrink.counter import TokenCounter
    counter = TokenCounter()
    text = "Hello   world.\n\n\n\nThis  is  a  test.\n\n\n"
    compressed = whitespace.compress(text, {})
    assert counter.count(compressed) <= counter.count(text)
