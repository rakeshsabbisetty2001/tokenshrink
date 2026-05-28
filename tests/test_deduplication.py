import pytest
from tokenshrink.compressors import deduplication


REPEATED = """\
TokenShrink is a library for reducing LLM token usage.

Some other content here that is unique.

TokenShrink is a library for reducing LLM token usage.

More unique content follows here.

TokenShrink is a library for reducing LLM token usage.
"""


def test_removes_duplicate_paragraphs():
    result = deduplication.compress(REPEATED, {})
    # Original text has 3 copies of the repeated para; result should have 1
    count = result.count("TokenShrink is a library for reducing LLM token usage.")
    assert count == 1


def test_keeps_unique_content():
    result = deduplication.compress(REPEATED, {})
    assert "Some other content here" in result
    assert "More unique content" in result


def test_idempotent():
    result = deduplication.compress(REPEATED, {})
    assert deduplication.compress(result, {}) == result


def test_no_change_when_unique():
    text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
    result = deduplication.compress(text, {})
    assert "Paragraph one" in result
    assert "Paragraph two" in result
    assert "Paragraph three" in result
