import pytest
from tokenshrink.compressors import format_optimizer


def test_strips_document_tags():
    text = "<document index=\"1\">\n<content>\nHello world.\n</content>\n</document>"
    result = format_optimizer.compress(text, {})
    assert "<document" not in result
    assert "</document>" not in result
    assert "Hello world" in result


def test_decodes_html_entities():
    text = "Tom &amp; Jerry &lt;test&gt; &quot;quoted&quot;"
    result = format_optimizer.compress(text, {})
    assert "&amp;" not in result
    assert "Tom & Jerry" in result
    assert "<test>" in result


def test_collapses_repeated_separators():
    text = "Para one.\n====\n====\n====\nPara two."
    result = format_optimizer.compress(text, {})
    assert result.count("====") <= 1


def test_preserves_code_blocks():
    text = "Intro\n```\n<document>code here</document>\n```\nOutro"
    result = format_optimizer.compress(text, {})
    # Code block content preserved
    assert "<document>code here</document>" in result
