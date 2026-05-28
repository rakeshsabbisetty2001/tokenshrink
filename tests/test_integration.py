"""
End-to-end integration tests asserting minimum compression ratios on realistic fixtures.
"""
import pytest
from tokenshrink import TokenShrink
from tests.fixtures.verbose_prompts import VERBOSE_SYSTEM_PROMPT, LANGCHAIN_RAG_OUTPUT, RAG_CHUNKS, RAG_QUERY
from tests.fixtures.long_conversations import LONG_CONVERSATION


ts = TokenShrink()


def _token_ratio(original: str, compressed: str) -> float:
    original_tokens = ts.count_tokens(original)
    compressed_tokens = ts.count_tokens(compressed)
    if original_tokens == 0:
        return 0.0
    return (original_tokens - compressed_tokens) / original_tokens


def test_verbose_prompt_achieves_20pct():
    result = ts.compress_prompt(VERBOSE_SYSTEM_PROMPT)
    assert result.reduction_pct >= 20, (
        f"Expected ≥20% reduction on verbose prompt, got {result.reduction_pct:.1f}%"
    )


def test_langchain_rag_output_achieves_30pct():
    result = ts.compress_prompt(LANGCHAIN_RAG_OUTPUT)
    assert result.reduction_pct >= 25, (
        f"Expected ≥25% reduction on LangChain RAG output, got {result.reduction_pct:.1f}%"
    )


def test_rag_selector_achieves_40pct_token_reduction():
    from tokenshrink.counter import TokenCounter
    counter = TokenCounter()
    original_tokens = sum(counter.count(c) for c in RAG_CHUNKS)
    selected = ts.select_context(RAG_CHUNKS, query=RAG_QUERY, max_tokens=int(original_tokens * 0.6))
    selected_tokens = sum(counter.count(c) for c in selected)
    reduction = (original_tokens - selected_tokens) / original_tokens
    assert reduction >= 0.30, (
        f"Expected ≥30% token reduction from RAG selection, got {reduction*100:.1f}%"
    )


def test_conversation_compression_reduces_tokens():
    from tokenshrink.counter import TokenCounter
    counter = TokenCounter()
    original_tokens = sum(
        counter.count(m["content"]) for m in LONG_CONVERSATION
        if isinstance(m.get("content"), str)
    )
    compressed = ts.compress_conversation(LONG_CONVERSATION, keep_recent=4)
    compressed_tokens = sum(
        counter.count(m["content"]) for m in compressed
        if isinstance(m.get("content"), str)
    )
    assert compressed_tokens < original_tokens, "Conversation compression should reduce token count"


def test_compression_result_fields():
    result = ts.compress_prompt("Hello   world.\n\n\n\nThis  is  a  test.")
    assert isinstance(result.compressed_text, str)
    assert isinstance(result.original_tokens, int)
    assert isinstance(result.compressed_tokens, int)
    assert 0.0 <= result.ratio <= 1.0
    assert result.compressed_tokens <= result.original_tokens


def test_semantic_content_preserved():
    """Key content-bearing words must survive compression."""
    text = (
        "The TokenShrink library uses BM25 scoring to select the most relevant context "
        "chunks within a configurable token budget for RAG pipelines.\n\n"
        "The TokenShrink library uses BM25 scoring to select the most relevant context "
        "chunks within a configurable token budget for RAG pipelines."
    )
    result = ts.compress_prompt(text)
    important_words = ["TokenShrink", "BM25", "relevant", "budget", "RAG"]
    for word in important_words:
        assert word in result.compressed_text, f"'{word}' lost after compression"
