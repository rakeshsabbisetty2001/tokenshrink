import pytest
from tokenshrink.counter import TokenCounter
from tokenshrink.compressors import rag_selector
from tests.fixtures.verbose_prompts import RAG_CHUNKS, RAG_QUERY


counter = TokenCounter()


def test_selects_relevant_chunks():
    selected = rag_selector.select_chunks(RAG_CHUNKS, RAG_QUERY, max_tokens=200, counter=counter, opts={})
    # The top chunks should be about TokenShrink, not the weather or French Revolution
    combined = " ".join(selected).lower()
    assert "tokenshrink" in combined or "compress" in combined


def test_respects_token_budget():
    budget = 100
    selected = rag_selector.select_chunks(RAG_CHUNKS, RAG_QUERY, max_tokens=budget, counter=counter, opts={})
    total = sum(counter.count(c) for c in selected)
    assert total <= budget


def test_returns_in_original_order():
    selected = rag_selector.select_chunks(RAG_CHUNKS, RAG_QUERY, max_tokens=500, counter=counter, opts={})
    # Verify the selected chunks appear in the same relative order as the original
    indices = [RAG_CHUNKS.index(c) for c in selected]
    assert indices == sorted(indices)


def test_empty_query_returns_all_within_budget():
    selected = rag_selector.select_chunks(RAG_CHUNKS, "", max_tokens=10000, counter=counter, opts={})
    assert len(selected) == len(RAG_CHUNKS)


def test_reduces_chunk_count():
    all_tokens = sum(counter.count(c) for c in RAG_CHUNKS)
    selected = rag_selector.select_chunks(RAG_CHUNKS, RAG_QUERY, max_tokens=all_tokens // 2, counter=counter, opts={})
    assert len(selected) < len(RAG_CHUNKS)
