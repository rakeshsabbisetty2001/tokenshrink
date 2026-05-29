"""Tests for SemanticCache — SimHash, SQLite backend, AnthropicWrapper integration."""
import sys
import os
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tokenshrink.cache.semantic_cache import simhash, hamming_similarity, SemanticCache


# ── SimHash ───────────────────────────────────────────────────────────────────

class TestSimHash:
    def test_identical_texts_equal(self):
        a = simhash("the quick brown fox jumps over the lazy dog")
        b = simhash("the quick brown fox jumps over the lazy dog")
        assert a == b

    def test_similar_texts_high_similarity(self):
        a = simhash("the quick brown fox jumps over the lazy dog")
        b = simhash("the quick brown fox jumps over the lazy cat")
        assert hamming_similarity(a, b) > 0.80

    def test_very_different_texts_low_similarity(self):
        a = simhash("machine learning neural networks deep learning")
        b = simhash("stock market financial analysis investment portfolio")
        assert hamming_similarity(a, b) < 0.90  # different domains

    def test_empty_string_stable(self):
        h1 = simhash("")
        h2 = simhash("")
        assert h1 == h2

    def test_returns_int(self):
        assert isinstance(simhash("hello world"), int)

    def test_hamming_identical(self):
        h = simhash("hello")
        assert hamming_similarity(h, h) == 1.0

    def test_hamming_opposite(self):
        # All bits different: 0 vs all-ones (using signed representation)
        all_ones = -1  # signed 64-bit all-ones
        assert hamming_similarity(0, all_ones) == 0.0


# ── SemanticCache ─────────────────────────────────────────────────────────────

@pytest.fixture
def cache(tmp_path):
    c = SemanticCache(path=tmp_path / "cache.db", ttl=3600, threshold=0.95)
    yield c
    c.close()


class TestSemanticCacheMissGet:
    def test_miss_returns_none(self, cache):
        assert cache.get("some prompt text here", "claude-sonnet-4-6") is None

    def test_stats_initial(self, cache):
        s = cache.stats()
        assert s["hits"] == 0
        assert s["misses"] == 0
        assert s["total_entries"] == 0


class TestSemanticCacheSetGet:
    def test_exact_hit(self, cache):
        prompt = "What is the capital of France?"
        response = {"content": "Paris"}
        cache.set(prompt, "model-a", response)
        result = cache.get(prompt, "model-a")
        assert result == response

    def test_different_model_miss(self, cache):
        prompt = "What is the capital of France?"
        cache.set(prompt, "model-a", {"content": "Paris"})
        assert cache.get(prompt, "model-b") is None

    def test_stats_after_hit(self, cache):
        prompt = "Hello world"
        cache.set(prompt, "m", {"r": 1})
        cache.get(prompt, "m")
        s = cache.stats()
        assert s["hits"] == 1
        assert s["misses"] == 0

    def test_stats_after_miss(self, cache):
        cache.get("never stored", "m")
        s = cache.stats()
        assert s["misses"] == 1

    def test_total_entries_increments(self, cache):
        cache.set("prompt one", "m", {"r": 1})
        cache.set("prompt two", "m", {"r": 2})
        assert cache.stats()["total_entries"] == 2


class TestSemanticCacheTTL:
    def test_expired_entry_is_miss(self, tmp_path):
        c = SemanticCache(path=tmp_path / "ttl.db", ttl=1, threshold=0.95)
        c.set("test prompt", "m", {"r": 1})
        time.sleep(1.1)
        result = c.get("test prompt", "m")
        c.close()
        assert result is None

    def test_clear_expired_removes_entries(self, tmp_path):
        c = SemanticCache(path=tmp_path / "exp.db", ttl=1, threshold=0.95)
        c.set("p", "m", {"r": 1})
        time.sleep(1.1)
        removed = c.clear_expired()
        c.close()
        assert removed == 1


class TestSemanticCacheThreshold:
    def test_high_threshold_rejects_borderline_similar(self, tmp_path):
        c = SemanticCache(path=tmp_path / "strict.db", ttl=3600, threshold=0.999)
        c.set("What is the capital of France?", "m", {"r": 1})
        # Moderately different text
        result = c.get("What is the capital city of France exactly?", "m")
        c.close()
        # Should miss with very high threshold
        assert result is None

    def test_low_threshold_accepts_similar(self, tmp_path):
        c = SemanticCache(path=tmp_path / "loose.db", ttl=3600, threshold=0.50)
        prompt = "the quick brown fox jumps over the lazy dog today"
        c.set(prompt, "m", {"r": 42})
        # Same prompt — should always hit
        result = c.get(prompt, "m")
        c.close()
        assert result == {"r": 42}


# ── AnthropicWrapper integration ──────────────────────────────────────────────

class _FakeMessages:
    def __init__(self):
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        return {"model": kwargs.get("model", "x"), "content": "answer", "call": self.calls}


class _FakeClient:
    def __init__(self):
        self._msgs = _FakeMessages()

    @property
    def messages(self):
        return self._msgs


class TestAnthropicWrapperWithCache:
    def test_cache_miss_calls_api(self, tmp_path):
        from tokenshrink import TokenShrink
        from tokenshrink.wrappers.anthropic_wrapper import AnthropicWrapper
        c = SemanticCache(path=tmp_path / "w.db", ttl=3600)
        client = _FakeClient()
        wrapper = AnthropicWrapper(client, TokenShrink(), cache=c)
        result = wrapper.messages.create(model="m", messages=[{"role": "user", "content": "hi"}])
        assert result["call"] == 1
        c.close()

    def test_cache_hit_skips_api(self, tmp_path):
        from tokenshrink import TokenShrink
        from tokenshrink.wrappers.anthropic_wrapper import AnthropicWrapper
        c = SemanticCache(path=tmp_path / "w2.db", ttl=3600)
        client = _FakeClient()
        wrapper = AnthropicWrapper(client, TokenShrink(), cache=c)
        msg = [{"role": "user", "content": "What is 2+2?"}]
        wrapper.messages.create(model="m", messages=msg)  # miss → stores
        result = wrapper.messages.create(model="m", messages=msg)  # hit
        assert client._msgs.calls == 1  # API called only once
        c.close()

    def test_no_cache_passthrough(self, tmp_path):
        from tokenshrink import TokenShrink
        from tokenshrink.wrappers.anthropic_wrapper import AnthropicWrapper
        client = _FakeClient()
        wrapper = AnthropicWrapper(client, TokenShrink())  # no cache
        wrapper.messages.create(model="m", messages=[{"role": "user", "content": "hello"}])
        wrapper.messages.create(model="m", messages=[{"role": "user", "content": "hello"}])
        assert client._msgs.calls == 2
