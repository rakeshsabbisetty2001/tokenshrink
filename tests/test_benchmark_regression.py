"""Regression benchmarks: each corpus case must meet a minimum token-reduction threshold.

These tests protect against compressor regressions. The thresholds are intentionally
loose lower bounds — they should not break on minor algorithm tweaks, only on meaningful
regressions that eliminate a technique's core effectiveness.
"""
from __future__ import annotations

import pytest

from tokenshrink import TokenShrink
from tests.fixtures.benchmark_corpus import ALL_CASES, BenchmarkCase


@pytest.mark.parametrize("case", ALL_CASES, ids=[c.name for c in ALL_CASES])
def test_reduction_meets_baseline(case: BenchmarkCase) -> None:
    ts = TokenShrink(techniques=[case.technique])
    result = ts.compress_prompt(case.text)

    assert result.reduction_pct >= case.min_reduction_pct, (
        f"[{case.name}] technique={case.technique!r}: "
        f"reduction {result.reduction_pct:.1f}% < required {case.min_reduction_pct}% "
        f"({result.original_tokens} -> {result.compressed_tokens} tokens)"
    )


def test_full_pipeline_beats_single_technique() -> None:
    """The combined pipeline should reduce more than any single technique alone."""
    from tests.fixtures.benchmark_corpus import JSON_API_RESPONSE, REPEATED_LOG_ERRORS

    for case in (JSON_API_RESPONSE, REPEATED_LOG_ERRORS):
        ts_full = TokenShrink()
        ts_single = TokenShrink(techniques=[case.technique])

        full = ts_full.compress_prompt(case.text)
        single = ts_single.compress_prompt(case.text)

        assert full.compressed_tokens <= single.compressed_tokens, (
            f"[{case.name}] full pipeline ({full.compressed_tokens}) should not be "
            f"worse than single technique ({single.compressed_tokens})"
        )


def test_json_minification_is_lossless_json() -> None:
    """Minified JSON must still parse and equal the original data."""
    import json as _json
    from tests.fixtures.benchmark_corpus import JSON_API_RESPONSE  # noqa: PLC0415

    ts = TokenShrink(techniques=["format"])
    result = ts.compress_prompt(JSON_API_RESPONSE.text)

    original_data = _json.loads(JSON_API_RESPONSE.text)
    compressed_data = _json.loads(result.compressed_text)
    assert original_data == compressed_data, "JSON minification changed the data"


def test_idempotency_across_cases() -> None:
    """Applying a single lossless technique twice should yield the same result as once.

    Each case is tested with only its own technique (not the full pipeline) so that
    interactions between stages don't introduce non-determinism.  semantic_dedup is
    excluded: it is approximate (MinHash) so strict idempotency is not guaranteed.
    """
    idempotent_techniques = {"whitespace", "deduplication", "format"}
    cases = [c for c in ALL_CASES if c.technique in idempotent_techniques]

    for case in cases:
        ts = TokenShrink(techniques=[case.technique])
        once = ts.compress_prompt(case.text).compressed_text
        twice = ts.compress_prompt(once).compressed_text
        assert once == twice, (
            f"[{case.name}] technique={case.technique!r} is not idempotent"
        )
