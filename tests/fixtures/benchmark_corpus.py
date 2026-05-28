"""Representative inputs for benchmark regression tests."""
from __future__ import annotations
import json
import textwrap
from dataclasses import dataclass


@dataclass
class BenchmarkCase:
    name: str
    text: str
    technique: str          # single technique to benchmark
    min_reduction_pct: float  # lower bound — test fails if reduction drops below this


# ── Case 1: Large indented JSON API response ──────────────────────────────────

_JSON_OBJ = {
    "status": "success",
    "data": {
        "users": [
            {
                "id": i,
                "name": f"User {i}",
                "email": f"user{i}@example.com",
                "role": "admin" if i % 5 == 0 else "member",
                "metadata": {
                    "created_at": "2024-01-15T10:30:00Z",
                    "last_login": "2024-03-20T08:45:00Z",
                    "preferences": {
                        "theme": "dark",
                        "notifications": True,
                        "language": "en-US",
                    },
                },
            }
            for i in range(1, 12)
        ],
        "pagination": {
            "page": 1,
            "per_page": 20,
            "total": 11,
            "total_pages": 1,
        },
    },
}

JSON_API_RESPONSE = BenchmarkCase(
    name="json_api_response",
    text=json.dumps(_JSON_OBJ, indent=2),
    technique="format",
    min_reduction_pct=20.0,
)


# ── Case 2: pytest output with duplicate deprecation warnings ─────────────────

_DEDUP_WARNING = (
    "DeprecationWarning: 'assertRaises' usage is deprecated. Use 'pytest.raises' instead.\n"
    "  warnings.warn(\"'assertRaises' usage is deprecated. Use 'pytest.raises' instead.\")\n"
)
_PYTEST_LINES = []
for i in range(1, 21):
    _PYTEST_LINES.append(f"tests/test_module_{i % 5}.py::test_feature_{i} PASSED [ {i * 5}%]")
    if i % 3 == 0:
        _PYTEST_LINES.append(_DEDUP_WARNING)
_PYTEST_LINES += [
    "",
    _DEDUP_WARNING,
    _DEDUP_WARNING,
    "FAILED tests/test_core.py::test_compress_empty - AssertionError: expected 0 got 1",
    "========================= 1 failed, 20 passed in 0.85s =================================",
]

PYTEST_OUTPUT = BenchmarkCase(
    name="pytest_output",
    text="\n".join(_PYTEST_LINES),
    # Repeated deprecation warnings are exact duplicate paragraphs — deduplication removes them
    technique="deduplication",
    min_reduction_pct=5.0,
)


# ── Case 3: Long conversation repeating the same concepts ─────────────────────

_REPEATED_TOPIC = (
    "List comprehensions in Python use the syntax [expr for item in iterable]. "
    "They are concise and Pythonic. You can also add an if clause to filter items."
)
_CONV_TURNS = []
for i in range(1, 8):
    _CONV_TURNS.append({
        "role": "user",
        "content": f"Question {i}: Can you explain list comprehensions again? " + _REPEATED_TOPIC,
    })
    _CONV_TURNS.append({
        "role": "assistant",
        "content": (
            f"Answer {i}: " + _REPEATED_TOPIC + " "
            "Here is an example: squares = [x**2 for x in range(10)]. "
            "This is equivalent to a for loop but more compact. "
            + _REPEATED_TOPIC
        ),
    })

REPEATED_CONVERSATION_TEXT = "\n\n".join(
    f"[{m['role']}] {m['content']}" for m in _CONV_TURNS
)

REPEATED_CONVERSATION = BenchmarkCase(
    name="repeated_conversation",
    text=REPEATED_CONVERSATION_TEXT,
    technique="semantic_dedup",
    min_reduction_pct=10.0,
)


# ── Case 4: Log file with repeated error paragraphs ──────────────────────────

_ERROR_BLOCK = (
    "ERROR [2024-03-15 10:42:01] Connection refused: upstream service at 10.0.0.5:8080\n"
    "  Retrying in 5 seconds... (attempt 1 of 3)\n"
    "  Check UPSTREAM_HOST and UPSTREAM_PORT environment variables."
)

_LOG_PARAGRAPHS = [
    "INFO [2024-03-15 10:41:55] Starting worker process pid=12345",
    _ERROR_BLOCK,
    "INFO [2024-03-15 10:42:06] Retrying connection to upstream service",
    _ERROR_BLOCK,
    "INFO [2024-03-15 10:42:11] Retrying connection to upstream service",
    _ERROR_BLOCK,
    "INFO [2024-03-15 10:42:16] Retrying connection to upstream service",
    _ERROR_BLOCK,
    "CRITICAL [2024-03-15 10:42:17] All retries exhausted. Shutting down.",
]

REPEATED_LOG_ERRORS = BenchmarkCase(
    name="repeated_log_errors",
    text="\n\n".join(_LOG_PARAGRAPHS),
    # The same error block repeats 4 times — deduplication removes the duplicates
    technique="deduplication",
    min_reduction_pct=30.0,
)


# ── Case 5: Markdown document with verbose boilerplate headers ────────────────

MARKDOWN_BOILERPLATE = BenchmarkCase(
    name="markdown_boilerplate",
    text=textwrap.dedent("""\
        <document>
        <content>
        Introduction
        ============

        This section covers the basics.

        Section 1
        =========

        <chunk>
        First chunk of content here. It contains useful information.
        </chunk>

        <chunk>
        Second chunk of content here. More useful information is present.
        </chunk>

        Section 2
        =========

        <chunk>
        Third chunk with additional details about the topic.
        </chunk>

        <chunk>
        Fourth chunk wrapping up the document content.
        </chunk>

        Conclusion
        ==========

        Final thoughts and summary.
        </content>
        </document>
    """),
    technique="format",
    min_reduction_pct=15.0,
)


# ── Case 6: Duplicate RAG chunks ──────────────────────────────────────────────

_CHUNK = (
    "TokenShrink reduces LLM token usage through lossless compression techniques "
    "including whitespace normalization, deduplication, and semantic similarity matching."
)

DUPLICATE_RAG = BenchmarkCase(
    name="duplicate_rag_chunks",
    text=("\n\n" + "---\n").join([_CHUNK] * 6 + ["This is a unique chunk with different content."]),
    technique="deduplication",
    min_reduction_pct=50.0,
)


ALL_CASES: list[BenchmarkCase] = [
    JSON_API_RESPONSE,
    PYTEST_OUTPUT,
    REPEATED_CONVERSATION,
    REPEATED_LOG_ERRORS,
    MARKDOWN_BOILERPLATE,
    DUPLICATE_RAG,
]
