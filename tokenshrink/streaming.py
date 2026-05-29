"""
Streaming-compatible compression for Anthropic streaming API contexts.

Usage with the Anthropic streaming API:

    from tokenshrink.streaming import StreamingCompressor

    sc = StreamingCompressor()
    with client.messages.stream(...) as stream:
        for raw_chunk in stream.text_stream:
            compressed = sc.feed(raw_chunk)
            if compressed:
                process(compressed)
        remainder = sc.flush()
        if remainder:
            process(remainder)

Or use the generator helper for non-streaming iterables:

    chunks = ["Hello   world.", "  This  is  a  test.", ""]
    for compressed in StreamingCompressor().compress_stream(iter(chunks)):
        print(compressed)
"""
from __future__ import annotations

from typing import Generator, Iterable

from .compressors import whitespace, deduplication, format_optimizer, semantic_dedup


class StreamingCompressor:
    """
    Chunk-based compressor that accumulates text and flushes complete
    paragraphs.  Lossless techniques only (whitespace, dedup, format, semantic
    dedup); the RAG selector and conversation compressor require the full text
    and are not streaming-compatible.

    The compressor holds a buffer and emits compressed output whenever it
    detects a paragraph boundary (double newline).  Call flush() at the end
    of the stream to emit any remaining buffered text.
    """

    def __init__(
        self,
        techniques: list[str] | None = None,
        similarity_threshold: float = 0.85,
        language: str = "auto",
    ):
        self._techniques = set(techniques) if techniques else {
            "whitespace", "deduplication", "format", "semantic_dedup"
        }
        self._opts: dict = {
            "similarity_threshold": similarity_threshold,
            "language": language,
        }
        self._buffer = ""
        # Shared MinHash signatures — persists across chunks for cross-chunk dedup
        self._seen_sigs: list[list[int]] = []

    def feed(self, chunk: str) -> str:
        """
        Ingest one chunk of text.  Returns any compressed output that can be
        emitted now (complete paragraphs).  May return an empty string if
        the buffer hasn't accumulated a complete paragraph yet.
        """
        if not chunk:
            return ""
        self._buffer += chunk
        return self._drain()

    def flush(self) -> str:
        """
        Compress and return whatever remains in the buffer.  Call once at the
        end of the stream.  Resets internal state for reuse.
        """
        remaining = self._compress_block(self._buffer)
        self._buffer = ""
        self._seen_sigs = []
        return remaining

    def _drain(self) -> str:
        """
        Emit all complete paragraphs (separated by double newline) from the
        buffer, keeping the final incomplete paragraph buffered.
        """
        if "\n\n" not in self._buffer:
            return ""

        parts = self._buffer.split("\n\n")
        # Keep the last part buffered (it may be incomplete)
        complete = parts[:-1]
        self._buffer = parts[-1]

        emitted: list[str] = []
        for para in complete:
            compressed = self._compress_block(para)
            if compressed:
                emitted.append(compressed)

        return "\n\n".join(emitted) + ("\n\n" if emitted else "")

    def _compress_block(self, text: str) -> str:
        if not text or not text.strip():
            return text

        if "whitespace" in self._techniques:
            text = whitespace.compress(text, self._opts)
        if "deduplication" in self._techniques:
            text = deduplication.compress(text, self._opts)
        if "format" in self._techniques:
            text = format_optimizer.compress(text, self._opts)
        if "semantic_dedup" in self._techniques:
            text = semantic_dedup.compress_with_seen(text, self._seen_sigs, self._opts)

        return text

    def compress_stream(self, chunks: Iterable[str]) -> Generator[str, None, None]:
        """
        Generator that accepts an iterable of text chunks and yields
        compressed output chunks.  Handles flush automatically.

        Example:
            for out in sc.compress_stream(raw_chunks):
                sys.stdout.write(out)
        """
        for chunk in chunks:
            output = self.feed(chunk)
            if output:
                yield output
        remainder = self.flush()
        if remainder:
            yield remainder

    def reset(self) -> None:
        """Reset buffer and dedup state for reuse on a new stream."""
        self._buffer = ""
        self._seen_sigs = []
