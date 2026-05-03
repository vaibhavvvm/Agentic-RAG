"""
RAG3 Semantic Chunker
======================
Splits text into semantically coherent chunks by detecting embedding-space
breakpoints between consecutive sentences.

Algorithm
---------
1. Split text into sentences using a lightweight regex splitter.
2. Embed each sentence via ``CachedOllamaEmbedder``.
3. Compute cosine similarities between consecutive sentence embeddings.
4. Identify breakpoints where similarity drops below the Nth percentile
   (configurable via ``CHUNK_SEMANTIC_BREAKPOINT_PERCENTILE``).
5. Group sentences between breakpoints into chunks.
6. Merge chunks that are shorter than ``min_chunk_size`` with their neighbours.
7. Split chunks that exceed ``max_chunk_size`` at sentence boundaries.

Returns Haystack ``Document`` objects with rich metadata.

Haystack 2.x contract
----------------------
``run(elements: list[ParsedElement], ...) -> {"documents": list[Document]}``

Usage::

    from src.ingestion.semantic_chunker import SemanticChunker

    chunker = SemanticChunker()
    result = chunker.run(elements=parsed_elements)
    docs = result["documents"]
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

from haystack import Document, component

from src.config import get_settings
from src.ingestion.embedder import CachedOllamaEmbedder
from src.ingestion.parser import ParsedElement
from src.monitoring.logger import get_logger, timed_operation
from src.monitoring.metrics import MetricsCollector

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Sentence splitter
# ---------------------------------------------------------------------------


def split_sentences(text: str) -> list[str]:
    """
    Split text into sentences using a robust regex approach.

    Handles common abbreviations (Dr., Mr., etc.) and decimal numbers
    to avoid false splits.
    """
    # Protect common abbreviations
    protected = re.sub(r"\b(Dr|Mr|Mrs|Ms|Prof|Sr|Jr|vs|etc|e\.g|i\.e|Fig|Eq)\.", r"\1<DOT>", text)
    protected = re.sub(r"(\d)\.(\d)", r"\1<DOT>\2", protected)  # decimals

    # Split on sentence-ending punctuation followed by whitespace + capital
    raw_sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z\"\'])", protected)

    # Restore protected dots
    sentences = [s.replace("<DOT>", ".").strip() for s in raw_sentences if s.strip()]
    return sentences if sentences else [text]


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two L2-normalised vectors (fast dot product)."""
    return sum(x * y for x, y in zip(a, b))


def _percentile_threshold(values: list[float], pct: float) -> float:
    """Return the ``pct``-th percentile of ``values`` (linear interpolation)."""
    if not values:
        return 0.5
    sorted_vals = sorted(values)
    idx = (pct / 100) * (len(sorted_vals) - 1)
    lo, hi = int(idx), min(int(idx) + 1, len(sorted_vals) - 1)
    frac = idx - lo
    return sorted_vals[lo] + frac * (sorted_vals[hi] - sorted_vals[lo])


# ---------------------------------------------------------------------------
# Haystack component
# ---------------------------------------------------------------------------


@component
class SemanticChunker:
    """
    Embedding-based semantic chunker.

    Args:
        embedder:          Pre-configured ``CachedOllamaEmbedder``.  If
                           ``None``, one is created from settings.
        breakpoint_percentile: Similarity percentile below which a new
                               chunk starts (lower → fewer, larger chunks).
        min_chunk_size:    Minimum characters per chunk (short chunks are
                           merged into neighbours).
        max_chunk_size:    Maximum characters per chunk (long chunks are
                           re-split at sentence boundaries).
        chunk_overlap_sentences: Number of trailing sentences from the
                                 previous chunk prepended to the next
                                 (for context continuity).
    """

    def __init__(
        self,
        embedder: CachedOllamaEmbedder | None = None,
        breakpoint_percentile: float | None = None,
        min_chunk_size: int | None = None,
        max_chunk_size: int | None = None,
        chunk_overlap_sentences: int = 1,
    ) -> None:
        cfg = get_settings().chunking
        self.embedder = embedder or CachedOllamaEmbedder()
        self.breakpoint_percentile = (
            breakpoint_percentile or cfg.semantic_breakpoint_percentile
        )
        self.min_chunk_size = min_chunk_size or cfg.semantic_min_chunk_size
        self.max_chunk_size = max_chunk_size or cfg.semantic_max_chunk_size
        self.overlap_sentences = chunk_overlap_sentences
        self._metrics = MetricsCollector.get_instance()

        log.info(
            "SemanticChunker initialised",
            extra={
                "breakpoint_percentile": self.breakpoint_percentile,
                "min_chunk_size": self.min_chunk_size,
                "max_chunk_size": self.max_chunk_size,
            },
        )

    # ------------------------------------------------------------------
    # Haystack run()
    # ------------------------------------------------------------------

    @component.output_types(documents=list)
    def run(
        self,
        elements: list[ParsedElement],
        doc_id_prefix: str = "sem",
    ) -> dict[str, list[Document]]:
        """
        Chunk parsed document elements into semantic ``Document`` objects.

        Only ``text`` and ``title`` element types are chunked; ``table`` and
        ``image`` elements are passed through as single-element chunks.

        Args:
            elements:       Output of ``DocumentParser.run()``.
            doc_id_prefix:  Prefix for generated document IDs.

        Returns:
            ``{"documents": list[Document]}`` ordered by source position.
        """
        if not elements:
            return {"documents": []}

        all_docs: list[Document] = []
        chunk_counter = 0

        with timed_operation("semantic_chunker.run", log, extra={"n_elements": len(elements)}):
            for el in elements:
                if el.element_type in ("table", "image"):
                    # Pass-through: tables and images become single documents
                    doc = Document(
                        content=el.text,
                        id=f"{doc_id_prefix}_{chunk_counter:05d}",
                        meta={
                            "element_type": el.element_type,
                            "page_number": el.page_number,
                            "chunk_index": chunk_counter,
                            "chunker": "passthrough",
                            **el.metadata,
                        },
                    )
                    all_docs.append(doc)
                    chunk_counter += 1
                    continue

                if not el.text or len(el.text.strip()) < 20:
                    continue

                chunks = self._chunk_text(el.text)
                for chunk_text in chunks:
                    doc = Document(
                        content=chunk_text,
                        id=f"{doc_id_prefix}_{chunk_counter:05d}",
                        meta={
                            "element_type": el.element_type,
                            "page_number": el.page_number,
                            "chunk_index": chunk_counter,
                            "chunker": "semantic",
                            **el.metadata,
                        },
                    )
                    all_docs.append(doc)
                    chunk_counter += 1

        self._metrics.record_event("semantic_chunker.chunks_produced", len(all_docs))
        log.info(
            "SemanticChunker complete",
            extra={"input_elements": len(elements), "output_chunks": len(all_docs)},
        )
        return {"documents": all_docs}

    # ------------------------------------------------------------------
    # Core chunking logic
    # ------------------------------------------------------------------

    def _chunk_text(self, text: str) -> list[str]:
        """Split a single text block into semantic chunks."""
        sentences = split_sentences(text)
        if len(sentences) <= 2:
            return self._split_by_size([text])

        # Embed all sentences in one batch
        embed_result = self.embedder.run(texts=sentences)
        embeddings: list[list[float]] = embed_result["embeddings"]

        # Compute consecutive cosine similarities
        similarities: list[float] = [
            _cosine_similarity(embeddings[i], embeddings[i + 1])
            for i in range(len(embeddings) - 1)
        ]

        # Find the breakpoint threshold
        threshold = _percentile_threshold(similarities, 100 - self.breakpoint_percentile)

        # Group sentences into raw chunks at breakpoints
        raw_chunks: list[list[str]] = []
        current: list[str] = [sentences[0]]

        for i, sim in enumerate(similarities):
            next_sentence = sentences[i + 1]
            if sim < threshold:
                # Breakpoint detected — start a new chunk
                # Optionally carry overlap sentences forward
                overlap = current[-self.overlap_sentences:] if self.overlap_sentences else []
                raw_chunks.append(current)
                current = overlap + [next_sentence]
            else:
                current.append(next_sentence)

        if current:
            raw_chunks.append(current)

        # Join and apply size constraints
        joined = [" ".join(chunk) for chunk in raw_chunks]
        return self._split_by_size(self._merge_short(joined))

    def _merge_short(self, chunks: list[str]) -> list[str]:
        """Merge chunks shorter than ``min_chunk_size`` into their successor."""
        if len(chunks) <= 1:
            return chunks
        merged: list[str] = []
        i = 0
        while i < len(chunks):
            chunk = chunks[i]
            if len(chunk) < self.min_chunk_size and i + 1 < len(chunks):
                chunks[i + 1] = chunk + " " + chunks[i + 1]
            else:
                merged.append(chunk)
            i += 1
        return merged if merged else chunks

    def _split_by_size(self, chunks: list[str]) -> list[str]:
        """Split any chunk that exceeds ``max_chunk_size`` at sentence boundaries."""
        result: list[str] = []
        for chunk in chunks:
            if len(chunk) <= self.max_chunk_size:
                result.append(chunk)
                continue
            sentences = split_sentences(chunk)
            current: list[str] = []
            current_len = 0
            for sent in sentences:
                if current_len + len(sent) > self.max_chunk_size and current:
                    result.append(" ".join(current))
                    current = [sent]
                    current_len = len(sent)
                else:
                    current.append(sent)
                    current_len += len(sent)
            if current:
                result.append(" ".join(current))
        return result or chunks
