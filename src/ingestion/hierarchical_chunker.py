"""
RAG Hierarchical Chunker
==========================
Creates a two-level parent → child chunk hierarchy for multi-granularity
retrieval.

Architecture
------------
* **Parent chunks** (large, ~2000 chars): Capture broad context.
  Used for answer generation — fed to the LLM as full context.
* **Child chunks** (small, ~400 chars, 80-char overlap): High-precision
  retrieval units embedded and stored in the vector index.
  Each carries a ``parent_id`` metadata key linking back to its parent.

Retrieval pattern
-----------------
1. Embed and index only child chunks.
2. At query time, retrieve the top-K child chunks.
3. Look up each child's ``parent_id`` and fetch the full parent chunk.
4. Pass parent chunks to the LLM — they contain richer context.

This gives high retrieval precision (small chunks) with high LLM quality
(large context windows).

Haystack 2.x contract
----------------------
``run(elements: list[ParsedElement]) ->
    {"parent_documents": list[Document], "child_documents": list[Document]}``

Usage::

    from src.ingestion.hierarchical_chunker import HierarchicalChunker

    chunker = HierarchicalChunker()
    result = chunker.run(elements=parsed_elements)
    parents = result["parent_documents"]
    children = result["child_documents"]
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass
from typing import Any

from haystack import Document, component

from src.config import get_settings
from src.ingestion.parser import ParsedElement
from src.ingestion.semantic_chunker import split_sentences
from src.monitoring.logger import get_logger, timed_operation
from src.monitoring.metrics import MetricsCollector

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Text splitter
# ---------------------------------------------------------------------------


def _character_split(
    text: str,
    chunk_size: int,
    overlap: int,
) -> list[str]:
    """
    Split text into overlapping fixed-size character windows.

    Splits are made at whitespace boundaries to avoid mid-word cuts.
    """
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end >= len(text):
            chunks.append(text[start:].strip())
            break
        # Walk back to the nearest whitespace
        while end > start and not text[end].isspace():
            end -= 1
        if end == start:
            end = start + chunk_size  # no whitespace found — hard cut
        chunks.append(text[start:end].strip())
        start = end - overlap
    return [c for c in chunks if c]


# ---------------------------------------------------------------------------
# Haystack component
# ---------------------------------------------------------------------------


@component
class HierarchicalChunker:
    """
    Two-level parent/child document chunker.

    Args:
        parent_chunk_size: Target character count for parent chunks.
        child_chunk_size:  Target character count for child chunks.
        child_chunk_overlap: Character overlap between consecutive children.
        doc_id_prefix:     Prefix for generated document IDs.
        include_parents_in_output: Whether to include parent documents in
                                   ``parent_documents`` output (always True —
                                   here as an explicit flag for clarity).
    """

    def __init__(
        self,
        parent_chunk_size: int | None = None,
        child_chunk_size: int | None = None,
        child_chunk_overlap: int | None = None,
        doc_id_prefix: str = "hier",
    ) -> None:
        cfg = get_settings().chunking
        self.parent_chunk_size = parent_chunk_size or cfg.parent_chunk_size
        self.child_chunk_size = child_chunk_size or cfg.child_chunk_size
        self.child_chunk_overlap = child_chunk_overlap or cfg.child_chunk_overlap
        self.doc_id_prefix = doc_id_prefix
        self._metrics = MetricsCollector.get_instance()

        log.info(
            "HierarchicalChunker initialised",
            extra={
                "parent_chunk_size": self.parent_chunk_size,
                "child_chunk_size": self.child_chunk_size,
                "child_overlap": self.child_chunk_overlap,
            },
        )

    # ------------------------------------------------------------------
    # Haystack run()
    # ------------------------------------------------------------------

    @component.output_types(parent_documents=list, child_documents=list)
    def run(
        self,
        elements: list[ParsedElement],
        source_metadata: dict[str, Any] | None = None,
    ) -> dict[str, list[Document]]:
        """
        Build the parent/child chunk hierarchy.

        Args:
            elements:        Output of ``DocumentParser.run()``.
            source_metadata: Extra metadata merged into every document.

        Returns:
            Dict with keys:
            * ``"parent_documents"`` — large context chunks (for LLM).
            * ``"child_documents"`` — small retrieval units with parent links.
        """
        if not elements:
            return {"parent_documents": [], "child_documents": []}

        extra_meta: dict[str, Any] = source_metadata or {}
        parent_docs: list[Document] = []
        child_docs: list[Document] = []

        with timed_operation(
            "hierarchical_chunker.run", log,
            extra={"n_elements": len(elements)},
        ):
            # Separate text from tables/images
            text_elements = [
                el for el in elements
                if el.element_type in ("text", "title") and el.text.strip()
            ]
            special_elements = [
                el for el in elements
                if el.element_type in ("table", "image")
            ]

            # Build parent chunks from concatenated text
            full_text = "\n\n".join(el.text for el in text_elements)
            parent_texts = _character_split(
                full_text,
                chunk_size=self.parent_chunk_size,
                overlap=0,  # parents don't overlap
            )

            for p_idx, parent_text in enumerate(parent_texts):
                parent_id = self._make_id("par", p_idx, parent_text)
                parent_doc = Document(
                    id=parent_id,
                    content=parent_text,
                    meta={
                        "chunk_level": "parent",
                        "parent_index": p_idx,
                        "child_ids": [],  # populated below
                        "chunker": "hierarchical",
                        **extra_meta,
                    },
                )
                parent_docs.append(parent_doc)

                # Split each parent into children
                children = _character_split(
                    parent_text,
                    chunk_size=self.child_chunk_size,
                    overlap=self.child_chunk_overlap,
                )
                child_ids: list[str] = []
                for c_idx, child_text in enumerate(children):
                    child_id = self._make_id("chd", f"{p_idx}_{c_idx}", child_text)
                    child_doc = Document(
                        id=child_id,
                        content=child_text,
                        meta={
                            "chunk_level": "child",
                            "parent_id": parent_id,
                            "parent_index": p_idx,
                            "child_index": c_idx,
                            "chunker": "hierarchical",
                            **extra_meta,
                        },
                    )
                    child_docs.append(child_doc)
                    child_ids.append(child_id)

                # Back-fill child_ids into parent metadata
                parent_doc.meta["child_ids"] = child_ids

            # Pass-through: tables and images become individual parent+child pairs
            for s_idx, el in enumerate(special_elements):
                special_id = self._make_id("spc", s_idx, el.text)
                page = el.page_number

                parent_doc = Document(
                    id=f"{special_id}_par",
                    content=el.text,
                    meta={
                        "chunk_level": "parent",
                        "element_type": el.element_type,
                        "page_number": page,
                        "child_ids": [f"{special_id}_chd"],
                        "chunker": "hierarchical",
                        **extra_meta,
                        **el.metadata,
                    },
                )
                child_doc = Document(
                    id=f"{special_id}_chd",
                    content=el.text,
                    meta={
                        "chunk_level": "child",
                        "parent_id": f"{special_id}_par",
                        "element_type": el.element_type,
                        "page_number": page,
                        "chunker": "hierarchical",
                        **extra_meta,
                        **el.metadata,
                    },
                )
                parent_docs.append(parent_doc)
                child_docs.append(child_doc)

        self._metrics.record_event("hierarchical_chunker.parent_chunks", len(parent_docs))
        self._metrics.record_event("hierarchical_chunker.child_chunks", len(child_docs))

        log.info(
            "HierarchicalChunker complete",
            extra={
                "parent_chunks": len(parent_docs),
                "child_chunks": len(child_docs),
            },
        )
        return {"parent_documents": parent_docs, "child_documents": child_docs}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_id(self, level: str, index: Any, text: str) -> str:
        """Deterministic document ID based on content hash."""
        content_hash = hashlib.sha256(text[:256].encode()).hexdigest()[:12]
        return f"{self.doc_id_prefix}_{level}_{index}_{content_hash}"
