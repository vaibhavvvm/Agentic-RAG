"""
RAG3 Summary Index Strategy
=============================
Companion component that produces document-level summaries at ingest
time (``SummaryGenerator``) and queries the ``PgSummaryStore`` at
retrieval time (``SummaryRetriever``).
"""

from __future__ import annotations

import asyncio
from typing import ClassVar

from haystack import component
from haystack.dataclasses import Document

from src.ingestion.embedder import CachedOllamaEmbedder
from src.monitoring.logger import get_logger
from src.storage.base import BaseSummaryStore, SummaryEntry
from src.utils.llm import chat_sync

log = get_logger(__name__)

SUMMARY_PROMPT = (
    "You write a concise 3–5 sentence summary of a document. Preserve named "
    "entities, key numbers, and decisions. No preamble, just the summary."
)


@component
class SummaryGenerator:
    """
    Runs over parent chunks or full docs, LLM-summarises each, and upserts
    them into a ``BaseSummaryStore`` along with their embeddings.

    Typically invoked *outside* the main Haystack pipeline, during a
    post-ingest pass that writes parent-level summaries.
    """

    OUTPUT_TYPES: ClassVar[dict[str, type]] = {"summaries": list}

    def __init__(
        self,
        store: BaseSummaryStore,
        embedder: CachedOllamaEmbedder | None = None,
        summary_type: str = "full",
    ) -> None:
        self._store = store
        self._embedder = embedder or CachedOllamaEmbedder()
        self._summary_type = summary_type

    @component.output_types(summaries=list)
    def run(self, documents: list[Document]) -> dict[str, list[SummaryEntry]]:
        results: list[SummaryEntry] = []
        for doc in documents:
            body = (doc.content or "")[:6000]
            if not body.strip():
                continue
            try:
                summary = chat_sync(
                    SUMMARY_PROMPT,
                    body,
                    fast=True,
                    temperature=0.1,
                    max_tokens=300,
                )
            except Exception as exc:
                log.warning(
                    "Summary generation failed", extra={"doc_id": doc.id, "error": str(exc)}
                )
                continue

            vec = self._embedder.run(texts=[summary])["embeddings"][0]
            entry = SummaryEntry(
                doc_id=doc.id,
                summary=summary,
                summary_type=self._summary_type,
                embedding=vec,
                metadata=dict(doc.meta or {}),
            )
            asyncio.run(self._store.upsert_summary(entry))
            results.append(entry)
        return {"summaries": results}


@component
class SummaryRetriever:
    """
    Query-time component: embeds query, hits summary store, returns
    top-k summary entries as ``Document`` objects.
    """

    OUTPUT_TYPES: ClassVar[dict[str, type]] = {"documents": list}

    def __init__(
        self,
        store: BaseSummaryStore,
        embedder: CachedOllamaEmbedder | None = None,
        top_k: int = 3,
    ) -> None:
        self._store = store
        self._embedder = embedder or CachedOllamaEmbedder()
        self._top_k = top_k

    @component.output_types(documents=list)
    def run(
        self, query: str, top_k: int | None = None
    ) -> dict[str, list[Document]]:
        vec = self._embedder.run(texts=[query])["embeddings"][0]
        entries = asyncio.run(
            self._store.search_summaries(vec, top_k=top_k or self._top_k)
        )
        docs = [
            Document(
                id=f"summary::{e.doc_id}::{e.summary_type}",
                content=e.summary,
                meta={**e.metadata, "source_doc_id": e.doc_id, "is_summary": True},
            )
            for e in entries
        ]
        return {"documents": docs}
