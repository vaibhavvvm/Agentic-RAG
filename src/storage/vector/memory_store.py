"""
RAG3 In-Memory Vector Store
=============================
Numpy-backed fallback that implements ``BaseVectorStore`` for cases
where pgvector is unreachable. Designed for developer laptops, CI, and
zero-dependency smoke tests — it is **not** persistent.

Hybrid search is approximated as:

    score = vector_weight * cosine + bm25_weight * token_overlap

No recall-precision optimisation; the idea is to keep the rest of the
RAG stack working end-to-end while the primary store is down.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from haystack.dataclasses import Document

from src.config import get_settings
from src.monitoring.logger import get_logger
from src.storage.base import BaseVectorStore, SearchMode, SearchResult

log = get_logger(__name__)


class InMemoryVectorStore(BaseVectorStore):
    """Numpy fallback vector store (non-persistent)."""

    def __init__(self) -> None:
        self._docs: dict[str, Document] = {}
        self._matrix: np.ndarray | None = None
        self._ids: list[str] = []
        cfg = get_settings().postgres
        self._v_w = cfg.vector_weight
        self._b_w = cfg.bm25_weight

    async def initialise(self) -> None:
        log.info("InMemoryVectorStore ready (non-persistent)")

    async def close(self) -> None:
        self._docs.clear()
        self._matrix = None
        self._ids.clear()

    async def upsert_documents(self, documents: list[Document]) -> list[str]:
        for d in documents:
            if not d.embedding:
                continue
            self._docs[d.id] = d
        self._rebuild()
        return [d.id for d in documents if d.embedding]

    def _rebuild(self) -> None:
        items = [(i, d) for i, d in self._docs.items() if d.embedding]
        if not items:
            self._matrix = None
            self._ids = []
            return
        self._ids = [i for i, _ in items]
        self._matrix = np.asarray([d.embedding for _, d in items], dtype=np.float32)
        norms = np.linalg.norm(self._matrix, axis=1, keepdims=True) + 1e-12
        self._matrix = self._matrix / norms  # store pre-normalised

    async def search(
        self,
        query_embedding: list[float],
        query_text: str,
        top_k: int = 10,
        mode: SearchMode = SearchMode.HYBRID,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        if self._matrix is None or not self._ids:
            return []

        q = np.asarray(query_embedding, dtype=np.float32)
        q = q / (np.linalg.norm(q) + 1e-12)
        cos = (self._matrix @ q).tolist()

        q_tokens = {t.lower() for t in query_text.split() if len(t) > 2}

        def overlap(doc: Document) -> float:
            d_tokens = {t.lower() for t in (doc.content or "").split() if len(t) > 2}
            if not q_tokens or not d_tokens:
                return 0.0
            return len(q_tokens & d_tokens) / math.sqrt(len(q_tokens) * len(d_tokens))

        entries: list[tuple[str, float, float]] = []
        for idx, doc_id in enumerate(self._ids):
            doc = self._docs[doc_id]
            if filters and not self._matches(doc, filters):
                continue
            c = float(cos[idx])
            b = overlap(doc) if mode != SearchMode.VECTOR else 0.0
            if mode == SearchMode.VECTOR:
                score = c
            elif mode == SearchMode.BM25:
                score = b
            else:
                score = self._v_w * c + self._b_w * b
            entries.append((doc_id, score, c))

        entries.sort(key=lambda x: x[1], reverse=True)
        top = entries[:top_k]
        return [
            SearchResult(
                document=self._docs[doc_id],
                score=score,
                source=mode,
                rank=rank,
            )
            for rank, (doc_id, score, _cos) in enumerate(top)
        ]

    @staticmethod
    def _matches(doc: Document, filters: dict[str, Any]) -> bool:
        meta = doc.meta or {}
        return all(meta.get(k) == v for k, v in filters.items())

    async def delete_documents(self, document_ids: list[str]) -> int:
        n = 0
        for did in document_ids:
            if did in self._docs:
                del self._docs[did]
                n += 1
        self._rebuild()
        return n

    async def count_documents(self, filters: dict[str, Any] | None = None) -> int:
        if not filters:
            return len(self._docs)
        return sum(1 for d in self._docs.values() if self._matches(d, filters))
