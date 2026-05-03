"""
RAG3 Vector Search Tool
=========================
Exposes the full vector retrieval sub-pipeline (query expansion →
vector/hybrid search → reranker → self-reflection) as a single
synchronous function wrapped as a Haystack 2.x ``Tool`` suitable for an
Agent's tool list.
"""

from __future__ import annotations

import asyncio
from typing import Any

from haystack.dataclasses import Document

from src.config import get_settings
from src.ingestion.embedder import CachedOllamaEmbedder
from src.monitoring.logger import get_logger
from src.retrieval.cache import TTLCache, hash_key
from src.retrieval.strategies.query_expansion import QueryExpander
from src.retrieval.strategies.reranking import OllamaRanker
from src.retrieval.strategies.self_reflection import SelfReflection
from src.storage.base import BaseVectorStore, SearchMode

log = get_logger(__name__)


class VectorSearchTool:
    """
    Callable encapsulating the end-to-end vector retrieval strategy.

    The public ``__call__(query) -> list[Document]`` makes it trivial to
    register as a Haystack Agent tool without importing Haystack types
    here.
    """

    def __init__(
        self,
        vector_store: BaseVectorStore,
        embedder: CachedOllamaEmbedder | None = None,
        expander: QueryExpander | None = None,
        reranker: OllamaRanker | None = None,
        reflector: SelfReflection | None = None,
        cache: TTLCache | None = None,
        mode: SearchMode = SearchMode.HYBRID,
    ) -> None:
        cfg = get_settings().retrieval
        self._store = vector_store
        self._embedder = embedder or CachedOllamaEmbedder()
        self._expander = expander or QueryExpander()
        self._reranker = reranker or OllamaRanker()
        self._reflector = reflector or SelfReflection()
        self._cache: TTLCache = cache or TTLCache()
        self._mode = mode
        self._top_k_vector = cfg.top_k_vector
        self._top_k_final = cfg.top_k_final

    # ------------------------------------------------------------------

    def __call__(self, query: str) -> list[Document]:
        return asyncio.run(self.arun(query))

    async def arun(self, query: str) -> list[Document]:
        cache_key = hash_key("vector_tool", query, self._mode.value)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        variants: list[str] = self._expander.run(query=query)["queries"]
        seen_ids: dict[str, Document] = {}
        for q in variants:
            qvec = self._embedder.run(texts=[q])["embeddings"][0]
            results = await self._store.search(
                query_embedding=qvec,
                query_text=q,
                top_k=self._top_k_vector,
                mode=self._mode,
            )
            for r in results:
                seen_ids.setdefault(r.document.id, r.document)

        candidates = list(seen_ids.values())
        if not candidates:
            return []

        reranked = self._reranker.run(
            query=query, documents=candidates, top_k=self._top_k_final * 2
        )["documents"]

        report = self._reflector.run(query=query, documents=reranked)
        accepted = bool(report["accepted"])
        final = reranked[: self._top_k_final]
        if not accepted and len(reranked) > self._top_k_final:
            # on rejection, widen a touch to give the LLM more to work with
            final = reranked[: self._top_k_final * 2]

        self._cache.set(cache_key, final)
        return final
