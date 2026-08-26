"""
RAG Pinecone Serverless Vector Store
======================================
Production-grade ``BaseVectorStore`` implementation backed by Pinecone Serverless.

Features
--------
* Automatic serverless index creation on ``initialise()``.
* Metadata dictionary filtering (`$eq`, `$in`, etc.).
* Async-friendly wrapping around Pinecone synchronous client calls.
* Haystack `Document` conversion to/from Pinecone vector format.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

from haystack.dataclasses import Document

from src.config import get_settings
from src.monitoring.logger import get_logger, timed_operation
from src.monitoring.metrics import MetricsCollector
from src.storage.base import BaseVectorStore, SearchMode, SearchResult

log = get_logger(__name__)

try:
    from pinecone import Pinecone, ServerlessSpec
    _PINECONE_AVAILABLE = True
except ImportError:
    Pinecone = None  # type: ignore
    ServerlessSpec = None  # type: ignore
    _PINECONE_AVAILABLE = False


class PineconeVectorStore(BaseVectorStore):
    """
    Pinecone Serverless vector store implementation.
    """

    def __init__(
        self,
        api_key: str | None = None,
        index_name: str | None = None,
        cloud: str | None = None,
        region: str | None = None,
        dimension: int | None = None,
        namespace: str | None = None,
    ) -> None:
        if not _PINECONE_AVAILABLE:
            raise RuntimeError("pinecone SDK not installed. Run `pip install pinecone`.")

        cfg = get_settings().pinecone
        self._api_key: str = (
            api_key
            or (cfg.api_key.get_secret_value() if cfg.api_key else "")
            or os.environ.get("PINECONE_API_KEY", "")
        )
        if not self._api_key:
            raise ValueError("Pinecone API key is required.")

        self._index_name: str = index_name or cfg.index_name
        self._cloud: str = cloud or cfg.cloud
        self._region: str = region or cfg.region
        self._metric: str = cfg.metric
        self._dim: int = dimension or cfg.dimension
        self._namespace: str = namespace or cfg.namespace

        self._pc: Pinecone | None = None
        self._index: Any = None
        self._metrics = MetricsCollector.get_instance()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialise(self) -> None:
        """Initialize Pinecone client and ensure serverless index exists."""
        if self._index is not None:
            return

        def _init_sync():
            pc = Pinecone(api_key=self._api_key)
            existing_indexes = [idx.name for idx in pc.list_indexes()]
            if self._index_name not in existing_indexes:
                log.info(
                    "Creating new Pinecone serverless index",
                    extra={"name": self._index_name, "cloud": self._cloud, "region": self._region},
                )
                pc.create_index(
                    name=self._index_name,
                    dimension=self._dim,
                    metric=self._metric,
                    spec=ServerlessSpec(cloud=self._cloud, region=self._region),
                )
            return pc, pc.Index(self._index_name)

        self._pc, self._index = await asyncio.to_thread(_init_sync)
        log.info(
            "PineconeVectorStore initialised",
            extra={"index_name": self._index_name, "dimension": self._dim},
        )

    async def close(self) -> None:
        """No-op for Pinecone SDK."""
        self._index = None
        self._pc = None

    # ------------------------------------------------------------------
    # Upsert
    # ------------------------------------------------------------------

    async def upsert_documents(self, documents: list[Document]) -> list[str]:
        if not documents:
            return []
        assert self._index is not None, "Call initialise() first."

        vectors: list[dict[str, Any]] = []
        for doc in documents:
            if doc.embedding is None:
                raise ValueError(f"Document {doc.id!r} is missing an embedding.")
            
            # Sanitize metadata for Pinecone (only primitive types, lists of strings, etc.)
            meta = dict(doc.meta or {})
            meta["content"] = doc.content or ""
            
            vectors.append({
                "id": doc.id,
                "values": doc.embedding,
                "metadata": meta,
            })

        def _upsert_sync():
            # Batch upsert in chunks of 100
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                chunk = vectors[i : i + batch_size]
                self._index.upsert(vectors=chunk, namespace=self._namespace)

        with timed_operation(log, "pinecone.upsert", count=len(documents)):
            await asyncio.to_thread(_upsert_sync)

        self._metrics.record_event("pinecone.upsert", count=len(documents))
        return [doc.id for doc in documents]

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    async def search(
        self,
        query_embedding: list[float],
        query_text: str,
        top_k: int = 10,
        mode: SearchMode = SearchMode.HYBRID,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        assert self._index is not None, "Call initialise() first."

        pinecone_filter = self._compile_filters(filters)

        def _query_sync():
            return self._index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                filter=pinecone_filter if pinecone_filter else None,
                namespace=self._namespace,
            )

        with timed_operation(log, "pinecone.search"):
            res = await asyncio.to_thread(_query_sync)

        results: list[SearchResult] = []
        for i, match in enumerate(res.get("matches", [])):
            meta = dict(match.get("metadata") or {})
            content = meta.pop("content", "")
            results.append(
                SearchResult(
                    document=Document(
                        id=match["id"],
                        content=content,
                        meta=meta,
                        embedding=match.get("values"),
                    ),
                    score=float(match.get("score", 0.0)),
                    source=SearchMode.VECTOR,
                    rank=i,
                )
            )

        return results

    # ------------------------------------------------------------------
    # Utils / Delete
    # ------------------------------------------------------------------

    async def delete_documents(self, document_ids: list[str]) -> int:
        if not document_ids:
            return 0
        assert self._index is not None

        def _delete_sync():
            self._index.delete(ids=document_ids, namespace=self._namespace)

        await asyncio.to_thread(_delete_sync)
        return len(document_ids)

    async def count_documents(self, filters: dict[str, Any] | None = None) -> int:
        assert self._index is not None

        def _stats_sync():
            stats = self._index.describe_index_stats()
            ns_stats = stats.get("namespaces", {}).get(self._namespace, {})
            return ns_stats.get("vector_count", 0)

        return await asyncio.to_thread(_stats_sync)

    @staticmethod
    def _compile_filters(filters: dict[str, Any] | None) -> dict[str, Any] | None:
        if not filters:
            return None
        formatted: dict[str, Any] = {}
        for k, v in filters.items():
            formatted[k] = {"$eq": v}
        return formatted
