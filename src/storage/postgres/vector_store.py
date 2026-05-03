"""
RAG3 PgVector Hybrid Store
============================
Production-grade ``BaseVectorStore`` implementation backed by PostgreSQL
with the ``pgvector`` extension and native full-text search (tsvector).

Features
--------
* **HNSW index** on the embedding column for sub-millisecond ANN.
* **tsvector + GIN** index for BM25-style keyword search.
* **Reciprocal Rank Fusion (RRF)** for hybrid retrieval — computed in
  Python on the union of the top-K vector + top-K BM25 results.
* **asyncpg connection pool** sized by ``postgres.pool_min/max_size``.
* **Metadata filter compilation** — flat dict → SQL ``WHERE`` clauses.
* **Idempotent schema bootstrap** — safe to call ``initialise()`` every
  startup; uses ``CREATE TABLE IF NOT EXISTS`` + ``CREATE INDEX IF NOT
  EXISTS``.

Schema
------
``documents(
    id           TEXT PRIMARY KEY,
    content      TEXT NOT NULL,
    metadata     JSONB NOT NULL DEFAULT '{}'::jsonb,
    embedding    VECTOR(<dim>) NOT NULL,
    tsv          TSVECTOR,
    created_at   TIMESTAMPTZ DEFAULT NOW()
)``

Indexes:
    * ``documents_embedding_hnsw`` — HNSW, ``vector_cosine_ops``
    * ``documents_tsv_gin``        — GIN on ``tsv``
    * ``documents_metadata_gin``   — GIN on ``metadata``
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import asyncpg
from haystack.dataclasses import Document

from src.config import get_settings
from src.monitoring.logger import get_logger, timed_operation
from src.monitoring.metrics import MetricsCollector
from src.storage.base import BaseVectorStore, SearchMode, SearchResult

log = get_logger(__name__)


def _vec_literal(vec: list[float]) -> str:
    """Format a Python list as a pgvector literal: '[0.1,0.2,...]'."""
    return "[" + ",".join(f"{float(v):.8f}" for v in vec) + "]"


class PgVectorStore(BaseVectorStore):
    """
    pgvector-backed hybrid search store.

    Args:
        dsn:              Optional DSN override; otherwise taken from settings.
        table_name:       Documents table (default ``documents``).
        vector_dim:       Override embedding dimension (must match inserts).
        rrf_k:            Smoothing constant for Reciprocal Rank Fusion.
    """

    def __init__(
        self,
        dsn: str | None = None,
        table_name: str = "documents",
        vector_dim: int | None = None,
        rrf_k: int | None = None,
    ) -> None:
        cfg = get_settings()
        self._dsn: str = dsn or self._asyncpg_dsn(cfg.postgres.sync_dsn)
        self._table: str = table_name
        self._dim: int = vector_dim or cfg.postgres.vector_dim
        self._rrf_k: int = rrf_k or cfg.retrieval.rrf_k
        self._bm25_weight: float = cfg.postgres.bm25_weight
        self._vector_weight: float = cfg.postgres.vector_weight
        self._hnsw_m: int = cfg.postgres.hnsw_m
        self._hnsw_ef_construction: int = cfg.postgres.hnsw_ef_construction
        self._hnsw_ef_search: int = cfg.postgres.hnsw_ef_search
        self._pool_min: int = cfg.postgres.pool_min_size
        self._pool_max: int = cfg.postgres.pool_max_size

        self._pool: asyncpg.Pool | None = None
        self._metrics = MetricsCollector.get_instance()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @staticmethod
    def _asyncpg_dsn(sync_dsn: str) -> str:
        """Convert psycopg/sqlalchemy DSN to a plain asyncpg-compatible one."""
        return sync_dsn.replace("postgresql+asyncpg://", "postgresql://")

    async def initialise(self) -> None:
        """Create pool, enable pgvector, and ensure schema + indices."""
        if self._pool is not None:
            return

        self._pool = await asyncpg.create_pool(
            dsn=self._dsn,
            min_size=self._pool_min,
            max_size=self._pool_max,
        )
        async with self._pool.acquire() as conn:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self._table} (
                    id         TEXT PRIMARY KEY,
                    content    TEXT NOT NULL,
                    metadata   JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    embedding  VECTOR({self._dim}) NOT NULL,
                    tsv        TSVECTOR,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
            """)
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS {self._table}_embedding_hnsw
                ON {self._table} USING hnsw (embedding vector_cosine_ops)
                WITH (m = {self._hnsw_m}, ef_construction = {self._hnsw_ef_construction});
            """)
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS {self._table}_tsv_gin
                ON {self._table} USING GIN (tsv);
            """)
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS {self._table}_metadata_gin
                ON {self._table} USING GIN (metadata);
            """)

        log.info(
            "PgVectorStore initialised",
            extra={"table": self._table, "vector_dim": self._dim},
        )

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    # ------------------------------------------------------------------
    # Upsert
    # ------------------------------------------------------------------

    async def upsert_documents(self, documents: list[Document]) -> list[str]:
        if not documents:
            return []
        assert self._pool is not None, "Call initialise() first."

        rows: list[tuple[str, str, str, str]] = []
        for doc in documents:
            if doc.embedding is None:
                raise ValueError(f"Document {doc.id!r} is missing an embedding.")
            if len(doc.embedding) != self._dim:
                raise ValueError(
                    f"Document {doc.id!r} embedding dim {len(doc.embedding)} "
                    f"!= configured {self._dim}"
                )
            rows.append((
                doc.id,
                doc.content or "",
                json.dumps(doc.meta or {}),
                _vec_literal(doc.embedding),
            ))

        sql = f"""
            INSERT INTO {self._table} (id, content, metadata, embedding, tsv)
            VALUES ($1, $2, $3::jsonb, $4::vector, to_tsvector('english', $2))
            ON CONFLICT (id) DO UPDATE SET
                content   = EXCLUDED.content,
                metadata  = EXCLUDED.metadata,
                embedding = EXCLUDED.embedding,
                tsv       = EXCLUDED.tsv;
        """

        with timed_operation(log, "pgvector.upsert", count=len(rows)):
            async with self._pool.acquire() as conn:
                async with conn.transaction():
                    await conn.executemany(sql, rows)

        self._metrics.record_event("pgvector.upsert", value=len(rows))
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
        assert self._pool is not None, "Call initialise() first."

        if mode == SearchMode.VECTOR:
            return await self._vector_search(query_embedding, top_k, filters)
        if mode == SearchMode.BM25:
            return await self._bm25_search(query_text, top_k, filters)

        # HYBRID: run both, fuse with RRF
        vec_task = self._vector_search(query_embedding, top_k, filters)
        bm25_task = self._bm25_search(query_text, top_k, filters)
        vec_hits, bm25_hits = await asyncio.gather(vec_task, bm25_task)
        return self._rrf_fuse(vec_hits, bm25_hits, top_k)

    async def _vector_search(
        self,
        query_embedding: list[float],
        top_k: int,
        filters: dict[str, Any] | None,
    ) -> list[SearchResult]:
        where_sql, where_params = self._compile_filters(filters, start_index=2)
        sql = f"""
            SET LOCAL hnsw.ef_search = {self._hnsw_ef_search};
            SELECT id, content, metadata,
                   1 - (embedding <=> $1::vector) AS score
            FROM {self._table}
            {where_sql}
            ORDER BY embedding <=> $1::vector
            LIMIT {int(top_k)};
        """
        params: list[Any] = [_vec_literal(query_embedding), *where_params]

        with timed_operation(log, "pgvector.search.vector"):
            async with self._pool.acquire() as conn:
                async with conn.transaction():
                    rows = await conn.fetch(sql, *params)

        return [
            SearchResult(
                document=Document(
                    id=r["id"],
                    content=r["content"],
                    meta=json.loads(r["metadata"]) if isinstance(r["metadata"], str) else (r["metadata"] or {}),
                ),
                score=float(r["score"]),
                source=SearchMode.VECTOR,
                rank=i,
            )
            for i, r in enumerate(rows)
        ]

    async def _bm25_search(
        self,
        query_text: str,
        top_k: int,
        filters: dict[str, Any] | None,
    ) -> list[SearchResult]:
        where_sql, where_params = self._compile_filters(filters, start_index=2)
        sql = f"""
            SELECT id, content, metadata,
                   ts_rank_cd(tsv, plainto_tsquery('english', $1)) AS score
            FROM {self._table}
            WHERE tsv @@ plainto_tsquery('english', $1)
            {where_sql.replace("WHERE", "AND", 1) if where_sql else ""}
            ORDER BY score DESC
            LIMIT {int(top_k)};
        """
        params: list[Any] = [query_text, *where_params]

        with timed_operation(log, "pgvector.search.bm25"):
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(sql, *params)

        return [
            SearchResult(
                document=Document(
                    id=r["id"],
                    content=r["content"],
                    meta=json.loads(r["metadata"]) if isinstance(r["metadata"], str) else (r["metadata"] or {}),
                ),
                score=float(r["score"] or 0.0),
                source=SearchMode.BM25,
                rank=i,
            )
            for i, r in enumerate(rows)
        ]

    def _rrf_fuse(
        self,
        vec_hits: list[SearchResult],
        bm25_hits: list[SearchResult],
        top_k: int,
    ) -> list[SearchResult]:
        """
        Reciprocal Rank Fusion.

        Final score per document = sum over retrieval sources of
        ``weight_s / (k + rank_s)``.
        """
        k = self._rrf_k
        scores: dict[str, float] = {}
        docs: dict[str, Document] = {}

        for hit in vec_hits:
            scores[hit.document.id] = (
                scores.get(hit.document.id, 0.0)
                + self._vector_weight / (k + hit.rank + 1)
            )
            docs[hit.document.id] = hit.document

        for hit in bm25_hits:
            scores[hit.document.id] = (
                scores.get(hit.document.id, 0.0)
                + self._bm25_weight / (k + hit.rank + 1)
            )
            docs.setdefault(hit.document.id, hit.document)

        fused = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        return [
            SearchResult(
                document=docs[doc_id],
                score=score,
                source=SearchMode.HYBRID,
                rank=i,
            )
            for i, (doc_id, score) in enumerate(fused)
        ]

    # ------------------------------------------------------------------
    # Filters / utils
    # ------------------------------------------------------------------

    @staticmethod
    def _compile_filters(
        filters: dict[str, Any] | None,
        start_index: int,
    ) -> tuple[str, list[Any]]:
        """Compile a flat dict into ``WHERE metadata->>'k' = $n`` clauses."""
        if not filters:
            return "", []
        parts: list[str] = []
        params: list[Any] = []
        i = start_index
        for key, value in filters.items():
            parts.append(f"metadata->>'{key}' = ${i}")
            params.append(str(value))
            i += 1
        return "WHERE " + " AND ".join(parts), params

    async def delete_documents(self, document_ids: list[str]) -> int:
        if not document_ids:
            return 0
        assert self._pool is not None
        sql = f"DELETE FROM {self._table} WHERE id = ANY($1::text[]);"
        async with self._pool.acquire() as conn:
            result = await conn.execute(sql, document_ids)
        # result looks like "DELETE N"
        try:
            return int(result.split()[-1])
        except (IndexError, ValueError):
            return 0

    async def count_documents(self, filters: dict[str, Any] | None = None) -> int:
        assert self._pool is not None
        where_sql, params = self._compile_filters(filters, start_index=1)
        sql = f"SELECT COUNT(*) FROM {self._table} {where_sql};"
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(sql, *params)
        return int(row[0]) if row else 0
