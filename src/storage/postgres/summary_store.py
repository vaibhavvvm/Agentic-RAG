"""
RAG3 Postgres Summary Store
============================
``BaseSummaryStore`` implementation for document-level summaries
(full / topic / section granularities).

The summary index keeps one compact row per summary, embedded with the
same model as the main vector store so a single query vector can search
both.  Used by the retrieval pipeline's summary-first fallback: when
chunk-level search returns low-confidence hits, we broaden to summary
search instead.

Schema
------
``summaries(
    id            SERIAL PRIMARY KEY,
    doc_id        TEXT NOT NULL,
    summary_type  TEXT NOT NULL,           -- 'full' | 'topic' | 'section'
    summary       TEXT NOT NULL,
    embedding     VECTOR(<dim>),
    metadata      JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at    TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(doc_id, summary_type, metadata->>'section_id')
)``
"""

from __future__ import annotations

import json
from typing import Any

import asyncpg

from src.config import get_settings
from src.monitoring.logger import get_logger, timed_operation
from src.storage.base import BaseSummaryStore, SummaryEntry

log = get_logger(__name__)


def _vec_literal(vec: list[float]) -> str:
    return "[" + ",".join(f"{float(v):.8f}" for v in vec) + "]"


class PgSummaryStore(BaseSummaryStore):
    """
    Postgres-backed summary index.

    Args:
        dsn:          Optional DSN override.
        table_name:   Table name (default ``summaries``).
        vector_dim:   Embedding dimension (must match main store).
    """

    def __init__(
        self,
        dsn: str | None = None,
        table_name: str = "summaries",
        vector_dim: int | None = None,
    ) -> None:
        cfg = get_settings()
        self._dsn: str = dsn or cfg.postgres.sync_dsn.replace(
            "postgresql+asyncpg://", "postgresql://"
        )
        self._table: str = table_name
        self._dim: int = vector_dim or cfg.postgres.vector_dim
        self._pool: asyncpg.Pool | None = None

    async def initialise(self) -> None:
        if self._pool is not None:
            return
        self._pool = await asyncpg.create_pool(
            dsn=self._dsn, min_size=1, max_size=5
        )
        async with self._pool.acquire() as conn:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self._table} (
                    id           SERIAL PRIMARY KEY,
                    doc_id       TEXT NOT NULL,
                    summary_type TEXT NOT NULL,
                    summary      TEXT NOT NULL,
                    embedding    VECTOR({self._dim}),
                    metadata     JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    created_at   TIMESTAMPTZ DEFAULT NOW()
                );
            """)
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS {self._table}_doc_id
                ON {self._table} (doc_id);
            """)
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS {self._table}_type
                ON {self._table} (summary_type);
            """)
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS {self._table}_embedding_hnsw
                ON {self._table} USING hnsw (embedding vector_cosine_ops);
            """)

        log.info("PgSummaryStore initialised", extra={"table": self._table})

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def upsert_summary(self, entry: SummaryEntry) -> str:
        assert self._pool is not None
        section_id = (entry.metadata or {}).get("section_id", "")

        sql = f"""
            INSERT INTO {self._table} (doc_id, summary_type, summary, embedding, metadata)
            SELECT $1, $2, $3, $4::vector, $5::jsonb
            WHERE NOT EXISTS (
                SELECT 1 FROM {self._table}
                WHERE doc_id = $1 AND summary_type = $2
                  AND COALESCE(metadata->>'section_id', '') = $6
            )
            RETURNING doc_id;
        """
        embedding_literal = (
            _vec_literal(entry.embedding) if entry.embedding else None
        )
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                sql,
                entry.doc_id,
                entry.summary_type,
                entry.summary,
                embedding_literal,
                json.dumps(entry.metadata or {}),
                str(section_id),
            )
            if row is None:
                # Update existing
                await conn.execute(
                    f"""
                    UPDATE {self._table}
                    SET summary = $3,
                        embedding = $4::vector,
                        metadata = $5::jsonb
                    WHERE doc_id = $1 AND summary_type = $2
                      AND COALESCE(metadata->>'section_id', '') = $6;
                    """,
                    entry.doc_id,
                    entry.summary_type,
                    entry.summary,
                    embedding_literal,
                    json.dumps(entry.metadata or {}),
                    str(section_id),
                )
        return entry.doc_id

    async def search_summaries(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        summary_type: str | None = None,
    ) -> list[SummaryEntry]:
        assert self._pool is not None
        where_clauses = ["embedding IS NOT NULL"]
        params: list[Any] = [_vec_literal(query_embedding)]
        if summary_type:
            where_clauses.append(f"summary_type = $2")
            params.append(summary_type)
        where_sql = " WHERE " + " AND ".join(where_clauses)

        sql = f"""
            SELECT doc_id, summary_type, summary, metadata,
                   1 - (embedding <=> $1::vector) AS score
            FROM {self._table}
            {where_sql}
            ORDER BY embedding <=> $1::vector
            LIMIT {int(top_k)};
        """
        with timed_operation(log, "summary.search"):
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(sql, *params)
        return [
            SummaryEntry(
                doc_id=r["doc_id"],
                summary=r["summary"],
                summary_type=r["summary_type"],
                metadata=(json.loads(r["metadata"]) if isinstance(r["metadata"], str) else r["metadata"]) or {},
            )
            for r in rows
        ]

    async def get_by_doc_id(self, doc_id: str) -> list[SummaryEntry]:
        assert self._pool is not None
        sql = f"""
            SELECT doc_id, summary_type, summary, metadata
            FROM {self._table}
            WHERE doc_id = $1;
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, doc_id)
        return [
            SummaryEntry(
                doc_id=r["doc_id"],
                summary=r["summary"],
                summary_type=r["summary_type"],
                metadata=(json.loads(r["metadata"]) if isinstance(r["metadata"], str) else r["metadata"]) or {},
            )
            for r in rows
        ]

    async def delete_by_doc_id(self, doc_id: str) -> int:
        assert self._pool is not None
        sql = f"DELETE FROM {self._table} WHERE doc_id = $1;"
        async with self._pool.acquire() as conn:
            result = await conn.execute(sql, doc_id)
        try:
            return int(result.split()[-1])
        except (IndexError, ValueError):
            return 0
