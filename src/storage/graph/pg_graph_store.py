"""
RAG3 Postgres-based Graph Store (PG-Graph)
=============================================
A lightweight graph implementation on top of PostgreSQL — used when the
user launches the system with ``--graph-backend pggraph``. It avoids the
operational cost of running Neo4j or FalkorDB alongside the existing
Postgres container.

Schema
------
  pg_graph_nodes(id TEXT PK, label TEXT, name TEXT, props JSONB)
  pg_graph_edges(id BIGSERIAL PK, src TEXT, dst TEXT, rel TEXT, props JSONB)
  pg_graph_episodes(id TEXT PK, content TEXT, meta JSONB)

Search is keyword-based (regex entity extraction + BFS through the edges
table up to ``max_hops``). It is intentionally simple — Neo4j or Falkor
are preferred for richer graph queries.
"""

from __future__ import annotations

import json
import re
from typing import Any

import asyncpg

from src.config import get_settings
from src.monitoring.logger import get_logger
from src.storage.base import (
    BaseGraphStore,
    GraphEdge,
    GraphNode,
    GraphSearchResult,
)

log = get_logger(__name__)

_STOPWORDS = {"the", "a", "an", "and", "or", "but", "of", "in", "on", "to", "is"}


def _extract_keywords(text: str, limit: int = 10) -> list[str]:
    toks = re.findall(r"[A-Za-z][A-Za-z0-9\-_]{2,}", text)
    seen: list[str] = []
    for t in toks:
        k = t.lower()
        if k not in _STOPWORDS and k not in seen:
            seen.append(k)
        if len(seen) >= limit:
            break
    return seen


class PgGraphStore(BaseGraphStore):
    """Postgres-backed adjacency-list graph store."""

    def __init__(self) -> None:
        cfg = get_settings().postgres
        self._dsn = cfg.sync_dsn.replace("postgresql://", "postgres://")
        self._pool: asyncpg.Pool | None = None

    async def initialise(self) -> None:
        self._pool = await asyncpg.create_pool(dsn=self._dsn, min_size=1, max_size=4)
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS pg_graph_nodes (
                    id    TEXT PRIMARY KEY,
                    label TEXT,
                    name  TEXT,
                    props JSONB NOT NULL DEFAULT '{}'::jsonb
                );
                CREATE INDEX IF NOT EXISTS idx_pggraph_nodes_name
                    ON pg_graph_nodes(name);

                CREATE TABLE IF NOT EXISTS pg_graph_edges (
                    id    BIGSERIAL PRIMARY KEY,
                    src   TEXT NOT NULL,
                    dst   TEXT NOT NULL,
                    rel   TEXT NOT NULL,
                    props JSONB NOT NULL DEFAULT '{}'::jsonb
                );
                CREATE INDEX IF NOT EXISTS idx_pggraph_edges_src ON pg_graph_edges(src);
                CREATE INDEX IF NOT EXISTS idx_pggraph_edges_dst ON pg_graph_edges(dst);

                CREATE TABLE IF NOT EXISTS pg_graph_episodes (
                    id      TEXT PRIMARY KEY,
                    content TEXT,
                    meta    JSONB NOT NULL DEFAULT '{}'::jsonb
                );
                """
            )

    async def close(self) -> None:
        if self._pool:
            await self._pool.close()
            self._pool = None

    async def add_episode(
        self,
        content: str,
        episode_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        assert self._pool is not None
        meta = json.dumps(metadata or {})
        keywords = _extract_keywords(content)
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    """INSERT INTO pg_graph_episodes(id, content, meta)
                       VALUES ($1,$2,$3::jsonb)
                       ON CONFLICT (id) DO UPDATE SET
                         content = EXCLUDED.content,
                         meta    = EXCLUDED.meta""",
                    episode_id, content[:4000], meta,
                )
                await conn.execute(
                    """INSERT INTO pg_graph_nodes(id, label, name, props)
                       VALUES ($1,'Episode',$2,$3::jsonb)
                       ON CONFLICT (id) DO NOTHING""",
                    f"ep:{episode_id}", episode_id, meta,
                )
                for kw in keywords:
                    nid = f"ent:{kw}"
                    await conn.execute(
                        """INSERT INTO pg_graph_nodes(id, label, name, props)
                           VALUES ($1,'Entity',$2,'{}'::jsonb)
                           ON CONFLICT (id) DO NOTHING""",
                        nid, kw,
                    )
                    await conn.execute(
                        """INSERT INTO pg_graph_edges(src, dst, rel, props)
                           VALUES ($1,$2,'MENTIONS','{}'::jsonb)""",
                        f"ep:{episode_id}", nid,
                    )

    async def search(
        self, query: str, top_k: int = 5, max_hops: int = 2
    ) -> GraphSearchResult:
        assert self._pool is not None
        keywords = _extract_keywords(query)
        if not keywords:
            return GraphSearchResult(nodes=[], edges=[])

        seed_ids = [f"ent:{k}" for k in keywords]
        visited: set[str] = set(seed_ids)
        frontier: set[str] = set(seed_ids)
        edges_out: list[GraphEdge] = []

        async with self._pool.acquire() as conn:
            for _ in range(max_hops):
                if not frontier:
                    break
                rows = await conn.fetch(
                    """SELECT id, src, dst, rel, props FROM pg_graph_edges
                       WHERE src = ANY($1::text[]) OR dst = ANY($1::text[])""",
                    list(frontier),
                )
                new_frontier: set[str] = set()
                for r in rows:
                    edges_out.append(GraphEdge(
                        edge_id=str(r["id"]),
                        rel_type=r["rel"],
                        source_id=r["src"],
                        target_id=r["dst"],
                        properties=dict(r["props"] or {}),
                    ))
                    for nid in (r["src"], r["dst"]):
                        if nid not in visited:
                            visited.add(nid)
                            new_frontier.add(nid)
                frontier = new_frontier

            node_rows = await conn.fetch(
                "SELECT id, label, name, props FROM pg_graph_nodes WHERE id = ANY($1::text[])",
                list(visited),
            )
            ep_rows = await conn.fetch(
                """SELECT e.id, e.content FROM pg_graph_episodes e
                   WHERE 'ep:' || e.id = ANY($1::text[]) LIMIT $2""",
                list(visited), top_k,
            )

        nodes = [
            GraphNode(
                node_id=r["id"],
                labels=frozenset([r["label"]] if r["label"] else []),
                properties={"name": r["name"], **(dict(r["props"] or {}))},
            )
            for r in node_rows
        ]
        facts = [(r["content"] or "")[:400] for r in ep_rows]
        return GraphSearchResult(
            nodes=nodes[: top_k * 5],
            edges=edges_out[: top_k * 5],
            graphiti_facts=facts,
        )

    async def get_entity_subgraph(
        self, entity_id: str, max_hops: int = 2
    ) -> GraphSearchResult:
        return await self.search(entity_id, top_k=5, max_hops=max_hops)

    async def add_triples(
        self,
        triples: list[dict[str, str]],
        episode_id: str | None = None,
    ) -> int:
        assert self._pool is not None
        if not triples:
            return 0
        n = 0
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                for t in triples:
                    s = (t.get("subject") or "").strip().lower()
                    r = (t.get("relation") or "").strip().lower() or "related"
                    o = (t.get("object") or "").strip().lower()
                    if not s or not o:
                        continue
                    s_id, o_id = f"ent:{s}", f"ent:{o}"
                    await conn.execute(
                        """INSERT INTO pg_graph_nodes(id,label,name,props)
                           VALUES ($1,'Entity',$2,'{}'::jsonb)
                           ON CONFLICT (id) DO NOTHING""",
                        s_id, s,
                    )
                    await conn.execute(
                        """INSERT INTO pg_graph_nodes(id,label,name,props)
                           VALUES ($1,'Entity',$2,'{}'::jsonb)
                           ON CONFLICT (id) DO NOTHING""",
                        o_id, o,
                    )
                    await conn.execute(
                        """INSERT INTO pg_graph_edges(src,dst,rel,props)
                           VALUES ($1,$2,$3,'{}'::jsonb)""",
                        s_id, o_id, r,
                    )
                    if episode_id:
                        ep_id = f"ep:{episode_id}"
                        await conn.execute(
                            """INSERT INTO pg_graph_edges(src,dst,rel,props)
                               VALUES ($1,$2,'HAS_TRIPLE','{}'::jsonb)""",
                            ep_id, s_id,
                        )
                    n += 1
        return n

    async def delete_episode(self, episode_id: str) -> bool:
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                deleted = await conn.fetchval(
                    "DELETE FROM pg_graph_episodes WHERE id = $1 RETURNING id",
                    episode_id,
                )
                await conn.execute(
                    "DELETE FROM pg_graph_edges WHERE src = $1 OR dst = $1",
                    f"ep:{episode_id}",
                )
                await conn.execute(
                    "DELETE FROM pg_graph_nodes WHERE id = $1",
                    f"ep:{episode_id}",
                )
        return deleted is not None
