"""
RAG FalkorDB Graph Store
===========================
FalkorDB is a Redis-module graph DB speaking OpenCypher. This adapter
implements the ``BaseGraphStore`` contract using the ``falkordb``
Python client. Used as a lighter-weight alternative to Neo4j when the
system is launched with ``--graph-backend falkor``.

Entity extraction is handled by the same regex fallback used in the
Neo4j adapter (Graphiti currently targets Neo4j directly).
"""

from __future__ import annotations

import re
from typing import Any

from src.config import get_settings
from src.monitoring.logger import get_logger
from src.storage.base import (
    BaseGraphStore,
    GraphEdge,
    GraphNode,
    GraphSearchResult,
)

log = get_logger(__name__)

try:  # pragma: no cover - optional
    from falkordb import FalkorDB
    _FALKOR_AVAILABLE = True
except Exception:
    FalkorDB = None  # type: ignore
    _FALKOR_AVAILABLE = False


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


class FalkorGraphStore(BaseGraphStore):
    """Graph store backed by FalkorDB (Redis OpenCypher graph)."""

    def __init__(self) -> None:
        cfg = get_settings().falkor
        if not _FALKOR_AVAILABLE:
            raise RuntimeError(
                "falkordb package not installed — run `pip install falkordb`."
            )
        pwd = cfg.password.get_secret_value() if cfg.password else None
        self._db = FalkorDB(host=cfg.host, port=cfg.port, password=pwd)
        self._graph = self._db.select_graph(cfg.graph_name)

    async def initialise(self) -> None:
        # FalkorDB creates indices lazily; no-op is fine.
        try:
            self._graph.query("CREATE INDEX FOR (e:Entity) ON (e.name)")
        except Exception:
            pass

    async def close(self) -> None:
        return None

    async def add_episode(
        self,
        content: str,
        episode_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        keywords = _extract_keywords(content)
        meta = metadata or {}
        self._graph.query(
            "MERGE (ep:Episode {id:$id}) SET ep.content=$c, ep.meta=$m",
            params={"id": episode_id, "c": content[:2000], "m": str(meta)},
        )
        for kw in keywords:
            self._graph.query(
                """MERGE (e:Entity {name:$n})
                   WITH e MATCH (ep:Episode {id:$id})
                   MERGE (ep)-[:MENTIONS]->(e)""",
                params={"n": kw, "id": episode_id},
            )

    async def search(
        self, query: str, top_k: int = 5, max_hops: int = 2
    ) -> GraphSearchResult:
        keywords = _extract_keywords(query)
        if not keywords:
            return GraphSearchResult(nodes=[], edges=[])
        # Match any entity by name; expand to attached episodes & neighbours.
        cypher = (
            "MATCH (e:Entity) WHERE e.name IN $kws "
            f"OPTIONAL MATCH (e)-[r*1..{max_hops}]-(m) "
            "RETURN e, r, m LIMIT $k"
        )
        result = self._graph.query(cypher, params={"kws": keywords, "k": top_k * 10})
        nodes: dict[str, GraphNode] = {}
        edges: list[GraphEdge] = []
        facts: list[str] = []

        for row in result.result_set:
            e = row[0]
            m = row[2] if len(row) > 2 else None
            if e is not None:
                nid = str(getattr(e, "id", id(e)))
                nodes[nid] = GraphNode(
                    node_id=nid,
                    labels=frozenset(getattr(e, "labels", []) or ["Entity"]),
                    properties=dict(getattr(e, "properties", {}) or {}),
                )
            if m is not None:
                mid = str(getattr(m, "id", id(m)))
                nodes[mid] = GraphNode(
                    node_id=mid,
                    labels=frozenset(getattr(m, "labels", []) or []),
                    properties=dict(getattr(m, "properties", {}) or {}),
                )
                p = dict(getattr(m, "properties", {}) or {})
                if "content" in p:
                    facts.append(str(p["content"])[:400])

        return GraphSearchResult(
            nodes=list(nodes.values())[:top_k * 5],
            edges=edges,
            paths=[],
            graphiti_facts=facts[:top_k],
        )

    async def get_entity_subgraph(
        self, entity_id: str, max_hops: int = 2
    ) -> GraphSearchResult:
        cypher = (
            "MATCH (e:Entity {name:$n}) "
            f"OPTIONAL MATCH (e)-[r*1..{max_hops}]-(m) "
            "RETURN e, r, m"
        )
        result = self._graph.query(cypher, params={"n": entity_id})
        nodes: dict[str, GraphNode] = {}
        for row in result.result_set:
            for obj in row:
                if obj is None:
                    continue
                nid = str(getattr(obj, "id", id(obj)))
                nodes[nid] = GraphNode(
                    node_id=nid,
                    labels=frozenset(getattr(obj, "labels", []) or []),
                    properties=dict(getattr(obj, "properties", {}) or {}),
                )
        return GraphSearchResult(nodes=list(nodes.values()), edges=[])

    async def add_triples(
        self,
        triples: list[dict[str, str]],
        episode_id: str | None = None,
    ) -> int:
        if not triples:
            return 0
        n = 0
        for t in triples:
            s = (t.get("subject") or "").strip().lower()
            r = (t.get("relation") or "").strip().lower() or "related"
            o = (t.get("object") or "").strip().lower()
            if not s or not o:
                continue
            try:
                self._graph.query(
                    """MERGE (a:Entity {name:$s})
                       MERGE (b:Entity {name:$o})
                       MERGE (a)-[:RELATED {type:$r}]->(b)""",
                    params={"s": s, "o": o, "r": r},
                )
                if episode_id:
                    self._graph.query(
                        """MERGE (ep:Episode {id:$id})
                           WITH ep MATCH (a:Entity {name:$s})
                           MERGE (ep)-[:HAS_TRIPLE]->(a)""",
                        params={"id": episode_id, "s": s},
                    )
                n += 1
            except Exception as exc:
                log.debug("falkor add_triple failed", extra={"err": str(exc)})
        return n

    async def delete_episode(self, episode_id: str) -> bool:
        r = self._graph.query(
            "MATCH (ep:Episode {id:$id}) DETACH DELETE ep RETURN count(ep)",
            params={"id": episode_id},
        )
        try:
            return int(r.result_set[0][0]) > 0
        except Exception:
            return False
