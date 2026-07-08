"""
RAG Neo4j + Graphiti Graph Store
===================================
Implements ``BaseGraphStore`` using:

* **Neo4j 5 async driver** — low-level Cypher for subgraph extraction.
* **Graphiti** — high-level entity/relation extraction from plain text,
  writing episodes, entities, and relationships into the same Neo4j DB.

Design
------
* Graphiti is *optional* at runtime — if the ``graphiti_core`` import
  fails or initialisation errors occur, the store falls back to a plain
  Neo4j-backed episode sink (episodes become ``:Episode`` nodes with a
  ``MENTIONS`` relationship to manually-extracted keyword nodes).  This
  keeps the pipeline runnable offline / in CI without heavy NLP models.
* All public methods are ``async``.
* ``search()`` performs semantic entity matching followed by
  ``max_hops`` BFS expansion and returns nodes, edges, and Graphiti
  facts.

Cypher schema touched
---------------------
* ``(:Episode {id, content, created_at, source})``
* ``(:Entity {name, canonical_name})``
* Relationships: ``[:MENTIONS]``, ``[:RELATES_TO {type, weight}]``
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
import json
from typing import Any

from neo4j import AsyncGraphDatabase

from src.config import get_settings
from src.monitoring.logger import get_logger, timed_operation
from src.storage.base import (
    BaseGraphStore,
    GraphEdge,
    GraphNode,
    GraphSearchResult,
)

log = get_logger(__name__)

# Graphiti is optional — import lazily
try:  # pragma: no cover - optional dep
    from graphiti_core import Graphiti  # type: ignore
    _GRAPHITI_AVAILABLE = True
except Exception:  # broad: graphiti may fail to import on some platforms
    Graphiti = None  # type: ignore[assignment]
    _GRAPHITI_AVAILABLE = False


_STOPWORDS = frozenset({
    "the", "is", "a", "an", "and", "or", "of", "to", "in", "for", "on",
    "with", "as", "by", "at", "from", "this", "that", "these", "those",
    "it", "its", "be", "are", "was", "were", "has", "have", "had", "but",
    "if", "then", "than", "which", "who", "whom", "what", "when", "where",
    "why", "how", "not", "no", "so", "do", "does", "did",
})


def _extract_keywords(text: str, max_keywords: int = 8) -> list[str]:
    """
    Very light keyword extractor — splits on non-word chars, lowercases,
    drops stopwords and tokens ≤ 2 chars, returns uniques preserving order.
    """
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9\-_]{2,}", text or "")
    seen: dict[str, None] = {}
    for tok in tokens:
        low = tok.lower()
        if low in _STOPWORDS:
            continue
        seen.setdefault(low, None)
        if len(seen) >= max_keywords:
            break
    return list(seen.keys())


class Neo4jGraphStore(BaseGraphStore):
    """
    Neo4j + Graphiti graph knowledge-base.

    Args:
        uri:            Neo4j bolt URI.
        user:           DB user.
        password:       DB password.
        database:       Target Neo4j database name.
        use_graphiti:   Attempt Graphiti initialisation. Defaults to True
                        when the library is importable.
    """

    def __init__(
        self,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
        database: str | None = None,
        use_graphiti: bool | None = None,
    ) -> None:
        cfg = get_settings().neo4j
        self._uri: str = uri or cfg.uri
        self._user: str = user or cfg.user
        self._password: str = password or cfg.password.get_secret_value()
        self._database: str = database or cfg.database
        self._max_hop_depth: int = cfg.max_hop_depth
        self._graphiti_limit: int = cfg.graphiti_episode_limit

        self._use_graphiti: bool = (
            use_graphiti if use_graphiti is not None else _GRAPHITI_AVAILABLE
        )
        self._driver: Any = None
        self._graphiti: Any = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialise(self) -> None:
        if self._driver is not None:
            return

        self._driver = AsyncGraphDatabase.driver(
            self._uri, auth=(self._user, self._password)
        )
        # Verify connectivity + create indices
        async with self._driver.session(database=self._database) as session:
            await session.run("RETURN 1;")
            await session.run(
                "CREATE CONSTRAINT episode_id IF NOT EXISTS "
                "FOR (e:Episode) REQUIRE e.id IS UNIQUE;"
            )
            await session.run(
                "CREATE INDEX entity_name IF NOT EXISTS "
                "FOR (n:Entity) ON (n.canonical_name);"
            )

        if self._use_graphiti and _GRAPHITI_AVAILABLE:
            try:  # pragma: no cover - runtime only
                self._graphiti = Graphiti(
                    self._uri, self._user, self._password
                )
                await self._graphiti.build_indices_and_constraints()
                log.info("Graphiti initialised")
            except Exception as exc:
                log.warning(
                    "Graphiti init failed, falling back to plain Neo4j",
                    extra={"error": str(exc)},
                )
                self._graphiti = None

        log.info("Neo4jGraphStore initialised", extra={"uri": self._uri})

    async def close(self) -> None:
        if self._graphiti is not None:
            try:  # pragma: no cover
                await self._graphiti.close()
            except Exception:
                pass
            self._graphiti = None
        if self._driver is not None:
            await self._driver.close()
            self._driver = None

    # ------------------------------------------------------------------
    # Episode ingestion
    # ------------------------------------------------------------------

    async def add_episode(
        self,
        content: str,
        episode_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if self._graphiti is not None:
            try:  # pragma: no cover
                await self._graphiti.add_episode(
                    name=episode_id,
                    episode_body=content,
                    source_description=(metadata or {}).get("source", "rag"),
                    reference_time=datetime.now(tz=timezone.utc),
                )
                return
            except Exception as exc:
                log.warning(
                    "Graphiti add_episode failed; falling back",
                    extra={"error": str(exc), "episode_id": episode_id},
                )

        # Fallback: plain Neo4j write
        assert self._driver is not None
        keywords = _extract_keywords(content)
        params = {
            "id": episode_id,
            "content": content,
            "created_at": datetime.now(tz=timezone.utc).isoformat(),
            "metadata": json.dumps(metadata) if metadata else "{}",
            "keywords": keywords,
        }
        cypher = """
            MERGE (e:Episode {id: $id})
            SET e.content = $content,
                e.created_at = $created_at,
                e.metadata = $metadata
            WITH e
            UNWIND $keywords AS kw
              MERGE (n:Entity {canonical_name: kw})
                ON CREATE SET n.name = kw
              MERGE (e)-[:MENTIONS]->(n);
        """
        with timed_operation(log, "neo4j.add_episode"):
            async with self._driver.session(database=self._database) as session:
                await session.run(cypher, **params)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    async def search(
        self,
        query: str,
        top_k: int = 5,
        max_hops: int = 2,
    ) -> GraphSearchResult:
        facts: list[str] = []
        seed_entity_names: list[str] = []

        if self._graphiti is not None:
            try:  # pragma: no cover
                results = await self._graphiti.search(
                    query=query, num_results=min(top_k, self._graphiti_limit)
                )
                for r in results or []:
                    fact = getattr(r, "fact", None) or str(r)
                    facts.append(fact)
                    name = getattr(r, "source_node_name", None) or getattr(
                        r, "name", None
                    )
                    if name:
                        seed_entity_names.append(str(name).lower())
            except Exception as exc:
                log.warning(
                    "Graphiti search failed; using keyword fallback",
                    extra={"error": str(exc)},
                )

        if not seed_entity_names:
            seed_entity_names = _extract_keywords(query, max_keywords=top_k)

        if not seed_entity_names:
            return GraphSearchResult(nodes=[], edges=[], graphiti_facts=facts)

        max_hops = min(max_hops, self._max_hop_depth)
        return await self._expand_subgraph(
            seed_entity_names, max_hops=max_hops, facts=facts
        )

    async def _expand_subgraph(
        self,
        seed_names: list[str],
        max_hops: int,
        facts: list[str],
    ) -> GraphSearchResult:
        assert self._driver is not None
        cypher = f"""
            UNWIND $names AS name
            MATCH (seed:Entity)
              WHERE toLower(seed.canonical_name) = toLower(name)
                 OR toLower(seed.name) = toLower(name)
            WITH collect(DISTINCT seed) AS seeds
            UNWIND seeds AS seed
            OPTIONAL MATCH path = (seed)-[*0..{int(max_hops)}]-(m)
            UNWIND nodes(path) AS n
            WITH collect(DISTINCT n) AS ns, collect(DISTINCT path) AS paths
            UNWIND paths AS p
            UNWIND relationships(p) AS r
            RETURN ns AS nodes, collect(DISTINCT r) AS rels;
        """
        async with self._driver.session(database=self._database) as session:
            record = await (await session.run(cypher, names=seed_names)).single()

        nodes: list[GraphNode] = []
        edges: list[GraphEdge] = []
        if record is not None:
            for n in record["nodes"] or []:
                nodes.append(
                    GraphNode(
                        node_id=str(n.element_id),
                        labels=frozenset(n.labels),
                        properties=dict(n),
                    )
                )
            for r in record["rels"] or []:
                edges.append(
                    GraphEdge(
                        edge_id=str(r.element_id),
                        rel_type=r.type,
                        source_id=str(r.start_node.element_id),
                        target_id=str(r.end_node.element_id),
                        properties=dict(r),
                    )
                )

        return GraphSearchResult(
            nodes=nodes,
            edges=edges,
            graphiti_facts=facts,
            score=float(len(facts) + len(nodes) * 0.1),
        )

    async def get_entity_subgraph(
        self,
        entity_id: str,
        max_hops: int = 2,
    ) -> GraphSearchResult:
        return await self._expand_subgraph(
            [entity_id], max_hops=max_hops, facts=[]
        )

    async def add_triples(
        self,
        triples: list[dict[str, str]],
        episode_id: str | None = None,
    ) -> int:
        """Persist ER-extracted triples as (:Entity)-[:REL]->(:Entity)."""
        assert self._driver is not None
        if not triples:
            return 0
        cypher = """
            UNWIND $rows AS row
            MERGE (s:Entity {name: toLower(row.subject)})
            MERGE (o:Entity {name: toLower(row.object)})
            MERGE (s)-[r:RELATED {type: row.relation}]->(o)
            ON CREATE SET r.created_at = timestamp()
            FOREACH (_ IN CASE WHEN $ep IS NULL THEN [] ELSE [1] END |
                MERGE (ep:Episode {id: $ep})
                MERGE (ep)-[:HAS_TRIPLE]->(s)
            )
            RETURN count(r) AS n
        """
        async with self._driver.session(database=self._database) as session:
            record = await (await session.run(
                cypher, rows=list(triples), ep=episode_id,
            )).single()
        return int(record["n"]) if record else 0

    async def delete_episode(self, episode_id: str) -> bool:
        assert self._driver is not None
        cypher = """
            MATCH (e:Episode {id: $id})
            DETACH DELETE e
            RETURN count(e) AS n;
        """
        async with self._driver.session(database=self._database) as session:
            record = await (await session.run(cypher, id=episode_id)).single()
        return bool(record and record["n"] > 0)
