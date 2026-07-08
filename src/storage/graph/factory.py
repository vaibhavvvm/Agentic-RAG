"""
RAG Graph Store Factory
==========================
Returns the correct ``BaseGraphStore`` implementation for the current
CLI/config choice, with a built-in fallback chain:

    neo4j    → FalkorDB → PG-Graph → NullGraphStore
    falkor   → Neo4j    → PG-Graph → NullGraphStore
    pggraph  → Neo4j    → FalkorDB → NullGraphStore
    none     → NullGraphStore (disables graph features)

The fallback chain means that if the requested backend cannot be
initialised (e.g. the service is down or the client library missing)
the system still runs — just with a different backend.
"""

from __future__ import annotations

from typing import Any

from src.config import get_settings
from src.monitoring.logger import get_logger
from src.storage.base import BaseGraphStore, GraphSearchResult

log = get_logger(__name__)


class NullGraphStore(BaseGraphStore):
    """No-op graph store used when graph features are disabled."""

    async def initialise(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def add_episode(
        self, content: str, episode_id: str, metadata: dict[str, Any] | None = None
    ) -> None:
        return None

    async def search(self, query: str, top_k: int = 5, max_hops: int = 2) -> GraphSearchResult:
        return GraphSearchResult(nodes=[], edges=[])

    async def get_entity_subgraph(self, entity_id: str, max_hops: int = 2) -> GraphSearchResult:
        return GraphSearchResult(nodes=[], edges=[])

    async def delete_episode(self, episode_id: str) -> bool:
        return False


_FALLBACK_ORDER: dict[str, list[str]] = {
    "neo4j":   ["neo4j", "falkor", "pggraph"],
    "falkor":  ["falkor", "neo4j", "pggraph"],
    "pggraph": ["pggraph", "neo4j", "falkor"],
    "none":    [],
}


async def _try_instantiate(name: str) -> BaseGraphStore | None:
    try:
        if name == "neo4j":
            from src.storage.graph.neo4j_store import Neo4jGraphStore
            store: BaseGraphStore = Neo4jGraphStore()
        elif name == "falkor":
            from src.storage.graph.falkor_store import FalkorGraphStore
            store = FalkorGraphStore()
        elif name == "pggraph":
            from src.storage.graph.pg_graph_store import PgGraphStore
            store = PgGraphStore()
        else:
            return None
        await store.initialise()
        log.info("graph backend online", extra={"backend": name})
        return store
    except Exception as exc:
        log.warning(
            "graph backend failed to initialise",
            extra={"backend": name, "error": str(exc)[:200]},
        )
        try:
            await store.close()  # type: ignore[has-type]
        except Exception:
            pass
        return None


async def build_graph_store(preferred: str | None = None) -> BaseGraphStore:
    """
    Instantiate a graph store following the fallback chain.

    Args:
        preferred: Override for ``settings.graph_backend`` (e.g. passed
                   from the CLI).
    Returns:
        A live ``BaseGraphStore`` (never raises — falls back to
        ``NullGraphStore`` if every option fails).
    """
    cfg = get_settings()
    choice = (preferred or cfg.graph_backend).lower()

    if choice == "none":
        log.info("graph features disabled (backend=none)")
        return NullGraphStore()

    for candidate in _FALLBACK_ORDER.get(choice, [choice]):
        store = await _try_instantiate(candidate)
        if store is not None:
            return store

    log.error("all graph backends failed — returning NullGraphStore")
    return NullGraphStore()
