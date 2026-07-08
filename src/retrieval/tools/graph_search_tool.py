"""
RAG Graph Search Tool
========================
Wraps the graph store's semantic + subgraph search into a simple
``__call__(query) -> list[Document]`` surface that the agent can use.
"""

from __future__ import annotations

import asyncio

from haystack.dataclasses import Document

from src.config import get_settings
from src.monitoring.logger import get_logger
from src.retrieval.cache import TTLCache, hash_key
from src.storage.base import BaseGraphStore

log = get_logger(__name__)


class GraphSearchTool:
    """Agent-compatible facade around ``BaseGraphStore.search``."""

    def __init__(
        self,
        graph_store: BaseGraphStore,
        cache: TTLCache | None = None,
        top_k: int | None = None,
        max_hops: int = 2,
    ) -> None:
        cfg = get_settings().retrieval
        self._store = graph_store
        self._cache: TTLCache = cache or TTLCache()
        self._top_k = top_k or cfg.top_k_graph
        self._max_hops = max_hops

    def __call__(self, query: str) -> list[Document]:
        return asyncio.run(self.arun(query))

    async def arun(self, query: str) -> list[Document]:
        cache_key = hash_key("graph_tool", query, self._top_k, self._max_hops)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        result = await self._store.search(
            query=query, top_k=self._top_k, max_hops=self._max_hops
        )

        docs: list[Document] = []
        for i, fact in enumerate(result.graphiti_facts):
            docs.append(
                Document(
                    id=f"graph::fact::{i}",
                    content=fact,
                    meta={"source": "graph_fact", "rank": i},
                )
            )
        for i, node in enumerate(result.nodes):
            name = node.properties.get("name") or node.properties.get("canonical_name") or node.node_id
            content = f"Entity: {name}\nLabels: {', '.join(node.labels)}"
            docs.append(
                Document(
                    id=f"graph::node::{node.node_id}",
                    content=content,
                    meta={"source": "graph_node", "node_id": node.node_id, "rank": i},
                )
            )

        self._cache.set(cache_key, docs)
        return docs
