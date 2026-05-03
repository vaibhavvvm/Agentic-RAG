"""
RAG3 Graph + Vector Fusion
============================
Runs vector and graph retrieval in parallel, converts graph nodes/facts
to ``Document`` objects, and fuses both streams via Reciprocal Rank
Fusion before handing off to the reranker.
"""

from __future__ import annotations

import asyncio
from typing import ClassVar

from haystack import component
from haystack.dataclasses import Document

from src.config import get_settings
from src.monitoring.logger import get_logger
from src.storage.base import (
    BaseGraphStore,
    BaseVectorStore,
    SearchMode,
    SearchResult,
)

log = get_logger(__name__)


@component
class GraphVectorFusion:
    """
    Parallel vector + graph retrieval with RRF fusion.

    Args:
        vector_store:  pgvector-backed store.
        graph_store:   neo4j-backed store.
        embedder:      Any callable(text)→vector or Haystack embedder-like.
        rrf_k:         RRF smoothing constant.
        top_k:         Final fused list size.
    """

    OUTPUT_TYPES: ClassVar[dict[str, type]] = {"documents": list}

    def __init__(
        self,
        vector_store: BaseVectorStore,
        graph_store: BaseGraphStore,
        embedder,  # CachedOllamaEmbedder-like
        rrf_k: int | None = None,
        top_k: int | None = None,
    ) -> None:
        cfg = get_settings().retrieval
        self._vector = vector_store
        self._graph = graph_store
        self._embedder = embedder
        self._rrf_k = rrf_k or cfg.rrf_k
        self._top_k = top_k or cfg.top_k_vector

    @component.output_types(documents=list)
    def run(self, query: str, top_k: int | None = None) -> dict[str, list[Document]]:
        k = top_k or self._top_k
        qvec = self._embedder.run(texts=[query])["embeddings"][0]
        return asyncio.run(self._run_async(query, qvec, k))

    async def _run_async(
        self, query: str, qvec: list[float], top_k: int
    ) -> dict[str, list[Document]]:
        vec_task = self._vector.search(
            query_embedding=qvec,
            query_text=query,
            top_k=top_k,
            mode=SearchMode.HYBRID,
        )
        graph_task = self._graph.search(query=query, top_k=top_k, max_hops=2)
        vec_hits, graph_res = await asyncio.gather(vec_task, graph_task)

        # Convert graph nodes + facts into Document form
        graph_docs: list[Document] = []
        for i, fact in enumerate(graph_res.graphiti_facts[:top_k]):
            graph_docs.append(
                Document(
                    id=f"graph::fact::{i}",
                    content=fact,
                    meta={"source": "graph_fact", "graph_rank": i},
                )
            )
        for i, node in enumerate(graph_res.nodes[:top_k]):
            name = node.properties.get("name") or node.properties.get("canonical_name") or node.node_id
            content_bits = [
                f"Entity: {name}",
                f"Labels: {', '.join(node.labels) or 'n/a'}",
            ]
            extra = {
                k_: v
                for k_, v in node.properties.items()
                if k_ not in ("name", "canonical_name") and isinstance(v, (str, int, float))
            }
            if extra:
                content_bits.append("Properties: " + ", ".join(f"{k_}={v}" for k_, v in extra.items()))
            graph_docs.append(
                Document(
                    id=f"graph::node::{node.node_id}",
                    content="\n".join(content_bits),
                    meta={"source": "graph_node", "node_id": node.node_id, "graph_rank": i},
                )
            )

        fused = self._rrf(vec_hits, graph_docs, top_k=top_k)
        return {"documents": fused}

    def _rrf(
        self,
        vec_hits: list[SearchResult],
        graph_docs: list[Document],
        top_k: int,
    ) -> list[Document]:
        k = self._rrf_k
        scores: dict[str, float] = {}
        docs: dict[str, Document] = {}

        for hit in vec_hits:
            scores[hit.document.id] = scores.get(hit.document.id, 0.0) + 1.0 / (
                k + hit.rank + 1
            )
            docs[hit.document.id] = hit.document

        for i, doc in enumerate(graph_docs):
            scores[doc.id] = scores.get(doc.id, 0.0) + 1.0 / (k + i + 1)
            docs.setdefault(doc.id, doc)

        ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        return [docs[doc_id] for doc_id, _ in ordered]
