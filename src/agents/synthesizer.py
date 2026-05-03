"""
RAG3 Response Synthesiser
===========================
Formats the final user-facing response and attaches intent-specific
metadata. Deliberately LLM-free: callers supply the answer text; this
class only normalises shape, citations, and trace fields.

Intent-specific metadata
------------------------
``general_chat``  — no sources, trims latency/intent only.
``vector_retrieval``   — citations ``[1]…[N]`` map to chunk snippets
                         with page + doc_id.
``graph_retrieval``    — sources split into ``facts`` and ``entities``.
``hybrid_retrieval``   — combines vector citations and graph-facts
                         buckets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from haystack.dataclasses import Document


@dataclass
class SynthesisedResponse:
    answer: str
    sources: list[dict[str, Any]] = field(default_factory=list)
    facts: list[str] = field(default_factory=list)
    entities: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "sources": self.sources,
            "facts": self.facts,
            "entities": self.entities,
            "metadata": self.metadata,
        }


class Synthesiser:
    """Compose the final response payload with intent-specific fields."""

    def __init__(self, include_sources: bool = True, max_sources: int = 5) -> None:
        self._include_sources = include_sources
        self._max_sources = max_sources

    def run(
        self,
        answer: str,
        documents: list[Document] | None = None,
        trace: dict[str, Any] | None = None,
    ) -> SynthesisedResponse:
        trace = dict(trace or {})
        intent = str(trace.get("intent", "")).lower()
        docs = documents or []

        response = SynthesisedResponse(answer=answer.strip(), metadata=trace)

        if not self._include_sources or not docs:
            return response

        if intent == "general_chat":
            return response

        if intent == "graph_retrieval":
            response.facts, response.entities = self._split_graph_docs(docs)
            response.sources = self._compact_sources(docs[: self._max_sources])
            return response

        if intent == "hybrid_retrieval":
            response.facts, response.entities = self._split_graph_docs(docs)
            vector_like = [
                d for d in docs
                if (d.meta or {}).get("source") not in {"graph_fact", "graph_node"}
            ]
            response.sources = self._citations(vector_like[: self._max_sources])
            return response

        # Default: vector_retrieval
        response.sources = self._citations(docs[: self._max_sources])
        return response

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _citations(docs: list[Document]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for i, doc in enumerate(docs, start=1):
            meta = doc.meta or {}
            out.append({
                "index": i,
                "id": doc.id,
                "snippet": (doc.content or "")[:320],
                "source": meta.get("source_doc_id") or meta.get("doc_id") or meta.get("source") or "",
                "page": meta.get("page_number") or meta.get("page"),
                "score": meta.get("rerank_score") or meta.get("score"),
            })
        return out

    @staticmethod
    def _compact_sources(docs: list[Document]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for doc in docs:
            meta = doc.meta or {}
            out.append({
                "id": doc.id,
                "snippet": (doc.content or "")[:200],
                "kind": meta.get("source", "graph"),
            })
        return out

    @staticmethod
    def _split_graph_docs(
        docs: list[Document],
    ) -> tuple[list[str], list[dict[str, Any]]]:
        facts: list[str] = []
        entities: list[dict[str, Any]] = []
        for d in docs:
            meta = d.meta or {}
            if meta.get("source") == "graph_fact":
                facts.append((d.content or "").strip())
            elif meta.get("source") == "graph_node":
                entities.append({
                    "id": meta.get("node_id", d.id),
                    "content": (d.content or "").strip(),
                })
        return facts, entities
