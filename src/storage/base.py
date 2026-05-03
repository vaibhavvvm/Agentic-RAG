"""
RAG3 Abstract Storage Interfaces
==================================
Defines the protocol contracts that every concrete storage backend must
satisfy.  Keeping these as pure ABCs (Abstract Base Classes) lets the
rest of the application depend on the *interface*, not on any specific
database library.

Interfaces defined
------------------
BaseVectorStore   — embedding-based document storage and hybrid search
BaseGraphStore    — Neo4j/Graphiti graph knowledge-base operations
BaseMemoryStore   — conversational memory (sliding window + episodic)
BaseSummaryStore  — document-level summary index storage

All methods that perform I/O are declared ``async`` to support both
``asyncpg`` (PostgreSQL) and the Neo4j async driver without blocking the
event loop.

Concrete implementations live in:
    src/storage/postgres/vector_store.py   → BaseVectorStore
    src/storage/postgres/summary_store.py  → BaseSummaryStore
    src/storage/graph/neo4j_store.py       → BaseGraphStore
    src/memory/vector_store.py             → BaseMemoryStore (FAISS)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from haystack.dataclasses import Document


# ---------------------------------------------------------------------------
# Shared value objects / DTOs
# ---------------------------------------------------------------------------


class SearchMode(str, Enum):
    """
    Retrieval strategy selector for vector stores.

    VECTOR  — pure cosine-similarity ANN search.
    BM25    — keyword/TF-IDF fulltext search (pgvector + tsvector).
    HYBRID  — combined vector + BM25 via Reciprocal Rank Fusion (RRF).
    """

    VECTOR = "vector"
    BM25 = "bm25"
    HYBRID = "hybrid"


@dataclass(frozen=True)
class SearchResult:
    """
    Single result entry returned by any store's search method.

    Attributes:
        document:    The Haystack ``Document`` object (contains content +
                     metadata + optional embedding).
        score:       Relevance score.  Semantics depend on the search mode:
                     vector → cosine similarity [0, 1];
                     BM25 → BM25 score (higher is better);
                     hybrid / RRF → fused rank score (lower rank = higher score).
        source:      Which retrieval path produced this result.
        rank:        Zero-based position in the result list (0 = most relevant).
    """

    document: Document
    score: float
    source: SearchMode = SearchMode.VECTOR
    rank: int = 0


@dataclass(frozen=True)
class GraphNode:
    """
    Lightweight representation of a Neo4j node.

    Attributes:
        node_id:    Neo4j element ID (string in Neo4j 5+).
        labels:     Set of Neo4j labels (e.g. ``{"Entity", "Person"}``).
        properties: Arbitrary node properties.
    """

    node_id: str
    labels: frozenset[str]
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GraphEdge:
    """
    Lightweight representation of a Neo4j relationship.

    Attributes:
        edge_id:    Neo4j element ID for the relationship.
        rel_type:   Relationship type string (e.g. ``"MENTIONS"``).
        source_id:  Element ID of the start node.
        target_id:  Element ID of the end node.
        properties: Arbitrary relationship properties.
    """

    edge_id: str
    rel_type: str
    source_id: str
    target_id: str
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GraphSearchResult:
    """
    Result bundle returned by graph search.

    Attributes:
        nodes:           Matched or reachable nodes.
        edges:           Traversed relationships.
        paths:           Optional list of node-ID sequences representing paths.
        graphiti_facts:  Plain-text fact strings extracted by Graphiti.
        score:           Aggregate relevance estimate (implementation-defined).
    """

    nodes: list[GraphNode]
    edges: list[GraphEdge]
    paths: list[list[str]] = field(default_factory=list)
    graphiti_facts: list[str] = field(default_factory=list)
    score: float = 0.0


@dataclass
class MemoryEntry:
    """
    A single memory item (message turn or summary) stored in the memory layer.

    Attributes:
        role:       Speaker role — ``"user"`` or ``"assistant"``.
        content:    Text content of the turn or summary.
        timestamp:  UTC creation time.
        metadata:   Optional bag of extra attributes (e.g. intent, session_id).
        is_summary: ``True`` if this entry is an LLM-generated summary of
                    older turns rather than a verbatim message.
        turn_index: Monotonically increasing turn counter within a session.
    """

    role: str
    content: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)
    is_summary: bool = False
    turn_index: int = 0


@dataclass(frozen=True)
class SummaryEntry:
    """
    A document-level or topic-level summary stored in the summary index.

    Attributes:
        doc_id:      Source document identifier.
        summary:     Summary text.
        summary_type: One of ``"full"``, ``"topic"``, ``"section"``.
        embedding:   Pre-computed vector for the summary (may be None if not
                     yet embedded).
        metadata:    Additional metadata (page range, section title, etc.).
    """

    doc_id: str
    summary: str
    summary_type: str  # "full" | "topic" | "section"
    embedding: list[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Abstract Base Classes
# ---------------------------------------------------------------------------


class BaseVectorStore(ABC):
    """
    Contract for pgvector-backed document stores.

    Implementors must support:
    * Upsert of Haystack ``Document`` objects with pre-computed embeddings.
    * Three retrieval modes: VECTOR, BM25, HYBRID (RRF).
    * Metadata filtering via a flexible filter dict.
    * Async initialisation / tear-down.
    """

    @abstractmethod
    async def initialise(self) -> None:
        """
        Ensure the backing schema (table, indices) exists.

        Must be idempotent — safe to call on every startup.
        """
        ...

    @abstractmethod
    async def close(self) -> None:
        """Release all database connections and resources."""
        ...

    @abstractmethod
    async def upsert_documents(self, documents: list[Document]) -> list[str]:
        """
        Insert or update documents in the store.

        Args:
            documents: Haystack ``Document`` objects.  Each must have a
                       non-None ``id`` and a populated ``embedding`` field.

        Returns:
            List of document IDs that were written (in the same order).
        """
        ...

    @abstractmethod
    async def search(
        self,
        query_embedding: list[float],
        query_text: str,
        top_k: int = 10,
        mode: SearchMode = SearchMode.HYBRID,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """
        Retrieve the top-k most relevant documents.

        Args:
            query_embedding: Dense vector for the query (same dim as stored
                             embeddings).
            query_text:      Raw query string, used for BM25 scoring.
            top_k:           Maximum number of results to return.
            mode:            Retrieval strategy (VECTOR / BM25 / HYBRID).
            filters:         Optional metadata equality filters, e.g.
                             ``{"source": "manual.pdf", "page": 3}``.

        Returns:
            Ordered list of ``SearchResult`` objects, most relevant first.
        """
        ...

    @abstractmethod
    async def delete_documents(self, document_ids: list[str]) -> int:
        """
        Remove documents by ID.

        Returns:
            Number of documents actually deleted.
        """
        ...

    @abstractmethod
    async def count_documents(self, filters: dict[str, Any] | None = None) -> int:
        """
        Return the number of stored documents, optionally filtered.

        Args:
            filters: Optional metadata equality filters.
        """
        ...


class BaseGraphStore(ABC):
    """
    Contract for Neo4j + Graphiti graph knowledge-base stores.

    Implementors must support:
    * Episode ingestion via Graphiti (entity/relation extraction).
    * Semantic and entity-based graph search.
    * Multi-hop subgraph extraction up to a configurable depth.
    * Async lifecycle management.
    """

    @abstractmethod
    async def initialise(self) -> None:
        """
        Verify Neo4j connectivity and ensure required indices/constraints exist.

        Must be idempotent.
        """
        ...

    @abstractmethod
    async def close(self) -> None:
        """Close the Neo4j driver and any Graphiti resources."""
        ...

    @abstractmethod
    async def add_episode(
        self,
        content: str,
        episode_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Ingest a text episode into Graphiti for entity/relation extraction.

        Graphiti will parse ``content``, identify named entities and
        relationships, and persist them as nodes/edges in Neo4j.

        Args:
            content:    Raw text of the episode (e.g. a document chunk or
                        a conversation turn).
            episode_id: Unique identifier for deduplication.
            metadata:   Optional bag of extra properties attached to the
                        episode node (timestamp, source, etc.).
        """
        ...

    @abstractmethod
    async def search(
        self,
        query: str,
        top_k: int = 5,
        max_hops: int = 2,
    ) -> GraphSearchResult:
        """
        Perform a hybrid semantic + structural graph search.

        The implementation should:
        1. Use Graphiti's semantic search to identify relevant entity nodes.
        2. Expand the neighbourhood up to ``max_hops`` to collect context.
        3. Return nodes, edges, traversal paths, and fact strings.

        Args:
            query:    Natural language query string.
            top_k:    Number of root entities to seed the traversal from.
            max_hops: Maximum relationship hops from seed entities.

        Returns:
            ``GraphSearchResult`` with populated nodes, edges, and facts.
        """
        ...

    @abstractmethod
    async def get_entity_subgraph(
        self,
        entity_id: str,
        max_hops: int = 2,
    ) -> GraphSearchResult:
        """
        Return the local subgraph centred on a specific entity.

        Args:
            entity_id: Neo4j node element ID or a named entity string.
            max_hops:  Neighbourhood radius to expand.
        """
        ...

    async def add_triples(
        self,
        triples: list[dict[str, str]],
        episode_id: str | None = None,
    ) -> int:
        """
        Persist ``(subject, relation, object)`` triples as graph edges.

        Triples are produced by the upstream ER extractor (gpt-oss-20b).
        Default implementation is a no-op so existing back-ends remain
        backwards-compatible; concrete stores override to insert entity
        nodes and relationship edges natively.

        Returns:
            Number of edges actually written.
        """
        return 0

    @abstractmethod
    async def delete_episode(self, episode_id: str) -> bool:
        """
        Remove an episode and its orphaned entities/relationships.

        Returns:
            ``True`` if the episode existed and was deleted, else ``False``.
        """
        ...


class BaseMemoryStore(ABC):
    """
    Contract for the conversational memory layer.

    The memory system operates at three levels:
    1. **Sliding window** — recent turns in RAM (fast, bounded).
    2. **Episodic FAISS** — compressed historical summaries in a local
       ANN index (medium latency, large capacity).
    3. **Graph episodic** — entity-rich history from Neo4j (async, semantic).

    Implementors must expose a unified ``retrieve_context`` that merges all
    three levels into a single ranked list of ``MemoryEntry`` objects.
    """

    @abstractmethod
    def add_turn(self, role: str, content: str, metadata: dict[str, Any] | None = None) -> None:
        """
        Append a conversation turn to the sliding window.

        Triggers episodic summarisation and FAISS persistence when the
        window exceeds its configured capacity.

        Args:
            role:     ``"user"`` or ``"assistant"``.
            content:  Text of the turn.
            metadata: Optional extra attributes (e.g. intent classification).
        """
        ...

    @abstractmethod
    def get_window(self) -> list[MemoryEntry]:
        """Return the current sliding-window turns (most recent first)."""
        ...

    @abstractmethod
    async def retrieve_context(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[MemoryEntry]:
        """
        Retrieve the most relevant historical context for a given query.

        Combines FAISS episodic search and (optionally) graph episodic
        search, then merges and deduplicates results.

        Args:
            query: The current user query string.
            top_k: Maximum number of memory entries to return.

        Returns:
            List of ``MemoryEntry`` objects ranked by relevance.
        """
        ...

    @abstractmethod
    async def flush(self) -> None:
        """
        Force-persist any buffered in-memory state to durable storage.

        Called on clean shutdown or when the session ends.
        """
        ...

    @abstractmethod
    def clear(self) -> None:
        """Reset the sliding window (does not delete persisted FAISS data)."""
        ...


class BaseSummaryStore(ABC):
    """
    Contract for the document-level summary index.

    Summaries are generated at three granularities (full / topic / section)
    and stored alongside their embeddings to support semantic retrieval of
    entire document sections without per-chunk overhead.
    """

    @abstractmethod
    async def initialise(self) -> None:
        """Ensure the backing schema and indices exist. Must be idempotent."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Release all resources."""
        ...

    @abstractmethod
    async def upsert_summary(self, entry: SummaryEntry) -> str:
        """
        Insert or replace a summary entry.

        Args:
            entry: ``SummaryEntry`` to persist.

        Returns:
            The persisted entry's doc_id.
        """
        ...

    @abstractmethod
    async def search_summaries(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        summary_type: str | None = None,
    ) -> list[SummaryEntry]:
        """
        Retrieve the most semantically relevant summaries.

        Args:
            query_embedding: Dense vector for the query.
            top_k:           Maximum results to return.
            summary_type:    Optional filter — ``"full"``, ``"topic"``, or
                             ``"section"``; ``None`` searches all types.

        Returns:
            Ordered list of ``SummaryEntry`` objects.
        """
        ...

    @abstractmethod
    async def get_by_doc_id(self, doc_id: str) -> list[SummaryEntry]:
        """
        Return all summary entries for a given source document.

        Args:
            doc_id: Source document identifier.
        """
        ...

    @abstractmethod
    async def delete_by_doc_id(self, doc_id: str) -> int:
        """
        Remove all summaries associated with ``doc_id``.

        Returns:
            Number of records deleted.
        """
        ...
