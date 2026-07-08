"""
RAG FAISS Episodic Memory + Graph Memory Manager
====================================================
Implements ``BaseMemoryStore`` with three layers:

1. **Sliding window** (RAM, bounded deque) — last N turns verbatim.
2. **FAISS episodic index** (local, persisted) — every *summary* entry
   is embedded and added to a flat inner-product index so older context
   can be retrieved by semantic similarity.
3. **Graph memory** — optional pointer to a ``BaseGraphStore`` so
   retrieval can mix in entity-rich history.

The FAISS index file lives at ``settings.memory.faiss_index_path``
alongside a parallel JSON list of ``MemoryEntry`` dicts (one per
vector).  Both are persisted by ``flush()``.
"""

from __future__ import annotations

import asyncio
import json
from collections import deque
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np

from src.config import get_settings
from src.ingestion.embedder import CachedOllamaEmbedder
from src.monitoring.logger import get_logger
from src.storage.base import BaseGraphStore, BaseMemoryStore, MemoryEntry

log = get_logger(__name__)

try:  # pragma: no cover - faiss optional on some platforms
    import faiss  # type: ignore
    _FAISS_AVAILABLE = True
except Exception:
    faiss = None  # type: ignore
    _FAISS_AVAILABLE = False


class _FaissEpisodicIndex:
    """Thin wrapper around a FAISS IndexFlatIP + parallel metadata list."""

    def __init__(self, dim: int, path: Path) -> None:
        self._dim = dim
        self._path = path
        self._index_file = path / "episodic.index"
        self._meta_file = path / "episodic.json"
        self._entries: list[MemoryEntry] = []
        self._index: Any = None
        self._lock = Lock()
        self._load()

    def _load(self) -> None:
        if not _FAISS_AVAILABLE:
            return
        if self._index_file.exists() and self._meta_file.exists():
            try:
                self._index = faiss.read_index(str(self._index_file))
                raw = json.loads(self._meta_file.read_text(encoding="utf-8"))
                self._entries = [self._from_dict(d) for d in raw]
                log.info(
                    "FAISS episodic index loaded",
                    extra={"entries": len(self._entries)},
                )
                return
            except Exception as exc:
                log.warning("FAISS load failed; starting fresh", extra={"error": str(exc)})
        self._index = faiss.IndexFlatIP(self._dim)

    def add(self, vector: list[float], entry: MemoryEntry) -> None:
        if self._index is None:
            return
        with self._lock:
            vec = np.asarray(vector, dtype="float32").reshape(1, -1)
            self._index.add(vec)
            self._entries.append(entry)

    def search(self, vector: list[float], top_k: int) -> list[tuple[MemoryEntry, float]]:
        if self._index is None or self._index.ntotal == 0:
            return []
        with self._lock:
            vec = np.asarray(vector, dtype="float32").reshape(1, -1)
            scores, idx = self._index.search(vec, min(top_k, self._index.ntotal))
        out: list[tuple[MemoryEntry, float]] = []
        for score, i in zip(scores[0].tolist(), idx[0].tolist(), strict=False):
            if 0 <= i < len(self._entries):
                out.append((self._entries[i], float(score)))
        return out

    def persist(self) -> None:
        if self._index is None:
            return
        self._path.mkdir(parents=True, exist_ok=True)
        with self._lock:
            faiss.write_index(self._index, str(self._index_file))
            self._meta_file.write_text(
                json.dumps([self._to_dict(e) for e in self._entries], default=str),
                encoding="utf-8",
            )

    @staticmethod
    def _to_dict(e: MemoryEntry) -> dict[str, Any]:
        return {
            "role": e.role,
            "content": e.content,
            "timestamp": e.timestamp.isoformat(),
            "metadata": e.metadata,
            "is_summary": e.is_summary,
            "turn_index": e.turn_index,
        }

    @staticmethod
    def _from_dict(d: dict[str, Any]) -> MemoryEntry:
        from datetime import datetime
        return MemoryEntry(
            role=d["role"],
            content=d["content"],
            timestamp=datetime.fromisoformat(d["timestamp"]),
            metadata=d.get("metadata") or {},
            is_summary=d.get("is_summary", False),
            turn_index=d.get("turn_index", 0),
        )


class EpisodicMemoryStore(BaseMemoryStore):
    """
    Sliding-window + FAISS episodic memory, with optional graph layer.

    Args:
        embedder:     Pre-built ``CachedOllamaEmbedder`` (reuse the ingest one).
        graph_store:  Optional ``BaseGraphStore`` for graph episodic recall.
        session_id:   Logical session identifier (used for metadata).
    """

    def __init__(
        self,
        embedder: CachedOllamaEmbedder | None = None,
        graph_store: BaseGraphStore | None = None,
        session_id: str = "default",
    ) -> None:
        cfg = get_settings()
        self._cfg = cfg.memory
        self._session_id = session_id
        self._embedder = embedder or CachedOllamaEmbedder()
        self._graph_store = graph_store

        self._window: deque[MemoryEntry] = deque(maxlen=self._cfg.window_size * 2)
        self._turn_counter: int = 0
        self._faiss = _FaissEpisodicIndex(
            dim=cfg.postgres.vector_dim,
            path=self._cfg.faiss_index_path,
        )

    # ------------------------------------------------------------------
    # Sliding window
    # ------------------------------------------------------------------

    def add_turn(
        self,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._turn_counter += 1
        entry = MemoryEntry(
            role=role,
            content=content,
            metadata=(metadata or {}) | {"session_id": self._session_id},
            turn_index=self._turn_counter,
        )
        self._window.append(entry)

    def get_window(self) -> list[MemoryEntry]:
        return list(self._window)

    def clear(self) -> None:
        self._window.clear()

    # ------------------------------------------------------------------
    # Episodic recall
    # ------------------------------------------------------------------

    def add_episodic(self, summary_entry: MemoryEntry) -> None:
        """Embed and persist a summary entry to the FAISS index."""
        try:
            vec = self._embedder.run(texts=[summary_entry.content])["embeddings"][0]
        except Exception as exc:
            log.warning(
                "Embedding summary failed; skipping episodic add",
                extra={"error": str(exc)},
            )
            return
        self._faiss.add(vec, summary_entry)

    async def retrieve_context(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[MemoryEntry]:
        # 1) FAISS episodic
        try:
            qvec = self._embedder.run(texts=[query])["embeddings"][0]
        except Exception:
            qvec = None

        faiss_hits: list[tuple[MemoryEntry, float]] = []
        if qvec is not None:
            faiss_hits = self._faiss.search(qvec, top_k=top_k)

        # 2) Graph episodic (optional, async)
        graph_entries: list[MemoryEntry] = []
        if self._graph_store is not None:
            try:
                gr = await self._graph_store.search(
                    query=query,
                    top_k=self._cfg.graph_memory_top_k,
                    max_hops=1,
                )
                for fact in gr.graphiti_facts[: self._cfg.graph_memory_top_k]:
                    graph_entries.append(
                        MemoryEntry(
                            role="system",
                            content=fact,
                            is_summary=True,
                            metadata={"source": "graph_memory"},
                        )
                    )
            except Exception as exc:
                log.warning(
                    "Graph memory lookup failed",
                    extra={"error": str(exc)},
                )

        # Merge + dedupe by content
        seen: set[str] = set()
        merged: list[MemoryEntry] = []
        for entry, _score in faiss_hits:
            if entry.content in seen:
                continue
            seen.add(entry.content)
            merged.append(entry)

        for entry in graph_entries:
            if entry.content in seen:
                continue
            seen.add(entry.content)
            merged.append(entry)

        return merged[:top_k]

    async def flush(self) -> None:
        await asyncio.to_thread(self._faiss.persist)
