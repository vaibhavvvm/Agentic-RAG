"""
RAG Session Container
=======================
``RAGSession`` glues together a session id, its conversational memory
store, and the per-session cache.  Its job is to hand the orchestrator
a consistent slice of state for one user across many turns.

Summarisation trigger: once the sliding window exceeds
``memory.summarise_after_turns`` entries, the oldest half is
summarised and flushed to the FAISS episodic index.
"""

from __future__ import annotations

import uuid
from typing import Any

from src.config import get_settings
from src.ingestion.embedder import CachedOllamaEmbedder
from src.memory.memory_tools import build_context_block
from src.memory.summarizer import summarise_turns
from src.memory.vector_store import EpisodicMemoryStore
from src.monitoring.logger import get_logger
from src.retrieval.cache import TTLCache
from src.storage.base import BaseGraphStore, MemoryEntry

log = get_logger(__name__)


class RAGSession:
    """
    Per-user session state container.

    Args:
        session_id:    Stable id; auto-generated if omitted.
        embedder:      Shared embedder (reuse the ingest singleton).
        graph_store:   Optional graph store for graph memory.
    """

    def __init__(
        self,
        session_id: str | None = None,
        embedder: CachedOllamaEmbedder | None = None,
        graph_store: BaseGraphStore | None = None,
    ) -> None:
        cfg = get_settings().memory
        self.session_id: str = session_id or str(uuid.uuid4())
        self.memory = EpisodicMemoryStore(
            embedder=embedder,
            graph_store=graph_store,
            session_id=self.session_id,
        )
        self.cache: TTLCache = TTLCache()
        self._summarise_threshold: int = cfg.summarise_after_turns

    # ------------------------------------------------------------------

    def add_turn(
        self,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.memory.add_turn(role, content, metadata)
        self._maybe_summarise()

    def _maybe_summarise(self) -> None:
        window = self.memory.get_window()
        if len(window) < self._summarise_threshold:
            return
        # Summarise the oldest half; leave the newest half intact
        split = len(window) // 2
        older, newer = window[:split], window[split:]
        if not older:
            return
        summary_entry = summarise_turns(older)
        if summary_entry.content.strip():
            self.memory.add_episodic(summary_entry)
        # reset window to just the "newer" slice
        self.memory.clear()
        for t in newer:
            self.memory.add_turn(t.role, t.content, t.metadata)

    async def build_context(self, query: str, *, top_k_episodic: int = 3) -> str:
        """Assemble the memory prefix used in system prompts."""
        window = self.memory.get_window()
        episodic = await self.memory.retrieve_context(query, top_k=top_k_episodic)
        return build_context_block(window, episodic)

    async def flush(self) -> None:
        await self.memory.flush()
