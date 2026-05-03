"""
Vector-store factory: pgvector primary, in-memory fallback.

CLI flag ``--vector-backend`` maps to ``settings.vector_backend``:
  * ``pgvector`` — strict; raises if unreachable.
  * ``memory``   — always uses ``InMemoryVectorStore``.
  * ``auto``     — tries pgvector, falls back to memory on failure.
"""

from __future__ import annotations

from src.config import get_settings
from src.monitoring.logger import get_logger
from src.storage.base import BaseVectorStore
from src.storage.vector.memory_store import InMemoryVectorStore

log = get_logger(__name__)


async def build_vector_store(preferred: str | None = None) -> BaseVectorStore:
    cfg = get_settings()
    choice = (preferred or cfg.vector_backend).lower()

    if choice == "memory":
        store: BaseVectorStore = InMemoryVectorStore()
        await store.initialise()
        return store

    # Try pgvector
    try:
        from src.storage.postgres.vector_store import PgVectorStore
        pg = PgVectorStore()
        await pg.initialise()
        log.info("vector backend online", extra={"backend": "pgvector"})
        return pg
    except Exception as exc:
        if choice == "pgvector":
            raise
        log.warning(
            "pgvector unreachable — falling back to in-memory store",
            extra={"error": str(exc)[:200]},
        )
        mem = InMemoryVectorStore()
        await mem.initialise()
        return mem
