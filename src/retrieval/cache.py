"""
RAG Retrieval Cache
======================
Thread-safe multi-level TTL LRU for retrieval results (query → docs).
Used by the orchestrator to short-circuit identical queries within a
session or across sessions during a burst.

API mirrors ``functools.lru_cache`` but adds TTL semantics.
"""

from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from threading import Lock
from typing import Any, Generic, TypeVar

from src.config import get_settings

T = TypeVar("T")


def hash_key(*parts: Any) -> str:
    """Stable SHA-256 key from any set of hashable/string-able parts."""
    h = hashlib.sha256()
    for p in parts:
        h.update(repr(p).encode("utf-8"))
        h.update(b"||")
    return h.hexdigest()


class TTLCache(Generic[T]):
    """
    Bounded LRU cache with per-entry TTL.

    Args:
        max_size:    Maximum number of entries (evicts LRU).
        ttl_seconds: Entry lifetime; 0 means "no expiry".
    """

    def __init__(self, max_size: int | None = None, ttl_seconds: int | None = None) -> None:
        cfg = get_settings().retrieval
        self._max = max_size or cfg.cache_max_size
        self._ttl = ttl_seconds if ttl_seconds is not None else cfg.cache_ttl_seconds
        self._store: OrderedDict[str, tuple[float, T]] = OrderedDict()
        self._lock = Lock()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> T | None:
        now = time.monotonic()
        with self._lock:
            item = self._store.get(key)
            if item is None:
                self._misses += 1
                return None
            expires_at, value = item
            if self._ttl and expires_at < now:
                del self._store[key]
                self._misses += 1
                return None
            # mark as most-recently-used
            self._store.move_to_end(key)
            self._hits += 1
            return value

    def set(self, key: str, value: T) -> None:
        expires_at = time.monotonic() + self._ttl if self._ttl else float("inf")
        with self._lock:
            self._store[key] = (expires_at, value)
            self._store.move_to_end(key)
            while len(self._store) > self._max:
                self._store.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
            self._hits = self._misses = 0

    def stats(self) -> dict[str, Any]:
        total = self._hits + self._misses
        return {
            "size": len(self._store),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total else 0.0,
        }
