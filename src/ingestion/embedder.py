"""
RAG Cached Ollama Text Embedder
==================================
Haystack 2.x ``@component`` wrapping the Ollama embedding API with:

* **Two-level cache** — L1 in-process LRU (dict) + L2 optional disk
  (``shelve``) so embeddings survive process restarts.
* **Batch processing** — single Ollama call per batch, respecting
  configurable batch sizes.
* **Automatic normalisation** — all vectors are L2-normalised before
  returning so cosine similarity equals dot product.
* **Metrics integration** — records cache hit/miss rates and latency.

Haystack contract
-----------------
``run(texts: list[str]) -> {"embeddings": list[list[float]]}``

Usage::

    from src.ingestion.embedder import CachedOllamaEmbedder

    embedder = CachedOllamaEmbedder()
    result = embedder.run(texts=["Hello world", "Semantic search"])
    vecs = result["embeddings"]          # list[list[float]]
"""

from __future__ import annotations

import hashlib
import json
import math
import shelve
import time
from collections import OrderedDict
from pathlib import Path
from threading import Lock
from typing import Any

import httpx
from haystack import component

from src.config import get_settings
from src.monitoring.logger import get_logger, timed_operation
from src.monitoring.metrics import MetricsCollector

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# L1: In-process LRU cache
# ---------------------------------------------------------------------------

class _LRUCache:
    """Thread-safe LRU cache with a fixed capacity."""

    def __init__(self, maxsize: int) -> None:
        self._cache: OrderedDict[str, list[float]] = OrderedDict()
        self._maxsize = maxsize
        self._lock = Lock()

    def get(self, key: str) -> list[float] | None:
        with self._lock:
            if key not in self._cache:
                return None
            self._cache.move_to_end(key)
            return self._cache[key]

    def put(self, key: str, value: list[float]) -> None:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = value
            if len(self._cache) > self._maxsize:
                self._cache.popitem(last=False)

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)


def _cache_key(model: str, text: str) -> str:
    """Deterministic cache key: sha256 of ``model + text``."""
    payload = f"{model}::{text}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _l2_normalise(vec: list[float]) -> list[float]:
    """Return L2-normalised copy of ``vec``."""
    magnitude = math.sqrt(sum(x * x for x in vec))
    if magnitude < 1e-9:
        return vec
    return [x / magnitude for x in vec]


# ---------------------------------------------------------------------------
# Haystack component
# ---------------------------------------------------------------------------


@component
class CachedOllamaEmbedder:
    """
    Ollama text embedder with two-level caching and batch support.

    Args:
        model:         Ollama embedding model tag (default from settings).
        base_url:      Ollama server URL (default from settings).
        batch_size:    Texts per Ollama HTTP request.
        l1_maxsize:    Maximum entries in the in-process LRU cache.
        disk_cache_path: Directory for the L2 shelve cache; ``None`` disables it.
        normalise:     Whether to L2-normalise output vectors.
        timeout:       HTTP request timeout in seconds.
    """

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        batch_size: int = 32,
        l1_maxsize: int = 4096,
        disk_cache_path: Path | None = None,
        normalise: bool = True,
        timeout: int | None = None,
    ) -> None:
        cfg = get_settings()
        self.model = model or cfg.ollama.embedding_model
        self.base_url = (base_url or str(cfg.ollama.base_url)).rstrip("/")
        self.batch_size = batch_size
        self.normalise = normalise
        self.timeout = timeout or cfg.ollama.timeout

        # L1 cache
        self._l1 = _LRUCache(maxsize=l1_maxsize)

        # L2 disk cache (shelve, optional)
        self._disk_path: Path | None = None
        if disk_cache_path is not None:
            disk_cache_path.mkdir(parents=True, exist_ok=True)
            self._disk_path = disk_cache_path / "embed_cache"

        self._metrics = MetricsCollector.get_instance()
        log.info(
            "CachedOllamaEmbedder initialised",
            extra={
                "model": self.model,
                "base_url": self.base_url,
                "l1_maxsize": l1_maxsize,
                "disk_cache": str(disk_cache_path) if disk_cache_path else "disabled",
            },
        )

    # ------------------------------------------------------------------
    # Haystack run()
    # ------------------------------------------------------------------

    @component.output_types(embeddings=list)
    def run(self, texts: list[str]) -> dict[str, list[list[float]]]:
        """
        Embed a list of texts, using caches for previously seen inputs.

        Args:
            texts: Non-empty list of strings to embed.

        Returns:
            ``{"embeddings": list[list[float]]}`` — one vector per input text,
            in the same order.
        """
        if not texts:
            return {"embeddings": []}

        results: dict[int, list[float]] = {}
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

        # ---- Cache lookup ----
        for i, text in enumerate(texts):
            key = _cache_key(self.model, text)
            vec = self._l1.get(key)
            if vec is None:
                vec = self._disk_get(key)
            if vec is not None:
                results[i] = vec
                self._metrics.record_event("embedder.cache_hit")
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)
                self._metrics.record_event("embedder.cache_miss")

        # ---- Embed uncached texts in batches ----
        if uncached_texts:
            new_vecs = self._embed_batched(uncached_texts)
            for idx, (orig_i, text, vec) in enumerate(
                zip(uncached_indices, uncached_texts, new_vecs)
            ):
                key = _cache_key(self.model, text)
                self._l1.put(key, vec)
                self._disk_put(key, vec)
                results[orig_i] = vec

        ordered = [results[i] for i in range(len(texts))]
        return {"embeddings": ordered}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _embed_batched(self, texts: list[str]) -> list[list[float]]:
        """Send texts to Ollama in batches; return list of vectors."""
        all_vecs: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            with self._metrics.measure("embedder.ollama_call"):
                batch_vecs = self._call_ollama(batch)
            all_vecs.extend(batch_vecs)
        return all_vecs

    def _call_ollama(self, texts: list[str]) -> list[list[float]]:
        """
        Call the Ollama ``/api/embed`` endpoint.

        Falls back to sequential ``/api/embeddings`` calls if the batch
        endpoint is unavailable (older Ollama versions).
        """
        url = f"{self.base_url}/api/embed"
        payload: dict[str, Any] = {"model": self.model, "input": texts}

        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(url, json=payload)
                resp.raise_for_status()
                data = resp.json()
                vecs: list[list[float]] = data["embeddings"]
        except (httpx.HTTPStatusError, KeyError):
            # Fallback: call single-embedding endpoint per text
            log.warning(
                "Ollama batch embed failed; falling back to single-embed",
                extra={"model": self.model, "batch_size": len(texts)},
            )
            vecs = [self._call_ollama_single(t) for t in texts]

        if self.normalise:
            vecs = [_l2_normalise(v) for v in vecs]
        return vecs

    def _call_ollama_single(self, text: str) -> list[float]:
        """Call the legacy ``/api/embeddings`` endpoint for a single text."""
        url = f"{self.base_url}/api/embeddings"
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(url, json={"model": self.model, "prompt": text})
            resp.raise_for_status()
            return resp.json()["embedding"]

    # ------------------------------------------------------------------
    # L2 disk helpers (shelve — simple key/value persistence)
    # ------------------------------------------------------------------

    def _disk_get(self, key: str) -> list[float] | None:
        if self._disk_path is None:
            return None
        try:
            with shelve.open(str(self._disk_path), flag="r") as db:
                return db.get(key)
        except Exception:
            return None

    def _disk_put(self, key: str, vec: list[float]) -> None:
        if self._disk_path is None:
            return
        try:
            with shelve.open(str(self._disk_path), flag="c") as db:
                db[key] = vec
        except Exception as exc:
            log.warning("Disk cache write failed", extra={"error": str(exc)})

    def warm_up(self) -> None:
        """Haystack lifecycle hook — validates Ollama connectivity."""
        try:
            self._call_ollama(["warmup probe"])
            log.info("CachedOllamaEmbedder warm_up: Ollama reachable", extra={"model": self.model})
        except Exception as exc:
            log.warning(
                "CachedOllamaEmbedder warm_up: Ollama unreachable (non-fatal)",
                extra={"error": str(exc)},
            )

    def cache_stats(self) -> dict[str, Any]:
        """Return current cache utilisation stats."""
        hits = self._metrics.get_counter("embedder.cache_hit")
        misses = self._metrics.get_counter("embedder.cache_miss")
        total = hits + misses
        return {
            "l1_size": len(self._l1),
            "cache_hits": hits,
            "cache_misses": misses,
            "hit_rate": round(hits / total, 4) if total else 0.0,
        }
