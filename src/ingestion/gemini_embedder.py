"""
RAG Cached Gemini Text Embedder
=================================
Haystack 2.x `@component` wrapping Google Gemini embedding API (`gemini-embedding-001`) with:

* **Two-level cache** — L1 in-process LRU (dict) + L2 optional disk (`shelve`)
* **Flexible output dimensionality** — Defaults to 768 to align with system defaults.
* **Automatic normalisation** — L2-normalised vectors.
* **Metrics integration** — Records cache hits/misses and embedding latency.

Haystack contract:
    `run(texts: list[str]) -> {"embeddings": list[list[float]]}`
"""

from __future__ import annotations

import hashlib
import math
import os
import shelve
from collections import OrderedDict
from pathlib import Path
from threading import Lock
from typing import Any

from haystack import component

from src.config import get_settings
from src.monitoring.logger import get_logger
from src.monitoring.metrics import MetricsCollector

log = get_logger(__name__)

try:
    from google import genai
    from google.genai import types
    _GEMINI_AVAILABLE = True
except ImportError:
    genai = None  # type: ignore
    types = None  # type: ignore
    _GEMINI_AVAILABLE = False


class _LRUCache:
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
    payload = f"{model}::{text}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _l2_normalise(vec: list[float]) -> list[float]:
    magnitude = math.sqrt(sum(x * x for x in vec))
    if magnitude < 1e-9:
        return vec
    return [x / magnitude for x in vec]


@component
class CachedGeminiEmbedder:
    """
    Gemini text embedder with two-level caching.
    """

    def __init__(
        self,
        model: str | None = None,
        dimensions: int | None = None,
        batch_size: int = 32,
        l1_maxsize: int = 4096,
        disk_cache_path: Path | None = None,
        normalise: bool = True,
    ) -> None:
        if not _GEMINI_AVAILABLE:
            raise RuntimeError("google-genai SDK not installed. Run `pip install google-genai`.")

        cfg = get_settings().gemini
        api_key = cfg.api_key.get_secret_value() if cfg.api_key else os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is missing.")

        self.client = genai.Client(api_key=api_key)
        self.model = model or cfg.embedding_model
        self.dimensions = dimensions or cfg.embedding_dimensions
        self.batch_size = batch_size
        self.normalise = normalise

        self._l1 = _LRUCache(maxsize=l1_maxsize)
        self._disk_path: Path | None = None
        if disk_cache_path is not None:
            disk_cache_path.mkdir(parents=True, exist_ok=True)
            self._disk_path = disk_cache_path / "gemini_embed_cache"

        self._metrics = MetricsCollector.get_instance()
        log.info(
            "CachedGeminiEmbedder initialised",
            extra={
                "model": self.model,
                "dimensions": self.dimensions,
                "disk_cache": str(disk_cache_path) if disk_cache_path else "disabled",
            },
        )

    @component.output_types(embeddings=list)
    def run(self, texts: list[str]) -> dict[str, list[list[float]]]:
        if not texts:
            return {"embeddings": []}

        results: dict[int, list[float]] = {}
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

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

        if uncached_texts:
            new_vecs = self._embed_batched(uncached_texts)
            for idx, (orig_i, text, vec) in enumerate(
                zip(uncached_indices, uncached_texts, new_vecs, strict=False)
            ):
                key = _cache_key(self.model, text)
                self._l1.put(key, vec)
                self._disk_put(key, vec)
                results[orig_i] = vec

        ordered = [results[i] for i in range(len(texts))]
        return {"embeddings": ordered}

    def _embed_batched(self, texts: list[str]) -> list[list[float]]:
        all_vecs: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            with self._metrics.measure("embedder.gemini_call"):
                batch_vecs = self._call_gemini(batch)
            all_vecs.extend(batch_vecs)
        return all_vecs

    def _call_gemini(self, texts: list[str]) -> list[list[float]]:
        vecs: list[list[float]] = []
        for text in texts:
            config = types.EmbedContentConfig(
                output_dimensionality=self.dimensions
            )
            response = self.client.models.embed_content(
                model=self.model,
                contents=text,
                config=config,
            )
            # Response contains embedding values list
            raw_vec = list(response.embedding.values)
            if self.normalise:
                raw_vec = _l2_normalise(raw_vec)
            vecs.append(raw_vec)
        return vecs

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
