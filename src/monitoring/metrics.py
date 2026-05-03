"""
RAG3 Metrics Collector
=======================
In-process latency and counter tracker with percentile support.

Design goals
------------
* Zero external dependencies — uses only the stdlib ``statistics`` module.
* Thread-safe via ``threading.Lock``; safe to call from multiple threads.
* Percentile computation (P50 / P90 / P99 configurable).
* Optional JSON snapshot export for CI or dashboarding.
* ``@track_latency`` decorator for transparent instrumentation.
* ``record_event`` for counters (errors, cache hits, etc.).

Usage::

    from src.monitoring.metrics import MetricsCollector

    metrics = MetricsCollector.get_instance()

    # As a context manager:
    with metrics.measure("vector_search"):
        results = store.search(query)

    # As a decorator:
    @metrics.track("rerank")
    def rerank(docs): ...

    # Snapshot:
    snap = metrics.snapshot()
    print(snap["latencies"]["vector_search"]["p99_ms"])
"""

from __future__ import annotations

import json
import statistics
import threading
import time
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Generator, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class LatencyWindow:
    """
    Sliding window of latency samples (milliseconds).

    Attributes:
        maxlen: Maximum samples to retain; oldest are evicted automatically.
        samples: Deque of float millisecond measurements.
    """

    maxlen: int = 1000
    samples: deque[float] = field(default_factory=lambda: deque(maxlen=1000))

    def record(self, ms: float) -> None:
        self.samples.append(ms)

    def percentile(self, p: float) -> float:
        """
        Return the ``p``-th percentile of current samples (0 < p ≤ 100).

        Returns 0.0 if no samples have been recorded yet.
        """
        if not self.samples:
            return 0.0
        sorted_samples = sorted(self.samples)
        idx = max(0, int(len(sorted_samples) * p / 100) - 1)
        return sorted_samples[min(idx, len(sorted_samples) - 1)]

    def mean(self) -> float:
        return statistics.mean(self.samples) if self.samples else 0.0

    def count(self) -> int:
        return len(self.samples)

    def summary(self, percentiles: list[float]) -> dict[str, float]:
        """
        Return a dict with mean_ms and one entry per percentile.

        Example with percentiles=[50.0, 90.0, 99.0]::

            {
                "count": 42,
                "mean_ms": 31.4,
                "p50_ms": 28.0,
                "p90_ms": 55.0,
                "p99_ms": 102.0,
            }
        """
        result: dict[str, float] = {
            "count": float(self.count()),
            "mean_ms": round(self.mean(), 3),
        }
        for p in percentiles:
            label = f"p{int(p)}_ms"
            result[label] = round(self.percentile(p), 3)
        return result


@dataclass
class Counter:
    """Simple monotonically increasing integer counter."""

    value: int = 0

    def increment(self, by: int = 1) -> None:
        self.value += by

    def reset(self) -> None:
        self.value = 0


# ---------------------------------------------------------------------------
# MetricsCollector
# ---------------------------------------------------------------------------


class MetricsCollector:
    """
    Singleton metrics collector for RAG3.

    All public methods are thread-safe.

    Attributes:
        percentiles: List of percentile values to compute (e.g. [50, 90, 99]).
        _lock: Threading lock protecting all mutable state.
        _latencies: Per-operation LatencyWindow instances.
        _counters: Named integer counters.
        _export_path: Optional Path to write JSON snapshots.
    """

    _instance: MetricsCollector | None = None
    _instance_lock: threading.Lock = threading.Lock()

    def __init__(
        self,
        percentiles: list[float] | None = None,
        export_path: Path | None = None,
        window_maxlen: int = 1000,
    ) -> None:
        self.percentiles: list[float] = percentiles or [50.0, 90.0, 99.0]
        self._export_path = export_path
        self._window_maxlen = window_maxlen
        self._lock = threading.Lock()
        self._latencies: dict[str, LatencyWindow] = defaultdict(
            lambda: LatencyWindow(maxlen=self._window_maxlen)
        )
        self._counters: dict[str, Counter] = defaultdict(Counter)
        self._start_time: datetime = datetime.now(tz=timezone.utc)

    # ------------------------------------------------------------------
    # Singleton access
    # ------------------------------------------------------------------

    @classmethod
    def get_instance(cls) -> "MetricsCollector":
        """
        Return the process-wide singleton, creating it on first call.

        Settings are read from ``src.config`` on first initialisation.
        """
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    from src.config import get_settings

                    cfg = get_settings().monitoring
                    cls._instance = cls(
                        percentiles=cfg.metrics_percentiles,
                        export_path=cfg.metrics_export_path,
                    )
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Destroy the singleton (useful in tests)."""
        with cls._instance_lock:
            cls._instance = None

    # ------------------------------------------------------------------
    # Latency recording
    # ------------------------------------------------------------------

    def record_latency(self, operation: str, duration_ms: float) -> None:
        """
        Record a single latency measurement for ``operation``.

        Args:
            operation:   Dotted name (e.g. ``"retrieval.vector_search"``).
            duration_ms: Elapsed time in milliseconds.
        """
        with self._lock:
            self._latencies[operation].record(duration_ms)

    @contextmanager
    def measure(self, operation: str) -> Generator[None, None, None]:
        """
        Context manager that times a block and records latency.

        Args:
            operation: Human-readable operation name.

        Example::

            with metrics.measure("retrieval.rerank"):
                docs = reranker.rank(query, docs)
        """
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            self.record_latency(operation, elapsed_ms)

    def track(self, operation: str) -> Callable[[F], F]:
        """
        Decorator factory that records latency for a synchronous function.

        Args:
            operation: Name to record latency under.

        Example::

            @metrics.track("ingestion.embed")
            def embed(text: str) -> list[float]:
                ...
        """
        def decorator(func: F) -> F:
            @wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                with self.measure(operation):
                    return func(*args, **kwargs)
            return wrapper  # type: ignore[return-value]
        return decorator

    # ------------------------------------------------------------------
    # Counter recording
    # ------------------------------------------------------------------

    def record_event(self, event: str, count: int = 1) -> None:
        """
        Increment a named counter.

        Args:
            event: Event name (e.g. ``"cache.hit"``, ``"groq.rate_limit"``).
            count: Amount to increment (default 1).
        """
        with self._lock:
            self._counters[event].increment(count)

    def get_counter(self, event: str) -> int:
        """Return current value of a named counter."""
        with self._lock:
            return self._counters[event].value

    # ------------------------------------------------------------------
    # Snapshots
    # ------------------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        """
        Return a full point-in-time snapshot of all metrics.

        Returns:
            Dictionary with the following top-level keys:

            * ``generated_at`` — ISO-8601 UTC timestamp
            * ``uptime_seconds`` — seconds since collector was created
            * ``latencies`` — per-operation summary dicts
            * ``counters`` — per-event integer counts
        """
        with self._lock:
            now = datetime.now(tz=timezone.utc)
            uptime = (now - self._start_time).total_seconds()

            latency_data: dict[str, dict[str, float]] = {
                op: window.summary(self.percentiles)
                for op, window in self._latencies.items()
            }
            counter_data: dict[str, int] = {
                ev: ctr.value for ev, ctr in self._counters.items()
            }

        return {
            "generated_at": now.isoformat(),
            "uptime_seconds": round(uptime, 2),
            "latencies": latency_data,
            "counters": counter_data,
        }

    def export_snapshot(self, path: Path | None = None) -> Path:
        """
        Write the current snapshot to a JSON file.

        Args:
            path: Override the default export path configured at init.
                  If neither is set, writes to ``metrics_snapshot.json``
                  in the current working directory.

        Returns:
            The ``Path`` of the written file.
        """
        target = path or self._export_path or Path("metrics_snapshot.json")
        target.parent.mkdir(parents=True, exist_ok=True)
        snap = self.snapshot()
        target.write_text(json.dumps(snap, indent=2), encoding="utf-8")
        return target

    def log_summary(self) -> None:
        """Emit the current snapshot to the structured logger at INFO level."""
        from src.monitoring.logger import get_logger

        log = get_logger(__name__)
        snap = self.snapshot()
        log.info(
            "Metrics snapshot",
            extra={
                "uptime_seconds": snap["uptime_seconds"],
                "latencies": snap["latencies"],
                "counters": snap["counters"],
            },
        )

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def get_p99(self, operation: str) -> float:
        """Return the P99 latency (ms) for ``operation``, or 0.0 if unseen."""
        with self._lock:
            window = self._latencies.get(operation)
            return window.percentile(99.0) if window else 0.0

    def get_mean(self, operation: str) -> float:
        """Return the mean latency (ms) for ``operation``, or 0.0 if unseen."""
        with self._lock:
            window = self._latencies.get(operation)
            return window.mean() if window else 0.0

    def list_operations(self) -> list[str]:
        """Return sorted list of all tracked operation names."""
        with self._lock:
            return sorted(self._latencies.keys())

    def list_events(self) -> list[str]:
        """Return sorted list of all tracked event/counter names."""
        with self._lock:
            return sorted(self._counters.keys())
