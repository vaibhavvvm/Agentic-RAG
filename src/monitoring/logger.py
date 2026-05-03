"""
RAG3 Structured JSON Logger
============================
Provides a module-level ``get_logger()`` factory that returns standard
``logging.Logger`` instances backed by a JSON formatter.

Features
--------
* Structured JSON output — every log record includes timestamp, level,
  logger name, message, and any extra keyword arguments passed at the
  call site.
* Optional human-readable ``text`` format for local development
  (controlled by ``MONITORING_LOG_FORMAT``).
* Optional file sink alongside stdout.
* ``request_id`` and ``session_id`` context-vars for correlated tracing.
* A ``timed_operation`` context manager that logs entry/exit with latency.

Usage::

    from src.monitoring.logger import get_logger

    log = get_logger(__name__)
    log.info("Document ingested", extra={"doc_id": "abc", "chunks": 12})
"""

from __future__ import annotations

import json
import logging
import sys
import time
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator, Literal

# ---------------------------------------------------------------------------
# Context variables — set once per request/session and automatically included
# in every log line within the same async/thread context.
# ---------------------------------------------------------------------------

_request_id_var: ContextVar[str] = ContextVar("request_id", default="")
_session_id_var: ContextVar[str] = ContextVar("session_id", default="")


def set_request_id(rid: str) -> None:
    """Bind a request ID to the current execution context."""
    _request_id_var.set(rid)


def set_session_id(sid: str) -> None:
    """Bind a session ID to the current execution context."""
    _session_id_var.set(sid)


def get_request_id() -> str:
    return _request_id_var.get()


def get_session_id() -> str:
    return _session_id_var.get()


# ---------------------------------------------------------------------------
# JSON Formatter
# ---------------------------------------------------------------------------


class _JSONFormatter(logging.Formatter):
    """
    Convert a ``LogRecord`` into a single-line JSON string.

    Standard fields emitted:
        timestamp   — ISO-8601 UTC
        level       — DEBUG / INFO / WARNING / ERROR / CRITICAL
        logger      — dotted module name
        message     — formatted message string
        request_id  — from context var (empty string if unset)
        session_id  — from context var (empty string if unset)

    Extra fields:
        Any key/value pairs passed via ``extra={}`` at the call site are
        merged into the top-level JSON object.  Built-in ``LogRecord``
        attributes are excluded to avoid noise.
    """

    # LogRecord attributes that should NOT appear as extra fields
    _SKIP_ATTRS: frozenset[str] = frozenset(
        {
            "name", "msg", "args", "levelname", "levelno", "pathname",
            "filename", "module", "exc_info", "exc_text", "stack_info",
            "lineno", "funcName", "created", "msecs", "relativeCreated",
            "thread", "threadName", "processName", "process", "message",
            "taskName",
        }
    )

    def format(self, record: logging.LogRecord) -> str:
        record.message = record.getMessage()
        if record.exc_info:
            record.exc_text = self.formatException(record.exc_info)

        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.message,
            "request_id": _request_id_var.get(),
            "session_id": _session_id_var.get(),
        }

        # Merge caller-supplied extra fields
        for key, value in record.__dict__.items():
            if key not in self._SKIP_ATTRS and not key.startswith("_"):
                payload[key] = value

        if record.exc_text:
            payload["exception"] = record.exc_text

        if record.stack_info:
            payload["stack_info"] = record.stack_info

        return json.dumps(payload, default=str, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Text Formatter (development convenience)
# ---------------------------------------------------------------------------


class _TextFormatter(logging.Formatter):
    """Simple coloured text formatter for local development."""

    _COLOURS: dict[str, str] = {
        "DEBUG": "\033[36m",     # cyan
        "INFO": "\033[32m",      # green
        "WARNING": "\033[33m",   # yellow
        "ERROR": "\033[31m",     # red
        "CRITICAL": "\033[35m",  # magenta
    }
    _RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        colour = self._COLOURS.get(record.levelname, "")
        ts = datetime.fromtimestamp(record.created, tz=timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%S.%f"
        )[:-3] + "Z"
        msg = record.getMessage()
        base = f"{colour}{ts} [{record.levelname:<8}] {record.name}: {msg}{self._RESET}"
        if record.exc_info:
            base += "\n" + self.formatException(record.exc_info)
        return base


# ---------------------------------------------------------------------------
# Root handler setup (called once)
# ---------------------------------------------------------------------------

_HANDLERS_INSTALLED = False


def _install_handlers(
    log_level: str,
    log_format: Literal["json", "text"],
    log_file: Path | None,
) -> None:
    """
    Configure the root logger exactly once per process.

    Idempotent — subsequent calls after the first are no-ops.
    """
    global _HANDLERS_INSTALLED
    if _HANDLERS_INSTALLED:
        return

    root = logging.getLogger()
    root.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    formatter: logging.Formatter = (
        _JSONFormatter() if log_format == "json" else _TextFormatter()
    )

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    root.addHandler(stdout_handler)

    if log_file is not None:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    # Silence noisy third-party loggers at WARNING
    for noisy in ("httpx", "httpcore", "urllib3", "asyncio", "haystack"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _HANDLERS_INSTALLED = True


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def get_logger(name: str) -> logging.Logger:
    """
    Return a ``logging.Logger`` configured for RAG3.

    On the first call this reads settings from ``src.config`` and installs
    the root handlers.  Subsequent calls with any ``name`` skip re-init.

    Args:
        name: Typically ``__name__`` of the calling module.

    Returns:
        A standard ``logging.Logger`` instance.
    """
    from src.config import get_settings  # lazy import to avoid circular deps

    cfg = get_settings().monitoring
    _install_handlers(cfg.log_level, cfg.log_format, cfg.log_file)
    return logging.getLogger(name)


# ---------------------------------------------------------------------------
# Utility: timed_operation context manager
# ---------------------------------------------------------------------------


@contextmanager
def timed_operation(
    operation: str,
    logger: logging.Logger | None = None,
    extra: dict[str, Any] | None = None,
) -> Generator[None, None, None]:
    """
    Context manager that logs entry and exit of a named operation.

    On entry  → DEBUG log with operation name.
    On exit   → INFO log with ``duration_ms`` field.
    On error  → ERROR log with exception details.

    Args:
        operation: Human-readable name for the operation being timed.
        logger:    Logger instance to use; defaults to module-level logger.
        extra:     Additional fields to include in both log lines.

    Example::

        with timed_operation("vector_search", log, extra={"query": q}):
            results = store.search(q)
    """
    _log = logger or get_logger(__name__)
    _extra: dict[str, Any] = extra or {}
    _log.debug("Starting operation", extra={"operation": operation, **_extra})
    t0 = time.perf_counter()
    try:
        yield
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        _log.error(
            "Operation failed",
            extra={
                "operation": operation,
                "duration_ms": round(elapsed_ms, 2),
                "error": str(exc),
                **_extra,
            },
            exc_info=True,
        )
        raise
    else:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        _log.info(
            "Operation completed",
            extra={
                "operation": operation,
                "duration_ms": round(elapsed_ms, 2),
                **_extra,
            },
        )
