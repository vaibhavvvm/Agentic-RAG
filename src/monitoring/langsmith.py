"""
LangSmith observability integration.
Sets the four env-vars LangGraph reads automatically to enable tracing.
"""

from __future__ import annotations

import os

from src.config import get_settings
from src.monitoring.logger import get_logger

log = get_logger(__name__)


def setup_langsmith() -> bool:
    """Configure LangSmith env vars. Returns True when tracing is active."""
    cfg = get_settings().langsmith
    if not cfg.tracing_enabled:
        return False
    key = cfg.api_key.get_secret_value() if cfg.api_key else None
    if not key:
        log.warning("LANGSMITH_TRACING_ENABLED=true but LANGSMITH_API_KEY not set — skipping")
        return False

    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_API_KEY", key)
    os.environ.setdefault("LANGCHAIN_PROJECT", cfg.project)
    os.environ.setdefault("LANGCHAIN_ENDPOINT", cfg.endpoint)
    log.info("LangSmith tracing active", extra={"project": cfg.project, "endpoint": cfg.endpoint})
    return True


def get_project_url() -> str | None:
    """Return the LangSmith project URL — embed this in your README."""
    cfg = get_settings().langsmith
    if cfg.tracing_enabled and cfg.api_key:
        return f"https://smith.langchain.com/projects/{cfg.project}"
    return None
