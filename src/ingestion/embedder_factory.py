"""
RAG Embedder Factory
======================
Returns the correct embedder component (`CachedGeminiEmbedder` or `CachedOllamaEmbedder`)
based on `settings.embedding_backend` or preferred CLI parameter.
"""

from __future__ import annotations

from typing import Any

from src.config import get_settings
from src.monitoring.logger import get_logger

log = get_logger(__name__)


def build_embedder(preferred: str | None = None) -> Any:
    """
    Instantiate the embedder component.

    Args:
        preferred: Optional override backend ('gemini' or 'ollama').
    """
    cfg = get_settings()
    choice = (preferred or cfg.embedding_backend).lower()

    if choice == "gemini":
        try:
            from src.ingestion.gemini_embedder import CachedGeminiEmbedder
            log.info("Using Gemini Embedder", extra={"backend": "gemini"})
            return CachedGeminiEmbedder()
        except Exception as exc:
            log.warning("Gemini embedder failed to load, falling back to Ollama embedder", extra={"error": str(exc)})

    # Default fallback
    from src.ingestion.embedder import CachedOllamaEmbedder
    log.info("Using Ollama Embedder", extra={"backend": "ollama"})
    return CachedOllamaEmbedder()
