"""
RAG Gemini Client Pipeline
============================
Thread-safe client around the Google GenAI SDK (`google-genai`).
Provides synchronous and asynchronous chat invocation for Gemini models
(gemini-3.5-flash, gemini-3.7-flash thinking model, etc.).
"""

from __future__ import annotations

import os
from threading import Lock
from typing import Any

from src.config import get_settings
from src.monitoring.logger import get_logger
from src.monitoring.metrics import MetricsCollector

log = get_logger(__name__)

try:
    from google import genai
    from google.genai import types
    _GEMINI_SDK_AVAILABLE = True
except ImportError:
    genai = None  # type: ignore
    types = None  # type: ignore
    _GEMINI_SDK_AVAILABLE = False
    log.warning("google-genai SDK not installed — GeminiClient unavailable")


class GeminiClient:
    """
    Singleton-style thread-safe client for Google Gemini models.
    """

    _client_instance: Any = None
    _lock: Lock = Lock()

    def __init__(self) -> None:
        if not _GEMINI_SDK_AVAILABLE:
            raise RuntimeError("google-genai SDK is not installed. Install with `pip install google-genai`.")

        cfg = get_settings().gemini
        api_key = cfg.api_key.get_secret_value() if cfg.api_key else os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not configured in settings or environment.")

        with self._lock:
            if GeminiClient._client_instance is None:
                GeminiClient._client_instance = genai.Client(api_key=api_key)

        self.client = GeminiClient._client_instance
        self.cfg = cfg
        self._metrics = MetricsCollector.get_instance()

    def chat(
        self,
        system: str,
        user: str,
        *,
        model: str | None = None,
        fast: bool = False,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """
        Execute a chat turn with Gemini models.
        """
        chosen_model = model or (self.cfg.fast_model if fast else self.cfg.primary_model)
        temp = temperature if temperature is not None else self.cfg.temperature
        max_t = max_tokens or self.cfg.max_tokens

        config = types.GenerateContentConfig(
            system_instruction=system if system else None,
            temperature=temp,
            max_output_tokens=max_t,
        )

        log.debug("Gemini chat invocation", extra={"model": chosen_model})
        with self._metrics.measure(f"gemini.{chosen_model}.completion"):
            try:
                response = self.client.models.generate_content(
                    model=chosen_model,
                    contents=user,
                    config=config,
                )
                self._metrics.record_event("gemini.success")
                return (response.text or "").strip()
            except Exception as exc:
                self._metrics.record_event("gemini.error")
                log.error("Gemini API call failed", extra={"error": str(exc), "model": chosen_model})
                raise
