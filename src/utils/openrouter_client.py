"""
RAG3 OpenRouter Client (Rotatable)
====================================
Httpx-based OpenAI-compatible chat client with **multi-key rotation**:
cycles through the pool when any one key hits 429 / 402 (rate limit or
credit exhaustion). Mirrors the rotation semantics of
``RotatableGroqGenerator`` so either provider can be used as a drop-in
fallback.
"""

from __future__ import annotations

import itertools
import random
import time
from threading import Lock
from typing import Any

import httpx

from src.config import get_settings
from src.monitoring.logger import get_logger

log = get_logger(__name__)


class AllKeysExhaustedException(RuntimeError):
    """Raised when every OpenRouter key has failed."""


class OpenRouterClient:
    """OpenAI-compatible OpenRouter client with key rotation."""

    def __init__(self) -> None:
        cfg = get_settings().openrouter
        self._cfg = cfg
        keys = cfg.api_keys
        if not keys:
            raise RuntimeError(
                "No OpenRouter keys configured (set OPENROUTER_API_KEY or "
                "OPENROUTER_API_KEYS=k1,k2)."
            )
        self._keys: list[str] = keys
        self._cycle = itertools.cycle(range(len(keys)))
        self._idx: int = next(self._cycle)
        self._lock = Lock()

    # ------------------------------------------------------------------

    def _headers(self, key: str) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "HTTP-Referer": self._cfg.referer,
            "X-Title": self._cfg.app_title,
        }

    def _rotate(self) -> int:
        with self._lock:
            self._idx = next(self._cycle)
            return self._idx

    def chat(
        self,
        system: str,
        user: str,
        *,
        model: str | None = None,
        fast: bool = False,
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> str:
        chosen = model or (self._cfg.fast_model if fast else self._cfg.primary_model)
        payload: dict[str, Any] = {
            "model": chosen,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        url = f"{self._cfg.base_url.rstrip('/')}/chat/completions"

        n_keys = len(self._keys)
        attempts = n_keys * max(1, self._cfg.max_retries_per_key)
        last_exc: Exception | None = None

        for attempt in range(attempts):
            key = self._keys[self._idx]
            try:
                with httpx.Client(timeout=self._cfg.request_timeout) as client:
                    resp = client.post(url, json=payload, headers=self._headers(key))
                if resp.status_code in (429, 402, 401):
                    log.warning(
                        "OpenRouter key exhausted — rotating",
                        extra={"status": resp.status_code, "key_index": self._idx},
                    )
                    self._rotate()
                    time.sleep(0.5 + random.random())
                    continue
                resp.raise_for_status()
                data = resp.json()
                choices = data.get("choices") or []
                if not choices:
                    return ""
                return (choices[0].get("message", {}).get("content") or "").strip()
            except httpx.HTTPStatusError as exc:
                last_exc = exc
                self._rotate()
            except Exception as exc:
                last_exc = exc
                # Network error: back off briefly, try next key
                self._rotate()
                time.sleep(min(2.0, 0.5 * (attempt + 1)))

        raise AllKeysExhaustedException(
            f"All {n_keys} OpenRouter key(s) failed; last error: {last_exc}"
        )
