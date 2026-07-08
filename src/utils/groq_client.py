"""
RAG RotatableGroqGenerator
============================
A Haystack 2.x-compatible ``ChatGenerator`` component that wraps the
Groq API with:

* **Key rotation** — cycles through a pool of API keys when any one hits
  its rate limit (HTTP 429) or quota.
* **Exponential back-off with jitter** — avoids thundering-herd on
  shared rate-limit windows.
* **Metrics integration** — records latency, key-rotation events, and
  error counts via ``MetricsCollector``.
* **Full Haystack 2.x interface** — implements ``@component`` with
  ``run()`` returning ``{"replies": list[ChatMessage]}``.

Haystack 2.x contract:
    * Decorated with ``@component``.
    * ``run()`` accepts ``messages: list[ChatMessage]`` and optional
      ``generation_kwargs`` overrides.
    * Returns ``{"replies": list[ChatMessage]}``.

Usage::

    from haystack.dataclasses import ChatMessage
    from src.utils.groq_client import RotatableGroqGenerator

    gen = RotatableGroqGenerator(model="llama3-70b-8192")
    result = gen.run(messages=[ChatMessage.from_user("Hello")])
    print(result["replies"][0].text)
"""

from __future__ import annotations

import itertools
import logging
import random
import time
from threading import Lock
from typing import Any, ClassVar

from groq import APIStatusError, AuthenticationError, RateLimitError
from groq import Groq as GroqClient
from haystack import component
from haystack.dataclasses import ChatMessage, ChatRole

from src.config import get_settings
from src.monitoring.logger import get_logger
from src.monitoring.metrics import MetricsCollector

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class AllKeysExhaustedException(RuntimeError):
    """Raised when every API key in the rotation pool has failed."""


class GroqGenerationError(RuntimeError):
    """Raised when the Groq API returns an unrecoverable error."""


# ---------------------------------------------------------------------------
# Internal key-pool manager
# ---------------------------------------------------------------------------


class _KeyPool:
    """
    Thread-safe round-robin pool of Groq API keys.

    Each key tracks its own back-off state independently so a key that
    hit a rate-limit can re-enter rotation after its cool-down expires.

    Attributes:
        _keys:     Ordered list of API key strings.
        _cooldown: Per-key UNIX timestamp after which the key may be reused.
        _cycle:    Infinite round-robin iterator over key indices.
        _lock:     Mutex protecting all mutable state.
    """

    def __init__(self, keys: list[str]) -> None:
        if not keys:
            raise ValueError("KeyPool requires at least one API key.")
        self._keys: list[str] = list(keys)
        self._cooldown: dict[str, float] = {k: 0.0 for k in self._keys}
        self._cycle = itertools.cycle(range(len(self._keys)))
        self._lock = Lock()
        self._current_idx: int = 0

    def acquire(self) -> str:
        """
        Return the next available (not in cool-down) API key.

        Raises:
            AllKeysExhaustedException: If every key is currently in cool-down.
        """
        with self._lock:
            now = time.monotonic()
            for _ in range(len(self._keys)):
                idx = next(self._cycle)
                key = self._keys[idx]
                if self._cooldown[key] <= now:
                    return key
            raise AllKeysExhaustedException(
                f"All {len(self._keys)} Groq API keys are currently rate-limited."
            )

    def penalise(self, key: str, delay_seconds: float) -> None:
        """
        Put ``key`` in cool-down for ``delay_seconds`` seconds.

        Args:
            key:           The API key string to penalise.
            delay_seconds: Duration of the cool-down window.
        """
        with self._lock:
            if key in self._cooldown:
                self._cooldown[key] = time.monotonic() + delay_seconds
                log.warning(
                    "Groq key penalised",
                    extra={
                        "key_suffix": key[-4:],
                        "cooldown_seconds": delay_seconds,
                    },
                )

    def size(self) -> int:
        return len(self._keys)


# ---------------------------------------------------------------------------
# Haystack 2.x Component
# ---------------------------------------------------------------------------


@component
class RotatableGroqGenerator:
    """
    Haystack 2.x ChatGenerator backed by Groq with key rotation.

    Args:
        model:           Groq model identifier (overrides ``groq.primary_model``
                         from settings if provided).
        api_keys:        Explicit list of API keys.  If ``None``, keys are
                         loaded from ``src.config.GroqSettings``.
        temperature:     Sampling temperature (0.0 – 2.0).
        max_tokens:      Maximum completion tokens.
        top_p:           Nucleus sampling probability.
        max_retries:     Maximum number of retry attempts across all keys.
        retry_base_delay: Base delay (seconds) for exponential back-off.
        retry_max_delay:  Upper cap on back-off delay (seconds).
        generation_kwargs: Additional kwargs forwarded to the Groq SDK on
                           every call (can be overridden per-run).

    Output::

        {"replies": list[ChatMessage]}
    """

    # Haystack requires output types declared at class level
    OUTPUT_TYPES: ClassVar[dict[str, type]] = {"replies": list}

    def __init__(
        self,
        model: str | None = None,
        api_keys: list[str] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        max_retries: int | None = None,
        retry_base_delay: float | None = None,
        retry_max_delay: float | None = None,
        generation_kwargs: dict[str, Any] | None = None,
    ) -> None:
        cfg = get_settings().groq

        self.model: str = model or cfg.primary_model
        self.temperature: float = temperature if temperature is not None else cfg.temperature
        self.max_tokens: int = max_tokens or cfg.max_tokens
        self.top_p: float = top_p if top_p is not None else cfg.top_p
        self.max_retries: int = max_retries or cfg.max_retries
        self.retry_base_delay: float = retry_base_delay or cfg.retry_base_delay
        self.retry_max_delay: float = retry_max_delay or cfg.retry_max_delay
        self.generation_kwargs: dict[str, Any] = generation_kwargs or {}

        # Resolve API keys
        resolved_keys = api_keys or cfg.api_keys
        if not resolved_keys:
            raise ValueError(
                "RotatableGroqGenerator: no API keys provided. "
                "Set GROQ_API_KEY or GROQ_API_KEYS in your environment."
            )
        self._pool = _KeyPool(resolved_keys)
        self._metrics = MetricsCollector.get_instance()

        log.info(
            "RotatableGroqGenerator initialised",
            extra={"model": self.model, "key_pool_size": self._pool.size()},
        )

    # ------------------------------------------------------------------
    # Haystack 2.x run() method
    # ------------------------------------------------------------------

    @component.output_types(replies=list)
    def run(
        self,
        messages: list[ChatMessage],
        generation_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, list[ChatMessage]]:
        """
        Execute a chat completion request against the Groq API.

        Args:
            messages:          Haystack ``ChatMessage`` list (system + user turns).
            generation_kwargs: Per-call overrides for temperature, max_tokens, etc.

        Returns:
            ``{"replies": [ChatMessage]}`` — always a single assistant reply.

        Raises:
            AllKeysExhaustedException: If every key is rate-limited.
            GroqGenerationError:        On unrecoverable API errors.
        """
        merged_kwargs: dict[str, Any] = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            **self.generation_kwargs,
            **(generation_kwargs or {}),
        }

        groq_messages = self._to_groq_messages(messages)
        last_exc: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                key = self._pool.acquire()
            except AllKeysExhaustedException:
                raise

            client = GroqClient(api_key=key)

            try:
                with self._metrics.measure(f"groq.{self.model}.completion"):
                    response = client.chat.completions.create(
                        model=self.model,
                        messages=groq_messages,
                        **merged_kwargs,
                    )

                reply_text = response.choices[0].message.content or ""
                self._metrics.record_event("groq.success")

                log.debug(
                    "Groq completion received",
                    extra={
                        "model": self.model,
                        "attempt": attempt,
                        "prompt_tokens": response.usage.prompt_tokens
                        if response.usage
                        else None,
                        "completion_tokens": response.usage.completion_tokens
                        if response.usage
                        else None,
                    },
                )

                return {
                    "replies": [
                        ChatMessage.from_assistant(reply_text)
                    ]
                }

            except RateLimitError as exc:
                delay = self._backoff_delay(attempt)
                self._pool.penalise(key, delay)
                self._metrics.record_event("groq.rate_limit")
                log.warning(
                    "Groq rate limit hit; rotating key",
                    extra={
                        "attempt": attempt,
                        "key_suffix": key[-4:],
                        "retry_in_seconds": delay,
                    },
                )
                last_exc = exc
                time.sleep(delay)

            except AuthenticationError as exc:
                # Invalid key — penalise permanently for this run
                self._pool.penalise(key, self.retry_max_delay * 100)
                self._metrics.record_event("groq.auth_error")
                log.error(
                    "Groq authentication error",
                    extra={"key_suffix": key[-4:]},
                )
                last_exc = exc

            except APIStatusError as exc:
                # Server-side errors (5xx) — short back-off, same key is fine
                delay = self._backoff_delay(attempt)
                self._metrics.record_event("groq.server_error")
                log.warning(
                    "Groq server error",
                    extra={
                        "status_code": exc.status_code,
                        "attempt": attempt,
                        "retry_in_seconds": delay,
                        "groq_error": str(exc),
                    },
                )
                last_exc = exc
                time.sleep(delay)

            except Exception as exc:
                self._metrics.record_event("groq.unknown_error")
                log.error("Unexpected Groq error", exc_info=True)
                last_exc = exc
                break

        raise GroqGenerationError(
            f"Groq generation failed after {self.max_retries} attempts."
        ) from last_exc

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _backoff_delay(self, attempt: int) -> float:
        """
        Return jittered exponential back-off delay for the given attempt number.

        Formula: ``min(base * 2^(attempt-1) + jitter, max_delay)``
        Jitter is uniform in ``[0, base]`` to spread retries across callers.
        """
        exponential = self.retry_base_delay * (2 ** (attempt - 1))
        jitter = random.uniform(0, self.retry_base_delay)
        return min(exponential + jitter, self.retry_max_delay)

    @staticmethod
    def _to_groq_messages(
        messages: list[ChatMessage],
    ) -> list[dict[str, str]]:
        """
        Convert Haystack ``ChatMessage`` objects to Groq SDK message dicts.

        Haystack ``ChatRole`` → Groq role string mapping:
            SYSTEM    → "system"
            USER      → "user"
            ASSISTANT → "assistant"
        """
        role_map: dict[ChatRole, str] = {
            ChatRole.SYSTEM: "system",
            ChatRole.USER: "user",
            ChatRole.ASSISTANT: "assistant",
        }
        result: list[dict[str, str]] = []
        for msg in messages:
            role = role_map.get(msg.role, "user")
            result.append({"role": role, "content": msg.text or ""})
        return result

    def warm_up(self) -> None:
        """
        Haystack lifecycle hook — called before the first pipeline run.

        Validates that at least one API key is reachable by attempting a
        minimal completion.  Logs a warning rather than raising if the
        check fails, to allow offline / mock testing.
        """
        log.info(
            "RotatableGroqGenerator warm_up: validating key pool",
            extra={"pool_size": self._pool.size()},
        )
        try:
            test_msg = [ChatMessage.from_user("ping")]
            self.run(
                messages=test_msg,
                generation_kwargs={"max_tokens": 5, "temperature": 0.0},
            )
            log.info("RotatableGroqGenerator warm_up: key validated OK")
        except Exception as exc:
            log.warning(
                "RotatableGroqGenerator warm_up validation failed (non-fatal)",
                extra={"error": str(exc)},
            )
