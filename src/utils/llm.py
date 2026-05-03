"""
RAG3 LLM Helper Pipeline
=========================
Tiny synchronous facade around ``RotatableGroqGenerator`` that the
retrieval strategies and agents call when they need a one-shot chat
completion (routing, grading, query expansion, summarisation).

Design
------
* A single shared ``chat_sync()`` function — avoids every caller
  instantiating its own generator.
* Per-model memoised generators via ``_get_generator()`` (thread-safe).
* ``chat_json()`` convenience: strips markdown fences and parses JSON.

Usage::

    from src.utils.llm import chat_sync, chat_json
    reply = chat_sync("You are a router.", "Classify: 'hi there'", fast=True)
    data = chat_json("Return JSON.", "{\"x\": 1}", fast=True)
"""

from __future__ import annotations

import json
import re
from threading import Lock
from typing import Any

from haystack.dataclasses import ChatMessage

from src.config import get_settings
from src.monitoring.logger import get_logger
from src.utils.groq_client import RotatableGroqGenerator

log = get_logger(__name__)

_GENERATORS: dict[str, RotatableGroqGenerator] = {}
_GEN_LOCK = Lock()


def _get_generator(model: str) -> RotatableGroqGenerator:
    """Return a cached ``RotatableGroqGenerator`` for the given model."""
    with _GEN_LOCK:
        gen = _GENERATORS.get(model)
        if gen is None:
            gen = RotatableGroqGenerator(model=model)
            _GENERATORS[model] = gen
        return gen


def _try_groq(system: str, user: str, fast: bool, model: str | None,
              temperature: float | None, max_tokens: int | None) -> str:
    cfg = get_settings().groq
    chosen = model or (cfg.fast_model if fast else cfg.primary_model)
    gen = _get_generator(chosen)
    overrides: dict[str, Any] = {}
    if temperature is not None:
        overrides["temperature"] = temperature
    if max_tokens is not None:
        overrides["max_tokens"] = max_tokens
    result = gen.run(
        messages=[
            ChatMessage.from_system(system),
            ChatMessage.from_user(user),
        ],
        generation_kwargs=overrides or None,
    )
    replies = result.get("replies", [])
    if not replies:
        return ""
    return (replies[0].content or "").strip()


def _try_openrouter(system: str, user: str, fast: bool, model: str | None,
                    temperature: float | None, max_tokens: int | None) -> str:
    from src.utils.openrouter_client import OpenRouterClient
    return OpenRouterClient().chat(
        system, user,
        model=model, fast=fast,
        temperature=0.1 if temperature is None else temperature,
        max_tokens=1024 if max_tokens is None else max_tokens,
    )


def _try_ollama(system: str, user: str, fast: bool, model: str | None,
                temperature: float | None, max_tokens: int | None) -> str:
    import httpx
    cfg = get_settings().ollama
    url = f"{str(cfg.base_url).rstrip('/')}/api/chat"
    payload = {
        "model": model or cfg.chat_model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "options": {
            "temperature": 0.1 if temperature is None else temperature,
            "num_predict": 1024 if max_tokens is None else max_tokens,
        },
        "stream": False,
    }
    with httpx.Client(timeout=cfg.timeout) as client:
        resp = client.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
    return (data.get("message", {}).get("content") or "").strip()


_PROVIDERS = {
    "groq": _try_groq,
    "openrouter": _try_openrouter,
    "ollama": _try_ollama,
}


def chat_sync(
    system: str,
    user: str,
    *,
    fast: bool = False,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> str:
    """
    Execute a single chat turn through the configured provider chain.

    Walks ``settings.llm_fallback_chain`` (default: groq → openrouter →
    ollama). First provider to return a non-empty reply wins.
    """
    chain = get_settings().llm_fallback_chain or ["groq"]
    last_error: Exception | None = None
    for provider in chain:
        fn = _PROVIDERS.get(provider)
        if fn is None:
            continue
        try:
            out = fn(system, user, fast, model, temperature, max_tokens)
            if out:
                return out
        except Exception as exc:
            last_error = exc
            log.warning(
                "LLM provider failed, trying next",
                extra={"provider": provider, "error": str(exc)[:200]},
            )
    if last_error is not None:
        log.error("All LLM providers failed", extra={"error": str(last_error)[:200]})
    return ""


_JSON_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)


def chat_json(
    system: str,
    user: str,
    *,
    fast: bool = True,
    model: str | None = None,
    temperature: float | None = 0.0,
    max_tokens: int | None = 512,
    default: Any = None,
) -> Any:
    """
    Run a chat turn expected to return JSON; parse and return the object.

    Strips optional ```json / ``` markdown fences that Groq often
    emits, then attempts ``json.loads``.  Returns ``default`` on failure.
    """
    raw = chat_sync(
        system,
        user,
        fast=fast,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if not raw:
        return default

    cleaned = _JSON_FENCE.sub("", raw).strip()
    # Fall back: extract first {...} or [...] blob
    if not (cleaned.startswith("{") or cleaned.startswith("[")):
        match = re.search(r"(\{.*\}|\[.*\])", cleaned, re.DOTALL)
        if match:
            cleaned = match.group(1)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        log.warning(
            "chat_json: failed to parse JSON reply",
            extra={"error": str(exc), "raw_preview": raw[:200]},
        )
        return default
