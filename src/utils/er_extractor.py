"""
RAG Entity-Relation Extractor
================================
Extracts ``(subject, relation, object)`` triples from free text for the
graph store. Uses a dedicated **gpt-oss-20b** model family with its own
fallback chain, independent from the general-purpose chat chain:

    Ollama  ``gpt-oss:20b``          (primary — local, confidential)
       ▼ failure
    Groq    ``openai/gpt-oss-20b``   (multi-key rotation via RotatableGroqGenerator)
       ▼ failure
    OpenRouter ``openai/gpt-oss-20b`` (multi-key rotation)

Triples come back as JSON; malformed replies are recovered where
possible (fence stripping + object/array extraction).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

import httpx

from src.config import get_settings
from src.monitoring.logger import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class Triple:
    subject: str
    relation: str
    object: str

    def as_dict(self) -> dict[str, str]:
        return {"subject": self.subject, "relation": self.relation, "object": self.object}


_SYSTEM_PROMPT = (
    "You extract knowledge-graph triples from text. For the passage the "
    "user provides, output JSON of the form:\n"
    '{"triples": [{"subject": "...", "relation": "...", "object": "..."}]}\n'
    "Rules:\n"
    " - Subjects and objects are specific entities, concepts, or terms.\n"
    " - Relations are short verb phrases (<= 4 words), lowercase.\n"
    " - Do NOT invent facts not present in the text.\n"
    " - Skip pronouns and vague references.\n"
    " - Limit to the most salient triples.\n"
    "Return ONLY the JSON object, no prose."
)


_JSON_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)


def _parse_triples(raw: str, limit: int) -> list[Triple]:
    if not raw:
        return []
    cleaned = _JSON_FENCE.sub("", raw).strip()
    if not (cleaned.startswith("{") or cleaned.startswith("[")):
        m = re.search(r"(\{.*\}|\[.*\])", cleaned, re.DOTALL)
        if m:
            cleaned = m.group(1)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        log.debug("ER extractor: malformed JSON", extra={"preview": raw[:200]})
        return []

    items: list[Any]
    if isinstance(data, dict):
        items = data.get("triples") or data.get("data") or []
    elif isinstance(data, list):
        items = data
    else:
        items = []

    out: list[Triple] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        s = str(it.get("subject") or it.get("head") or "").strip()
        r = str(it.get("relation") or it.get("predicate") or it.get("rel") or "").strip().lower()
        o = str(it.get("object") or it.get("tail") or "").strip()
        if s and r and o:
            out.append(Triple(subject=s, relation=r, object=o))
        if len(out) >= limit:
            break
    return out


# ---------------------------------------------------------------------------
# Provider calls
# ---------------------------------------------------------------------------


def _call_ollama(text: str) -> str:
    cfg = get_settings()
    url = f"{str(cfg.ollama.base_url).rstrip('/')}/api/chat"
    payload = {
        "model": cfg.er.ollama_model,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        "options": {
            "temperature": cfg.er.temperature,
            "num_predict": cfg.er.max_tokens,
        },
        "format": "json",
        "stream": False,
    }
    with httpx.Client(timeout=cfg.ollama.timeout) as c:
        r = c.post(url, json=payload)
        r.raise_for_status()
        data = r.json()
    return (data.get("message", {}).get("content") or "").strip()


def _call_groq(text: str) -> str:
    from haystack.dataclasses import ChatMessage
    from src.utils.groq_client import RotatableGroqGenerator

    cfg = get_settings()
    gen = RotatableGroqGenerator(model=cfg.er.groq_model)
    result = gen.run(
        messages=[
            ChatMessage.from_system(_SYSTEM_PROMPT),
            ChatMessage.from_user(text),
        ],
        generation_kwargs={
            "temperature": cfg.er.temperature,
            "max_tokens": cfg.er.max_tokens,
        },
    )
    replies = result.get("replies") or []
    return (replies[0].text or "").strip() if replies else ""


def _call_openrouter(text: str) -> str:
    from src.utils.openrouter_client import OpenRouterClient

    cfg = get_settings()
    return OpenRouterClient().chat(
        _SYSTEM_PROMPT, text,
        model=cfg.er.openrouter_model,
        temperature=cfg.er.temperature,
        max_tokens=cfg.er.max_tokens,
    )


_PROVIDERS = {
    "ollama": _call_ollama,
    "groq": _call_groq,
    "openrouter": _call_openrouter,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_triples(text: str, *, limit: int | None = None) -> list[Triple]:
    """
    Extract triples via the configured fallback chain.

    The first provider that returns parseable triples wins. If all
    providers fail (or return empty), returns an empty list — ingestion
    can still proceed with a keyword-only graph.
    """
    cfg = get_settings().er
    text = (text or "").strip()
    if not text:
        return []

    lim = limit or cfg.max_triples_per_episode
    last_err: Exception | None = None

    for provider in cfg.fallback_chain:
        fn = _PROVIDERS.get(provider)
        if fn is None:
            continue
        try:
            raw = fn(text)
        except Exception as exc:
            last_err = exc
            log.warning(
                "ER provider failed, trying next",
                extra={"provider": provider, "error": str(exc)[:200]},
            )
            continue
        triples = _parse_triples(raw, lim)
        if triples:
            log.debug(
                "ER extraction succeeded",
                extra={"provider": provider, "count": len(triples)},
            )
            return triples

    if last_err is not None:
        log.error("All ER providers failed", extra={"error": str(last_err)[:200]})
    return []
