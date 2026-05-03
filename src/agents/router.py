"""
RAG3 Three-Tier Intent Router
==============================
Decides *whether* retrieval is needed and, if so, which worker agent
should handle the query:

    GeneralChat       — small talk, meta, greetings → General agent
    VectorRetrieval   — factual lookup → Vector agent
    GraphRetrieval    — relational multi-hop → Graph agent
    HybridRetrieval   — mixed → Graph+Vector fusion via Vector agent

Tiers:
    1. Regex   — instant, free, catches obvious patterns
    2. Keyword — scored hints, skips LLM if confidence high
    3. LLM     — arbiter fallback for ambiguous inputs
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar

from src.agents.lc_models import RouterOutput, make_router_chain
from src.config import get_settings
from src.monitoring.logger import get_logger
from src.utils.llm import chat_json

log = get_logger(__name__)


class Intent(str, Enum):
    GENERAL = "general_chat"
    VECTOR = "vector_retrieval"
    GRAPH = "graph_retrieval"
    HYBRID = "hybrid_retrieval"


@dataclass(frozen=True)
class RouterDecision:
    intent: Intent
    confidence: float
    tier: str  # "regex" | "keyword" | "llm"
    reason: str = ""


# ---------------------------------------------------------------------------
# Tier 1 — regex
# ---------------------------------------------------------------------------

_GREETING_RE = re.compile(
    r"^\s*(hi|hello|hey|yo|thanks|thank you|bye|ok|okay|cool|lol|sup)\s*[\.!\?]*\s*$",
    re.IGNORECASE,
)
_SMALLTALK_RE = re.compile(
    r"\b(how\s+are\s+you|what(?:'s| is)\s+up|tell\s+me\s+a\s+joke)\b",
    re.IGNORECASE,
)

_GRAPH_RE = re.compile(
    r"\b(relationship|connection|how\s+does\s+\w+\s+(?:relate|connect|affect|influence)|"
    r"linked\s+to|associated\s+with|network\s+of|paths?\s+between|"
    r"multi[-\s]?hop|causal)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Tier 2 — keyword scoring
# ---------------------------------------------------------------------------

_VECTOR_KEYWORDS = {
    "what", "define", "explain", "describe", "when", "who", "where",
    "list", "show", "give", "summary", "overview", "definition",
}
_GRAPH_KEYWORDS = {
    "related", "relation", "linked", "network", "connects", "graph",
    "between", "relationship", "associated", "traversal",
}
_HYBRID_KEYWORDS = {
    "compare", "contrast", "versus", "vs", "impact", "effect", "influence",
}


def _score_keywords(query: str, vocab: set[str]) -> float:
    tokens = re.findall(r"[a-z]+", query.lower())
    if not tokens:
        return 0.0
    hits = sum(1 for t in tokens if t in vocab)
    return hits / max(len(tokens), 1)


# ---------------------------------------------------------------------------
# Tier 3 — LLM arbiter
# ---------------------------------------------------------------------------

_LLM_SYSTEM = (
    "Classify the user query for a RAG system as one of:\n"
    '  "general_chat" — greeting, small talk, meta\n'
    '  "vector_retrieval" — factual lookup\n'
    '  "graph_retrieval" — relational / multi-hop\n'
    '  "hybrid_retrieval" — mix of lookup + relations\n'
    'Return JSON: {"intent": "...", "confidence": 0.0-1.0, "reason": "short"}'
)


class IntentRouter:
    """Three-tier intent router. Tier 3 uses LangChain ChatPromptTemplate + PydanticOutputParser."""

    def __init__(self) -> None:
        cfg = get_settings().router
        self._kw_threshold: float = cfg.keyword_confidence_threshold
        self._lc_chain = make_router_chain()  # None if langchain-core not installed

    def route(self, query: str) -> RouterDecision:
        q = (query or "").strip()
        if not q:
            return RouterDecision(Intent.GENERAL, 1.0, "regex", "empty input")

        # Tier 1
        if _GREETING_RE.match(q) or _SMALLTALK_RE.search(q):
            return RouterDecision(Intent.GENERAL, 0.95, "regex", "greeting/smalltalk")
        if _GRAPH_RE.search(q):
            return RouterDecision(Intent.GRAPH, 0.85, "regex", "graph pattern match")

        # Tier 2
        v = _score_keywords(q, _VECTOR_KEYWORDS)
        g = _score_keywords(q, _GRAPH_KEYWORDS)
        h = _score_keywords(q, _HYBRID_KEYWORDS)
        best = max(v, g, h)
        if best >= self._kw_threshold:
            if h == best:
                return RouterDecision(Intent.HYBRID, h, "keyword", "hybrid keywords")
            if g == best:
                return RouterDecision(Intent.GRAPH, g, "keyword", "graph keywords")
            return RouterDecision(Intent.VECTOR, v, "keyword", "vector keywords")

        # Tier 3 — LangChain ChatPromptTemplate + PydanticOutputParser
        if self._lc_chain is not None:
            try:
                out: RouterOutput = self._lc_chain.invoke({"query": q})
                intent = {
                    "general_chat": Intent.GENERAL,
                    "vector_retrieval": Intent.VECTOR,
                    "graph_retrieval": Intent.GRAPH,
                    "hybrid_retrieval": Intent.HYBRID,
                }.get(out.intent, Intent.VECTOR)
                return RouterDecision(intent, out.confidence, "llm_lc", out.reason)
            except Exception as exc:
                log.warning("LangChain router chain failed, falling back to chat_json",
                            extra={"error": str(exc)[:120]})

        # Fallback: raw JSON call
        raw = chat_json(
            _LLM_SYSTEM, q, fast=True, temperature=0.0, max_tokens=96, default={}
        )
        intent_raw = str((raw or {}).get("intent", "")).lower()
        intent = {
            "general_chat": Intent.GENERAL,
            "vector_retrieval": Intent.VECTOR,
            "graph_retrieval": Intent.GRAPH,
            "hybrid_retrieval": Intent.HYBRID,
        }.get(intent_raw, Intent.VECTOR)
        try:
            conf = float((raw or {}).get("confidence", 0.5))
        except (TypeError, ValueError):
            conf = 0.5
        reason = str((raw or {}).get("reason", ""))[:140]
        return RouterDecision(intent, max(0.0, min(1.0, conf)), "llm", reason)
