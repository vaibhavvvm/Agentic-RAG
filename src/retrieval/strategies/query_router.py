"""
RAG Query Router (Tier-1 regex inside retrieval)
====================================================
The retrieval-level router classifies *how* to search — vector vs. graph
vs. hybrid — given a query and (optionally) conversation context.

This is orthogonal to the top-level ``IntentRouter`` in ``src/agents``
which decides whether retrieval is needed at all.  Here we assume
retrieval is on; the question is only which index to hit.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import ClassVar, Literal

from haystack import component

from src.monitoring.logger import get_logger
from src.utils.llm import chat_json

log = get_logger(__name__)

RouteLabel = Literal["vector", "graph", "hybrid"]

# Keywords hinting at graph traversal
_GRAPH_HINTS = (
    r"\brelat(?:ed|ion|ionship)\b",
    r"\bconnect(?:ed|ion)\b",
    r"\bhow\s+(?:does|do)\s+\w+\s+(?:relate|connect|affect)",
    r"\bwho\s+(?:is|are)\b.*\b(?:linked|associated)\b",
    r"\bnetwork\b",
    r"\binfluenc(?:e|es|ed)\b",
    r"\bcaus(?:e|es|ed)\b",
)
_GRAPH_RE = re.compile("|".join(_GRAPH_HINTS), re.IGNORECASE)

SYSTEM_PROMPT = (
    "Classify a RAG query as one of: \"vector\", \"graph\", \"hybrid\".\n"
    "- vector: simple lookup / factual / definitional\n"
    "- graph:  relational, multi-hop, connection-seeking\n"
    "- hybrid: mixes definitional lookup with relationships\n"
    "Return JSON: {\"route\": \"...\", \"confidence\": 0.0–1.0}"
)


@dataclass(frozen=True)
class RouteDecision:
    route: RouteLabel
    confidence: float
    reason: str = ""


@component
class QueryRouter:
    """Regex-first, LLM-fallback router between vector/graph/hybrid stores."""

    OUTPUT_TYPES: ClassVar[dict[str, type]] = {"decision": RouteDecision}

    def __init__(self, llm_fallback: bool = True) -> None:
        self.llm_fallback = llm_fallback

    @component.output_types(decision=RouteDecision)
    def run(self, query: str) -> dict[str, RouteDecision]:
        if _GRAPH_RE.search(query):
            return {
                "decision": RouteDecision(
                    route="graph", confidence=0.7, reason="graph keyword match"
                )
            }

        if not self.llm_fallback:
            return {
                "decision": RouteDecision(
                    route="hybrid", confidence=0.5, reason="default hybrid"
                )
            }

        raw = chat_json(
            SYSTEM_PROMPT,
            query,
            fast=True,
            temperature=0.0,
            max_tokens=64,
            default={},
        )
        route_raw = str((raw or {}).get("route", "")).lower()
        route: RouteLabel = (
            "vector" if route_raw == "vector"
            else "graph" if route_raw == "graph"
            else "hybrid"
        )
        try:
            conf = float((raw or {}).get("confidence", 0.5))
        except (TypeError, ValueError):
            conf = 0.5
        return {
            "decision": RouteDecision(
                route=route, confidence=max(0.0, min(1.0, conf)), reason="llm"
            )
        }
