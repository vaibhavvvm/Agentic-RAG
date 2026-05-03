"""
RAG3 Self-Reflection (CRAG-style)
====================================
Grades the retrieved context for a query on three axes — relevance,
sufficiency, and faithfulness potential — and returns a composite
score plus a decision signal indicating whether to *accept*, *expand*,
or *abstain*.

This is the quality gate before passing context to the final answer
LLM.  Orchestrator uses the verdict to decide whether to trigger
another retrieval round (with query expansion or graph fallback).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Literal

from haystack import component
from haystack.dataclasses import Document

from src.config import get_settings
from src.monitoring.logger import get_logger
from src.utils.llm import chat_json

log = get_logger(__name__)

Verdict = Literal["accept", "expand", "abstain"]

SYSTEM_PROMPT = (
    "You grade retrieval results for a RAG system. Given a query and a list of "
    "snippets, score on three dimensions (0.0 – 1.0):\n"
    "  relevance    — do the snippets match the query topic?\n"
    "  sufficiency  — is there enough info to answer?\n"
    "  faithfulness — is the evidence unambiguous (few contradictions)?\n"
    "Return JSON: {\"relevance\": f, \"sufficiency\": f, \"faithfulness\": f, "
    "\"verdict\": \"accept|expand|abstain\", \"reason\": \"short\"}"
)


@dataclass(frozen=True)
class ReflectionReport:
    relevance: float
    sufficiency: float
    faithfulness: float
    verdict: Verdict
    reason: str

    @property
    def composite(self) -> float:
        return (self.relevance + self.sufficiency + self.faithfulness) / 3.0


@component
class SelfReflection:
    """
    LLM-based retrieval quality grader.

    Args:
        threshold: Minimum composite score to auto-accept (below → ``expand``).
    """

    OUTPUT_TYPES: ClassVar[dict[str, type]] = {
        "report": ReflectionReport,
        "accepted": bool,
    }

    def __init__(self, threshold: float | None = None) -> None:
        cfg = get_settings().retrieval
        self.threshold: float = (
            threshold if threshold is not None else cfg.self_reflection_threshold
        )
        self.enabled: bool = cfg.self_reflection_enabled

    @component.output_types(report=ReflectionReport, accepted=bool)
    def run(
        self, query: str, documents: list[Document]
    ) -> dict[str, object]:
        if not self.enabled:
            report = ReflectionReport(1.0, 1.0, 1.0, "accept", "reflection disabled")
            return {"report": report, "accepted": True}

        snippets = "\n---\n".join(
            (d.content or "")[:600] for d in documents[:6]
        ) or "(no context)"

        prompt = (
            f"Query: {query}\n\n"
            f"Snippets:\n{snippets}\n\n"
            "Grade now."
        )
        raw = chat_json(
            SYSTEM_PROMPT,
            prompt,
            fast=True,
            temperature=0.0,
            max_tokens=256,
            default={},
        )
        if not isinstance(raw, dict):
            raw = {}

        def _f(k: str) -> float:
            try:
                return max(0.0, min(1.0, float(raw.get(k, 0.0))))
            except (TypeError, ValueError):
                return 0.0

        verdict_raw = str(raw.get("verdict", "")).lower()
        verdict: Verdict = (
            "accept" if verdict_raw == "accept"
            else "abstain" if verdict_raw == "abstain"
            else "expand"
        )
        report = ReflectionReport(
            relevance=_f("relevance"),
            sufficiency=_f("sufficiency"),
            faithfulness=_f("faithfulness"),
            verdict=verdict,
            reason=str(raw.get("reason", ""))[:200],
        )
        # Rule: verdict wins, but composite score overrides "accept" below threshold
        accepted = report.verdict == "accept" and report.composite >= self.threshold
        return {"report": report, "accepted": accepted}
