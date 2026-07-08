"""
RAG Progressive Retrieval Fallback
=====================================
Chains retrieval strategies in increasing-cost order and returns the
first result set that satisfies a quality gate.

Order (default):
    1. vector-only (cheap, fast)
    2. hybrid (vector + BM25 RRF)
    3. graph+vector fusion
    4. summary index broadening

Each stage runs only when the previous stage fails the gate.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from haystack.dataclasses import Document

from src.monitoring.logger import get_logger

log = get_logger(__name__)

RetrieveFn = Callable[[str], Awaitable[list[Document]]]
GateFn = Callable[[list[Document]], bool]


@dataclass
class FallbackStage:
    name: str
    retrieve: RetrieveFn
    gate: GateFn


class ProgressiveFallback:
    """Runs stages sequentially; stops when ``gate`` accepts the result."""

    def __init__(self, stages: list[FallbackStage]) -> None:
        if not stages:
            raise ValueError("ProgressiveFallback requires at least one stage.")
        self._stages = stages

    async def run(self, query: str) -> tuple[list[Document], str]:
        """
        Returns:
            (documents, winning_stage_name)
        """
        last_docs: list[Document] = []
        for stage in self._stages:
            try:
                docs = await stage.retrieve(query)
            except Exception as exc:
                log.warning(
                    "Fallback stage errored",
                    extra={"stage": stage.name, "error": str(exc)},
                )
                continue
            last_docs = docs
            if stage.gate(docs):
                log.info(
                    "Fallback accepted",
                    extra={"stage": stage.name, "n_docs": len(docs)},
                )
                return docs, stage.name
        log.warning(
            "All fallback stages failed gate; returning last non-empty result",
            extra={"n_docs": len(last_docs)},
        )
        return last_docs, self._stages[-1].name


def min_docs_gate(min_docs: int = 1) -> GateFn:
    """Accept if at least ``min_docs`` documents were returned."""
    return lambda docs: len(docs) >= min_docs


def min_score_gate(score_key: str, threshold: float) -> GateFn:
    """Accept if the top document's ``meta[score_key]`` ≥ ``threshold``."""
    def _gate(docs: list[Document]) -> bool:
        if not docs:
            return False
        top = docs[0].meta or {}
        try:
            return float(top.get(score_key, 0.0)) >= threshold
        except (TypeError, ValueError):
            return False
    return _gate
