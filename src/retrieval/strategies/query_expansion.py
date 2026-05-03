"""
RAG3 Query Expansion
=====================
LLM-driven query reformulation: given the user's question, produce N
diverse rewrites (paraphrases, entity-focused variants, higher-level
abstractions).  Each variant is then retrieved independently and the
results are fused (RRF) downstream.

Output shape from the LLM is a JSON array of strings; malformed
responses gracefully degrade to the original query.
"""

from __future__ import annotations

from typing import ClassVar

from haystack import component

from src.config import get_settings
from src.monitoring.logger import get_logger
from src.utils.llm import chat_json

log = get_logger(__name__)

SYSTEM_PROMPT = (
    "You rewrite a search query into diverse variants that capture different "
    "intents (paraphrase, entity-centric, broader, narrower). Return ONLY a "
    "JSON array of strings — no prose, no markdown fences."
)


@component
class QueryExpander:
    """
    Generate multiple query variants via an LLM.

    Args:
        num_variants: How many rewrites to produce (excluding the original).
        model:        Model override; uses ``groq.fast_model`` by default.
    """

    OUTPUT_TYPES: ClassVar[dict[str, type]] = {"queries": list}

    def __init__(
        self,
        num_variants: int | None = None,
        model: str | None = None,
    ) -> None:
        cfg = get_settings().retrieval
        self.num_variants: int = num_variants or cfg.num_expanded_queries
        self.model: str | None = model
        self.enabled: bool = cfg.query_expansion_enabled

    @component.output_types(queries=list)
    def run(self, query: str) -> dict[str, list[str]]:
        if not self.enabled or self.num_variants <= 0:
            return {"queries": [query]}

        prompt = (
            f"Original query: {query}\n"
            f"Produce exactly {self.num_variants} diverse rewrites as a JSON array."
        )
        variants = chat_json(
            SYSTEM_PROMPT,
            prompt,
            fast=True,
            model=self.model,
            temperature=0.3,
            max_tokens=256,
            default=[],
        )
        if not isinstance(variants, list):
            variants = []

        cleaned: list[str] = [query]
        for v in variants:
            if isinstance(v, str) and v.strip() and v.strip() != query:
                cleaned.append(v.strip())
            if len(cleaned) > self.num_variants + 1:
                break

        return {"queries": cleaned}
