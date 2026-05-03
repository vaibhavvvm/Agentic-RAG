"""
RAG3 Vector Worker Agent
==========================
Thin conversational wrapper around the ``AdvancedRAGAgent`` inner
class — a small ReAct-shaped controller that:

  1. Expands the query (``QueryExpander``).
  2. Runs hybrid vector search (``VectorSearchTool``).
  3. Grades results via ``SelfReflection``.
  4. Retries with expanded context if the grader rejects.
  5. Calls the LLM to produce a cited answer.

Separating the "advanced" loop from the outward-facing ``VectorAgent``
makes it easy to swap retrieval tactics per-intent without touching
the LLM-prompt layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from haystack.dataclasses import Document

from src.config import get_settings
from src.monitoring.logger import get_logger
from src.retrieval.session import RAGSession
from src.retrieval.tools.vector_tool import VectorSearchTool
from src.utils.llm import chat_sync

log = get_logger(__name__)


SYSTEM_PROMPT = (
    "You answer using ONLY the provided context snippets. Cite them "
    "inline as [1], [2] etc. corresponding to snippet order. If the "
    "answer is not in the context, say so plainly. Do not invent facts."
)


def _format_context(docs: list[Document]) -> str:
    parts: list[str] = []
    for i, d in enumerate(docs, start=1):
        parts.append(f"[{i}] {(d.content or '').strip()}")
    return "\n\n".join(parts) or "(no context)"


# ---------------------------------------------------------------------------
# Inner controller
# ---------------------------------------------------------------------------


@dataclass
class AdvancedRAGTrace:
    retries: int = 0
    rejected_by_reflector: bool = False
    accepted: bool = True
    doc_ids: list[str] = field(default_factory=list)


class AdvancedRAGAgent:
    """
    Inner retrieval+grading controller used by the outward ``VectorAgent``.

    Separating this lets tests exercise the retrieval contract without
    touching prompt/LLM wiring.

    Args:
        tool:        Configured ``VectorSearchTool`` (embeds query expansion,
                     hybrid search, reranker, and reflection internally).
        max_retries: How many times to broaden context when the reflector
                     rejects.  0 disables retry (still honours reflector).
    """

    def __init__(self, tool: VectorSearchTool, max_retries: int = 1) -> None:
        self._tool = tool
        self._max_retries = max(0, int(max_retries))

    def retrieve(self, query: str) -> tuple[list[Document], AdvancedRAGTrace]:
        trace = AdvancedRAGTrace()
        docs = self._tool(query)
        trace.doc_ids = [d.id for d in docs]

        # The VectorSearchTool already runs SelfReflection internally and
        # widens top_k when rejected. We still expose a second-chance
        # retry by re-issuing with a broader prompt.
        if not docs and self._max_retries > 0:
            trace.rejected_by_reflector = True
            broadened = f"{query} (broader / conceptual)"
            docs = self._tool(broadened)
            trace.retries = 1
            trace.doc_ids = [d.id for d in docs]

        trace.accepted = bool(docs)
        return docs, trace


# ---------------------------------------------------------------------------
# Public worker agent
# ---------------------------------------------------------------------------


class VectorAgent:
    """
    Conversational vector-RAG worker. Handles memory injection, prompt
    construction, and the final LLM call; delegates retrieval to
    ``AdvancedRAGAgent``.
    """

    def __init__(
        self,
        tool: VectorSearchTool,
        max_retries: int | None = None,
    ) -> None:
        cfg = get_settings().retrieval
        retries = cfg.max_reflection_rounds if max_retries is None else max_retries
        self._inner = AdvancedRAGAgent(tool=tool, max_retries=retries)

    async def run(
        self,
        query: str,
        session: RAGSession,
    ) -> tuple[str, list[Document]]:
        docs, _trace = self._inner.retrieve(query)
        mem_context = await session.build_context(query, top_k_episodic=3)

        parts: list[str] = []
        if mem_context:
            parts.append(mem_context)
        parts.append("[CONTEXT]\n" + _format_context(docs))
        parts.append(f"[QUESTION]\n{query}")
        user_msg = "\n\n".join(parts)

        answer = chat_sync(
            SYSTEM_PROMPT,
            user_msg,
            fast=False,
            temperature=0.1,
            max_tokens=768,
        )
        return answer, docs
