"""
RAG3 LangGraph-Based Graph Retrieval Agent
=============================================
Stateful multi-step graph worker implemented as a ``langgraph`` state
machine. Each node is a pure function that mutates the shared
``GraphAgentState``; edges encode conditional control flow.

State machine
-------------

    entity_extract ─► graph_search ─► grade ──accept──► answer ─► END
                                         │
                                         └──expand──► fact_expand ─► answer

Nodes
-----
* **entity_extract** — pulls candidate entities from the query via a
  fast LLM call (falls back to keyword extractor if the LLM errors).
* **graph_search** — runs the shared ``GraphSearchTool`` to fetch facts
  + entity subgraph up to ``max_hops``.
* **grade** — asks a fast LLM whether the retrieved context is enough
  to answer; returns ``accept`` | ``expand``.
* **fact_expand** — re-issues search with an expanded hop radius when
  the grader says ``expand``.
* **answer** — final LLM answer constrained to cite only graph facts.

If ``langgraph`` is not installed, the module falls back to a
sequential Python implementation with identical semantics so the system
still runs in stripped environments / CI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

from haystack.dataclasses import Document

from src.monitoring.logger import get_logger
from src.retrieval.session import RAGSession
from src.retrieval.tools.graph_search_tool import GraphSearchTool
from src.utils.llm import chat_json, chat_sync

log = get_logger(__name__)

try:  # pragma: no cover - optional dep
    from langgraph.graph import END, StateGraph  # type: ignore
    _LANGGRAPH_AVAILABLE = True
except Exception:
    StateGraph = None  # type: ignore
    END = "__end__"  # type: ignore
    _LANGGRAPH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------


@dataclass
class GraphAgentState:
    query: str
    mem_context: str = ""
    entities: list[str] = field(default_factory=list)
    docs: list[Document] = field(default_factory=list)
    max_hops: int = 2
    verdict: str = ""                # "accept" | "expand"
    expansions: int = 0
    max_expansions: int = 1
    answer: str = ""


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_ENTITY_PROMPT = (
    "Extract 1–6 named entities, concepts, or keywords from the query "
    "that would be nodes in a knowledge graph. Return JSON: "
    '{"entities": ["..."]}. Lowercase, no duplicates, no stopwords.'
)

_GRADER_PROMPT = (
    "Decide whether the graph facts and entity records are sufficient to "
    "answer the user's question. Return JSON: "
    '{"verdict": "accept|expand", "reason": "short"}. '
    'Choose "expand" if the answer would require information not present.'
)

_ANSWER_PROMPT = (
    "You answer relational / multi-hop questions using ONLY the provided "
    "graph facts and entity records. Reconstruct the chain of "
    "relationships step by step. If evidence is insufficient, say so and "
    "do not invent edges."
)


# ---------------------------------------------------------------------------
# Node implementations
# ---------------------------------------------------------------------------


def _extract_entities_node(state: GraphAgentState) -> GraphAgentState:
    raw = chat_json(
        _ENTITY_PROMPT,
        state.query,
        fast=True,
        temperature=0.0,
        max_tokens=128,
        default={"entities": []},
    )
    ents = (raw or {}).get("entities") or []
    state.entities = [
        str(e).strip().lower() for e in ents if isinstance(e, str) and e.strip()
    ][:6]
    if not state.entities:
        # Last-ditch: cheap keyword fallback
        import re
        toks = re.findall(r"[A-Za-z][A-Za-z0-9\-_]{2,}", state.query)
        state.entities = list({t.lower() for t in toks})[:6]
    return state


def _graph_search_node(tool: GraphSearchTool):
    def _node(state: GraphAgentState) -> GraphAgentState:
        query = state.query
        if state.entities:
            query = state.query + " | entities: " + ", ".join(state.entities)
        state.docs = tool(query)
        return state
    return _node


def _grade_node(state: GraphAgentState) -> GraphAgentState:
    if not state.docs:
        state.verdict = "expand"
        return state
    context = _format_graph_context(state.docs)
    raw = chat_json(
        _GRADER_PROMPT,
        f"Query: {state.query}\n\nContext:\n{context}",
        fast=True,
        temperature=0.0,
        max_tokens=96,
        default={"verdict": "accept"},
    )
    state.verdict = str((raw or {}).get("verdict", "accept")).lower()
    if state.verdict not in ("accept", "expand"):
        state.verdict = "accept"
    return state


def _expand_node(tool: GraphSearchTool):
    def _node(state: GraphAgentState) -> GraphAgentState:
        state.expansions += 1
        state.max_hops = min(state.max_hops + 1, 4)
        query = state.query + " | deeper relationships | entities: " + ", ".join(state.entities)
        more = tool(query)
        seen = {d.id for d in state.docs}
        for d in more:
            if d.id not in seen:
                state.docs.append(d)
        return state
    return _node


def _answer_node(state: GraphAgentState) -> GraphAgentState:
    context = _format_graph_context(state.docs) or "(no graph context)"
    user_msg_parts: list[str] = []
    if state.mem_context:
        user_msg_parts.append(state.mem_context)
    user_msg_parts.append("[GRAPH CONTEXT]\n" + context)
    user_msg_parts.append(f"[QUESTION]\n{state.query}")

    state.answer = chat_sync(
        _ANSWER_PROMPT,
        "\n\n".join(user_msg_parts),
        fast=False,
        temperature=0.1,
        max_tokens=768,
    )
    return state


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _format_graph_context(docs: list[Document]) -> str:
    facts: list[str] = []
    entities: list[str] = []
    for d in docs:
        meta = d.meta or {}
        bucket = facts if meta.get("source") == "graph_fact" else entities
        bucket.append((d.content or "").strip())
    out: list[str] = []
    if facts:
        out.append("FACTS:\n" + "\n".join(f"- {f}" for f in facts))
    if entities:
        out.append("ENTITIES:\n" + "\n---\n".join(entities))
    return "\n\n".join(out)


def _grade_router(state: GraphAgentState) -> str:
    """Conditional edge function used by langgraph."""
    if state.verdict == "expand" and state.expansions < state.max_expansions:
        return "expand"
    return "answer"


# ---------------------------------------------------------------------------
# Public agent
# ---------------------------------------------------------------------------


class GraphAgent:
    """
    LangGraph state-machine agent.

    Args:
        tool:            Shared ``GraphSearchTool`` instance.
        max_expansions:  Maximum number of expand→search rounds before
                         committing to an answer with whatever exists.
    """

    def __init__(
        self,
        tool: GraphSearchTool,
        max_expansions: int = 1,
    ) -> None:
        self._tool = tool
        self._max_expansions = max_expansions
        self._graph = self._build_graph() if _LANGGRAPH_AVAILABLE else None
        if not _LANGGRAPH_AVAILABLE:
            log.warning(
                "langgraph not installed — GraphAgent using sequential fallback"
            )

    def _build_graph(self):  # pragma: no cover - requires langgraph
        g = StateGraph(GraphAgentState)
        g.add_node("entity_extract", _extract_entities_node)
        g.add_node("graph_search", _graph_search_node(self._tool))
        g.add_node("grade", _grade_node)
        g.add_node("fact_expand", _expand_node(self._tool))
        g.add_node("answer", _answer_node)

        g.set_entry_point("entity_extract")
        g.add_edge("entity_extract", "graph_search")
        g.add_edge("graph_search", "grade")
        g.add_conditional_edges(
            "grade", _grade_router, {"expand": "fact_expand", "answer": "answer"}
        )
        g.add_edge("fact_expand", "grade")
        g.add_edge("answer", END)
        return g.compile()

    # ------------------------------------------------------------------

    async def run(
        self, query: str, session: RAGSession
    ) -> tuple[str, list[Document]]:
        mem_context = await session.build_context(query, top_k_episodic=3)
        init = GraphAgentState(
            query=query,
            mem_context=mem_context,
            max_expansions=self._max_expansions,
        )

        if self._graph is not None:  # pragma: no cover
            out = self._graph.invoke(init)
            # langgraph may return a dict-like state
            if isinstance(out, dict):
                docs = out.get("docs", []) or []
                answer = out.get("answer", "") or ""
            else:
                docs = getattr(out, "docs", [])
                answer = getattr(out, "answer", "")
            return answer, docs

        # Sequential fallback
        state = _extract_entities_node(init)
        state = _graph_search_node(self._tool)(state)
        state = _grade_node(state)
        while (
            state.verdict == "expand"
            and state.expansions < state.max_expansions
        ):
            state = _expand_node(self._tool)(state)
            state = _grade_node(state)
        state = _answer_node(state)
        return state.answer, state.docs
