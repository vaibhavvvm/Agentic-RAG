"""
RAG LangGraph Orchestrator
=============================
Multi-agent supervisor implemented as a LangGraph state machine with
conditional edges and a reflect / rewrite loop.

State machine::

    START
      │
    route ──► general ──────────────────────────────────────► END
      │
      ├──► vector_search ──────────────────────┐
      └──► graph_search  ──────────────────────┤
                                               ▼
                                            reflect
                                           /       \
                                      accept        rewrite (capped)
                                      /                  \
                               synthesize          rewrite_query
                                   │                    │
                                  END       ┌───────────┘
                                            ├──► vector_search  (loop back)
                                            └──► graph_search

Nodes use LangChain ``ChatPromptTemplate + PydanticOutputParser`` for
structured outputs (reflect, rewrite, synthesize). LangGraph + LangSmith
env-vars already set by ``src.monitoring.langsmith.setup_langsmith()``.
"""

from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from haystack.dataclasses import Document

from src.agents.lc_models import (
    make_reflect_chain,
    make_rewrite_chain,
    make_synth_chain,
)
from src.agents.router import Intent, IntentRouter, RouterDecision
from src.agents.synthesizer import SynthesisedResponse, Synthesiser
from src.ingestion.embedder import CachedOllamaEmbedder
from src.monitoring.logger import get_logger, set_request_id, set_session_id
from src.monitoring.metrics import MetricsCollector
from src.retrieval.session import RAGSession
from src.retrieval.strategies.reranking import OllamaRanker
from src.retrieval.tools.graph_search_tool import GraphSearchTool
from src.retrieval.tools.vector_tool import VectorSearchTool
from src.storage.base import BaseGraphStore, BaseVectorStore
from src.utils.llm import chat_json, chat_sync

log = get_logger(__name__)

try:
    from langgraph.graph import END, StateGraph
    _LANGGRAPH_AVAILABLE = True
except Exception:
    StateGraph = None  # type: ignore
    END = "__end__"  # type: ignore
    _LANGGRAPH_AVAILABLE = False


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


@dataclass
class OrchestratorState:
    query: str                                      # current (possibly rewritten) query
    original_query: str = ""
    mem_context: str = ""                           # pre-fetched from session
    intent: str = ""                                # "general"|"vector"|"graph"|"hybrid"
    docs: list[Document] = field(default_factory=list)
    answer: str = ""
    reflect_verdict: str = "accept"                 # "accept" | "rewrite"
    reflect_reason: str = ""
    retry_count: int = 0
    max_retries: int = 2
    request_id: str = ""
    trace: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Dependency container
# ---------------------------------------------------------------------------


@dataclass
class _Deps:
    router: IntentRouter
    vector_tool: VectorSearchTool
    graph_tool: GraphSearchTool


# ---------------------------------------------------------------------------
# Prompt strings
# ---------------------------------------------------------------------------

_ENTITY_PROMPT = (
    "Extract 1–6 named entities, technical terms, or key concepts from the "
    "user query that would be useful as graph search nodes. "
    'Return valid JSON: {"entities": ["entity1", "entity2"]}. '
    "Rules: lowercase, no duplicates, prefer specific terms over generic ones. "
    'If the query is conversational (greetings, opinions), return {"entities": []}.'
)

_SYNTH_FALLBACK_SYSTEM = (
    "You are an expert research assistant. Answer the user's question using "
    "ONLY the provided context snippets and conversation memory.\n\n"
    "Rules:\n"
    "1. Cite sources inline as [1], [2], etc. matching the context snippet numbers.\n"
    "2. If the answer is NOT in the context, state clearly: "
    '"I don\'t have enough information in the knowledge base to answer this."\n'
    "3. Do NOT invent, hallucinate, or extrapolate beyond what the context states.\n"
    "4. Use the conversation memory to understand follow-up questions and maintain coherence.\n"
    "5. Structure your answer clearly with markdown formatting when appropriate.\n"
    "6. If multiple sources corroborate, synthesize them into a unified answer."
)


# ---------------------------------------------------------------------------
# Node implementations
# ---------------------------------------------------------------------------


def _route_node(deps: _Deps):
    def _node(state: OrchestratorState) -> OrchestratorState:
        decision: RouterDecision = deps.router.route(state.query)
        state.intent = {
            Intent.GENERAL: "general",
            Intent.VECTOR: "vector",
            Intent.GRAPH: "graph",
            Intent.HYBRID: "hybrid",
        }.get(decision.intent, "vector")
        state.trace.update({
            "intent": state.intent,
            "router_tier": decision.tier,
            "router_confidence": decision.confidence,
            "router_reason": decision.reason,
        })
        log.info("Routed", extra={"intent": state.intent, "tier": decision.tier})
        return state
    return _node


def _general_node():
    _system = (
        "You are an intelligent assistant powered by a Retrieval-Augmented Generation system.\n"
        "You have access to the user's conversation history and episodic memory summaries.\n\n"
        "Rules:\n"
        "1. Use the conversation history and memory context to give personalized, contextual answers.\n"
        "2. If memory context references past discussions, acknowledge them naturally.\n"
        "3. Be concise but thorough. Prefer structured answers (bullet points, numbered lists).\n"
        "4. Never fabricate information. If unsure, say so.\n"
        "5. For greetings and small talk, be warm and brief."
    )

    def _node(state: OrchestratorState) -> OrchestratorState:
        user = (
            f"{state.mem_context}\n\n[CURRENT TURN]\nUSER: {state.query}"
            if state.mem_context else state.query
        )
        state.answer = chat_sync(_system, user, fast=True, temperature=0.4, max_tokens=512)
        return state
    return _node


def _vector_search_node(deps: _Deps):
    def _node(state: OrchestratorState) -> OrchestratorState:
        state.docs = deps.vector_tool(state.query)
        if not state.docs and state.retry_count == 0:
            state.docs = deps.vector_tool(state.query + " broader context overview")
        log.debug("Vector search", extra={"n_docs": len(state.docs), "retry": state.retry_count})
        return state
    return _node


def _graph_search_node(deps: _Deps):
    def _node(state: OrchestratorState) -> OrchestratorState:
        raw = chat_json(
            _ENTITY_PROMPT, state.query,
            fast=True, temperature=0.0, max_tokens=128,
            default={"entities": []},
        )
        entities = [
            str(e).strip().lower()
            for e in (raw or {}).get("entities", [])
            if isinstance(e, str) and e.strip()
        ][:6]
        if not entities:
            entities = list({t.lower() for t in re.findall(r"[A-Za-z][A-Za-z0-9\-_]{2,}", state.query)})[:6]

        search_q = state.query + (" | entities: " + ", ".join(entities) if entities else "")
        state.docs = deps.graph_tool(search_q)
        log.debug("Graph search", extra={"n_docs": len(state.docs), "entities": entities})
        return state
    return _node


def _reflect_node(reflect_chain, rewrite_chain):
    """Grade retrieved context; set verdict + optionally a rewritten query."""

    def _node(state: OrchestratorState) -> OrchestratorState:
        if not state.docs:
            state.reflect_verdict = "rewrite"
            state.reflect_reason = "no documents retrieved"
            log.debug("Reflect: no docs → rewrite")
            return state

        context = "\n\n".join(
            f"[{i+1}] {(d.content or '')[:300]}"
            for i, d in enumerate(state.docs[:6])
        )

        # --- LangChain PydanticOutputParser path ---
        if reflect_chain is not None:
            try:
                out = reflect_chain.invoke({"query": state.query, "context": context})
                state.reflect_verdict = out.verdict
                state.reflect_reason = out.reason
                log.info("Reflect (LangChain)", extra={"verdict": out.verdict})
                # If rewrite requested, generate the new query now
                if out.verdict == "rewrite" and rewrite_chain is not None:
                    try:
                        rw = rewrite_chain.invoke({
                            "query": state.original_query,
                            "reason": out.reason or "insufficient context",
                        })
                        state.query = rw.query or state.original_query
                    except Exception as exc:
                        log.debug("Rewrite chain failed", extra={"err": str(exc)[:100]})
                return state
            except Exception as exc:
                log.warning("Reflect chain failed, using heuristic", extra={"err": str(exc)[:120]})

        # --- Heuristic fallback ---
        state.reflect_verdict = "accept" if len(state.docs) >= 2 else "rewrite"
        state.reflect_reason = "heuristic"
        return state

    return _node


def _rewrite_query_node(state: OrchestratorState) -> OrchestratorState:
    """Increment retry counter. Query already updated by reflect node if chain worked."""
    state.retry_count += 1
    if state.query == state.original_query:
        state.query = state.original_query + " alternative explanation context"
    log.info("Query rewritten", extra={"retry": state.retry_count, "query": state.query[:80]})
    return state


def _synthesize_node(synth_chain):
    """Final LLM synthesis — LangChain ChatPromptTemplate + PydanticOutputParser."""

    def _node(state: OrchestratorState) -> OrchestratorState:
        context = _format_context(state.docs)

        is_analytical = any(
            kw in state.original_query.lower()
            for kw in ["explain", "detail", "deep", "analyze", "impact", "effect", "difference"]
        )
        verbosity_instruction = (
            "\n\n[INSTRUCTION]\nThe user has requested an in-depth explanation or analysis. "
            "Please provide a comprehensive, multi-paragraph response covering all relevant nuances and details."
            if is_analytical else ""
        )

        if synth_chain is not None:
            try:
                out = synth_chain.invoke({
                    "query": state.original_query + verbosity_instruction,
                    "context": context,
                    "memory": state.mem_context or "(no prior conversation)",
                })
                state.answer = out.answer
                state.trace["synth_confidence"] = out.confidence
                state.trace["synth_backend"] = "langchain"
                log.info("Synthesized (LangChain)", extra={"confidence": out.confidence, "analytical": is_analytical})
                return state
            except Exception as exc:
                log.warning("Synth chain failed, using chat_sync", extra={"err": str(exc)[:120]})

        user = (
            (state.mem_context + "\n\n" if state.mem_context else "")
            + f"[CONTEXT]\n{context}\n\n[QUESTION]\n{state.original_query}"
            + verbosity_instruction
        )
        state.answer = chat_sync(
            _SYNTH_FALLBACK_SYSTEM, user, fast=False, temperature=0.1, max_tokens=2048
        )
        state.trace["synth_backend"] = "chat_sync"
        return state

    return _node


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _format_context(docs: list[Document]) -> str:
    return "\n\n".join(
        f"[{i+1}] {(d.content or '').strip()}"
        for i, d in enumerate(docs)
    ) or "(no context retrieved)"


# ---------------------------------------------------------------------------
# Conditional edge functions
# ---------------------------------------------------------------------------


def _intent_router(state: OrchestratorState) -> str:
    if state.intent == "general":
        return "general"
    if state.intent == "graph":
        return "graph_search"
    return "vector_search"  # vector | hybrid both start with vector


def _reflect_router(state: OrchestratorState) -> str:
    if state.reflect_verdict == "rewrite" and state.retry_count < state.max_retries:
        return "rewrite_query"
    return "synthesize"


def _retry_router(state: OrchestratorState) -> str:
    return "graph_search" if state.intent == "graph" else "vector_search"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


def _build_langgraph(deps: _Deps, reflect_chain, rewrite_chain, synth_chain):
    g = StateGraph(OrchestratorState)

    g.add_node("route", _route_node(deps))
    g.add_node("general", _general_node())
    g.add_node("vector_search", _vector_search_node(deps))
    g.add_node("graph_search", _graph_search_node(deps))
    g.add_node("reflect", _reflect_node(reflect_chain, rewrite_chain))
    g.add_node("rewrite_query", _rewrite_query_node)
    g.add_node("synthesize", _synthesize_node(synth_chain))

    g.set_entry_point("route")
    g.add_conditional_edges(
        "route", _intent_router,
        {"general": "general", "vector_search": "vector_search", "graph_search": "graph_search"},
    )
    g.add_edge("general", END)
    g.add_edge("vector_search", "reflect")
    g.add_edge("graph_search", "reflect")
    g.add_conditional_edges(
        "reflect", _reflect_router,
        {"rewrite_query": "rewrite_query", "synthesize": "synthesize"},
    )
    g.add_conditional_edges(
        "rewrite_query", _retry_router,
        {"vector_search": "vector_search", "graph_search": "graph_search"},
    )
    g.add_edge("synthesize", END)

    return g.compile()


# ---------------------------------------------------------------------------
# Sequential fallback (identical semantics without langgraph)
# ---------------------------------------------------------------------------


def _run_sequential(
    state: OrchestratorState,
    deps: _Deps,
    reflect_chain,
    rewrite_chain,
    synth_chain,
) -> OrchestratorState:
    state = _route_node(deps)(state)

    if state.intent == "general":
        state = _general_node()(state)
        return state

    search = _vector_search_node(deps) if state.intent != "graph" else _graph_search_node(deps)
    reflect = _reflect_node(reflect_chain, rewrite_chain)
    synth = _synthesize_node(synth_chain)

    state = search(state)
    state = reflect(state)

    while state.reflect_verdict == "rewrite" and state.retry_count < state.max_retries:
        state = _rewrite_query_node(state)
        state = search(state)
        state = reflect(state)

    state = synth(state)
    return state


# ---------------------------------------------------------------------------
# Public orchestrator
# ---------------------------------------------------------------------------


class Orchestrator:
    """
    LangGraph multi-agent supervisor. Falls back to sequential execution
    when ``langgraph`` is not installed.

    Build once per process; call ``ask(query, session)`` per turn.
    """

    def __init__(
        self,
        vector_store: BaseVectorStore,
        graph_store: BaseGraphStore,
        embedder: CachedOllamaEmbedder | None = None,
        reranker: OllamaRanker | None = None,
    ) -> None:
        _embedder = embedder or CachedOllamaEmbedder()
        _reranker = reranker or OllamaRanker()

        self._deps = _Deps(
            router=IntentRouter(),
            vector_tool=VectorSearchTool(
                vector_store=vector_store,
                embedder=_embedder,
                reranker=_reranker,
            ),
            graph_tool=GraphSearchTool(graph_store=graph_store),
        )

        self._reflect_chain = make_reflect_chain()
        self._rewrite_chain = make_rewrite_chain()
        self._synth_chain = make_synth_chain()

        self._graph = (
            _build_langgraph(
                self._deps,
                self._reflect_chain,
                self._rewrite_chain,
                self._synth_chain,
            )
            if _LANGGRAPH_AVAILABLE else None
        )
        if not _LANGGRAPH_AVAILABLE:
            log.warning("langgraph not installed — Orchestrator uses sequential fallback")

        self._synth = Synthesiser()
        self._metrics = MetricsCollector.get_instance()

    # ------------------------------------------------------------------

    async def ask(self, query: str, session: RAGSession) -> SynthesisedResponse:
        request_id = str(uuid.uuid4())[:8]
        set_request_id(request_id)
        set_session_id(session.session_id)
        started = time.monotonic()

        mem_context = await session.build_context(query, top_k_episodic=3)
        session.add_turn("user", query)

        state = OrchestratorState(
            query=query,
            original_query=query,
            mem_context=mem_context,
            max_retries=2,
            request_id=request_id,
        )

        try:
            with self._metrics.measure("orchestrator.run"):
                if self._graph is not None:
                    out = self._graph.invoke(state)
                    if isinstance(out, dict):
                        answer = out.get("answer", "") or ""
                        docs = out.get("docs", []) or []
                        trace = out.get("trace", {}) or {}
                        intent = out.get("intent", "")
                    else:
                        answer = getattr(out, "answer", "")
                        docs = getattr(out, "docs", [])
                        trace = getattr(out, "trace", {})
                        intent = getattr(out, "intent", "")
                else:
                    state = _run_sequential(
                        state, self._deps,
                        self._reflect_chain, self._rewrite_chain, self._synth_chain,
                    )
                    answer, docs, trace, intent = (
                        state.answer, state.docs, state.trace, state.intent
                    )
        except Exception:
            log.exception("Orchestrator failed")
            self._metrics.record_event("orchestrator.errors")
            answer = "Sorry, I ran into an internal error. Please try again."
            docs, trace, intent = [], {}, "error"

        latency_ms = (time.monotonic() - started) * 1000
        trace.update({
            "request_id": request_id,
            "session_id": session.session_id,
            "latency_ms": round(latency_ms, 2),
            "n_documents": len(docs),
            "intent": intent,
        })

        session.add_turn("assistant", answer, metadata={"intent": intent})
        return self._synth.run(answer=answer, documents=docs, trace=trace)
