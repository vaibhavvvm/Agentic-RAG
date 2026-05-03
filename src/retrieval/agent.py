"""
RAG3 Haystack ReAct Retrieval Agent
=====================================
ReAct-style (Reason + Act) retrieval controller. Given a query, the LLM
is prompted to choose a *tool* from a registered set (vector, graph,
fusion, summary), observe the results, and decide whether to act again
or stop. Each tool returns a list of ``Document`` objects.

We use a lightweight in-house ReAct loop rather than Haystack's
higher-level ``Agent`` so we retain full control of:
  * the tool schema (matches our own ``*_tool.py`` facades),
  * the token budget per observation,
  * the termination signal (``FINAL:`` prefix).

The class still conforms to Haystack 2.x conventions: it is a plain
Python object and can be swapped into a Haystack pipeline via a thin
``@component`` wrapper if needed.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

from haystack.dataclasses import Document

from src.monitoring.logger import get_logger
from src.retrieval.strategies.graph_fusion import GraphVectorFusion
from src.retrieval.strategies.query_router import QueryRouter, RouteDecision
from src.retrieval.tools.graph_search_tool import GraphSearchTool
from src.retrieval.tools.vector_tool import VectorSearchTool
from src.utils.llm import chat_sync

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------


@dataclass
class ToolSpec:
    name: str
    description: str
    call: Callable[[str], list[Document]]


@dataclass
class AgentTrace:
    route: str = ""
    route_confidence: float = 0.0
    tools_called: list[str] = field(default_factory=list)
    observations: list[int] = field(default_factory=list)  # per-step doc counts
    n_docs: int = 0
    notes: list[str] = field(default_factory=list)
    terminated_by: str = ""


# ---------------------------------------------------------------------------
# ReAct prompts
# ---------------------------------------------------------------------------

_REACT_SYSTEM = """\
You are a retrieval planner that solves questions by calling tools.
Available tools:
{tool_block}

Workflow:
1. Read the question.
2. Choose ONE tool to call.
3. Observe the retrieved snippets (provided as [1], [2], ...).
4. Either call another tool (if more evidence is needed) OR reply
   starting with \"FINAL:\" to stop.

Response format per turn (strict):
  THOUGHT: <one sentence>
  ACTION: <tool name exactly>
  INPUT: <query to send to the tool>

When you have enough context, instead reply:
  FINAL: done

Never invent tool names. Never call the same tool twice with the same INPUT.
"""

_ACTION_RE = re.compile(r"ACTION:\s*(\S+)", re.IGNORECASE)
_INPUT_RE = re.compile(r"INPUT:\s*(.+)", re.IGNORECASE)
_FINAL_RE = re.compile(r"^\s*FINAL\s*:", re.IGNORECASE | re.MULTILINE)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class RetrievalAgent:
    """
    ReAct-style retrieval agent over vector / graph / fusion tools.

    Args:
        vector_tool:           Full vector sub-pipeline.
        graph_tool:            Graph search facade.
        fusion:                Optional ``GraphVectorFusion`` for hybrid.
        router:                Query router used as a fast-path prior.
        max_steps:             Max ReAct iterations.
        confidence_threshold:  Below this the router output is treated
                               as "uncertain" and the ReAct loop is
                               invoked instead of a direct dispatch.
    """

    def __init__(
        self,
        vector_tool: VectorSearchTool,
        graph_tool: GraphSearchTool,
        fusion: GraphVectorFusion | None = None,
        router: QueryRouter | None = None,
        max_steps: int = 3,
        confidence_threshold: float = 0.75,
    ) -> None:
        self._router = router or QueryRouter()
        self._max_steps = max_steps
        self._confidence_threshold = confidence_threshold

        self._tools: dict[str, ToolSpec] = {
            "vector_search": ToolSpec(
                name="vector_search",
                description="Dense+BM25 hybrid search for factual lookups / definitions.",
                call=vector_tool,
            ),
            "graph_search": ToolSpec(
                name="graph_search",
                description="Graph traversal for relational / multi-hop questions.",
                call=graph_tool,
            ),
        }
        if fusion is not None:
            self._tools["fusion_search"] = ToolSpec(
                name="fusion_search",
                description=(
                    "Run vector + graph in parallel and fuse with RRF. Use when the "
                    "question mixes definitional lookup with relationships."
                ),
                call=lambda q: fusion.run(query=q)["documents"],  # type: ignore[misc]
            )

    # ------------------------------------------------------------------

    def run(self, query: str) -> tuple[list[Document], AgentTrace]:
        trace = AgentTrace()
        decision: RouteDecision = self._router.run(query=query)["decision"]
        trace.route = decision.route
        trace.route_confidence = decision.confidence

        # Fast path: confident router → dispatch directly, skip ReAct
        if decision.confidence >= self._confidence_threshold:
            tool_name = {
                "vector": "vector_search",
                "graph": "graph_search",
                "hybrid": "fusion_search" if "fusion_search" in self._tools else "vector_search",
            }.get(decision.route, "vector_search")
            trace.tools_called.append(tool_name)
            trace.terminated_by = "router_fastpath"
            docs = self._tools[tool_name].call(query)
            trace.observations.append(len(docs))
            trace.n_docs = len(docs)
            return docs, trace

        # Slow path: ReAct loop
        return self._react_loop(query, trace)

    # ------------------------------------------------------------------
    # ReAct loop
    # ------------------------------------------------------------------

    def _react_loop(
        self, query: str, trace: AgentTrace
    ) -> tuple[list[Document], AgentTrace]:
        tool_block = "\n".join(
            f"- {t.name}: {t.description}" for t in self._tools.values()
        )
        system_prompt = _REACT_SYSTEM.format(tool_block=tool_block)

        scratchpad: list[str] = [f"QUESTION: {query}"]
        collected: dict[str, Document] = {}
        seen_actions: set[tuple[str, str]] = set()

        for step in range(1, self._max_steps + 1):
            reply = chat_sync(
                system_prompt,
                "\n".join(scratchpad) + "\nWhat is your next step?",
                fast=True,
                temperature=0.0,
                max_tokens=256,
            )
            scratchpad.append(f"STEP {step}:\n{reply}")

            if _FINAL_RE.search(reply):
                trace.terminated_by = "final"
                break

            action_match = _ACTION_RE.search(reply)
            input_match = _INPUT_RE.search(reply)
            if not action_match or not input_match:
                trace.notes.append(f"step {step}: malformed reply")
                trace.terminated_by = "malformed"
                break

            action = action_match.group(1).strip().lower()
            tool_input = input_match.group(1).strip()
            if action not in self._tools:
                trace.notes.append(f"step {step}: unknown tool {action!r}")
                trace.terminated_by = "unknown_tool"
                break
            if (action, tool_input) in seen_actions:
                trace.notes.append(f"step {step}: repeated action")
                trace.terminated_by = "repeat"
                break
            seen_actions.add((action, tool_input))

            docs = self._tools[action].call(tool_input)
            trace.tools_called.append(action)
            trace.observations.append(len(docs))

            for d in docs:
                collected.setdefault(d.id, d)

            # Feed an abbreviated observation back into the scratchpad
            snippets = [
                f"[{i+1}] {(d.content or '')[:180]}"
                for i, d in enumerate(docs[:5])
            ]
            scratchpad.append(
                "OBSERVATION (top results):\n"
                + ("\n".join(snippets) if snippets else "(no results)")
            )
        else:
            trace.terminated_by = "max_steps"

        ordered = list(collected.values())
        trace.n_docs = len(ordered)
        return ordered, trace
