"""
LangChain structured-output helpers.

Builds ChatPromptTemplate | LLM | PydanticOutputParser chains using the
project's existing chat_sync() provider chain as the LLM backend so no
extra API credentials are needed.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from src.monitoring.logger import get_logger
from src.utils.llm import chat_sync

log = get_logger(__name__)

try:
    from langchain_core.output_parsers import PydanticOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnableLambda, RunnableSerializable
    _LC_AVAILABLE = True
except ImportError:
    _LC_AVAILABLE = False
    log.warning("langchain-core not installed — structured chains unavailable")


# ---------------------------------------------------------------------------
# Output schemas
# ---------------------------------------------------------------------------


class RouterOutput(BaseModel):
    intent: Literal[
        "general_chat", "vector_retrieval", "graph_retrieval", "hybrid_retrieval"
    ]
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    reason: str = ""


class ReflectOutput(BaseModel):
    verdict: Literal["accept", "rewrite"]
    reason: str = ""


class RewriteOutput(BaseModel):
    query: str


class SynthesisOutput(BaseModel):
    answer: str
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_ROUTER_TMPL = """\
You are classifying a user query for an intelligent RAG system.

Intent categories:
- "general_chat": greetings, opinions, small talk, meta questions about the assistant itself
- "vector_retrieval": factual questions, definitions, "what is X", "explain Y", document content lookup
- "graph_retrieval": relational questions, "how does X relate to Y", multi-hop reasoning, causal chains, entity connections
- "hybrid_retrieval": questions needing both factual document content AND entity relationships

Guidelines:
- Follow-up questions (e.g. "tell me more", "what about X") should inherit the previous intent type
- Questions about document content → vector_retrieval
- Questions about connections between concepts → graph_retrieval
- When in doubt, prefer vector_retrieval over graph_retrieval

{format_instructions}

Query: {query}"""

_REFLECT_TMPL = """\
Grade the retrieved context for answering the query.
- "accept": context is sufficient
- "rewrite": context is insufficient or irrelevant

{format_instructions}

Query: {query}

Retrieved context (excerpt):
{context}"""

_REWRITE_TMPL = """\
Rewrite the user query to improve retrieval. Keep the same intent but use
different phrasing, synonyms, or decompose into sub-topics.

{format_instructions}

Original query: {query}
Retrieval failure reason: {reason}"""

_SYNTH_TMPL = """\
You are an expert research assistant. Answer using ONLY the provided context.

Rules:
1. Cite inline as [1], [2], etc. matching the context snippet numbers.
2. If the context does not contain the answer, say: "The knowledge base does not contain information about this topic."
3. Never invent facts. Never extrapolate beyond the context.
4. Use markdown formatting for clarity (headers, bullet points, code blocks).
5. If sources conflict, acknowledge the discrepancy.
6. Use the conversation memory below (if present) to understand follow-up questions.

{format_instructions}

Conversation memory:
{memory}

Context:
{context}

Question: {query}"""


# ---------------------------------------------------------------------------
# LLM runnable wrapper
# ---------------------------------------------------------------------------


def _make_llm(fast: bool = True, temperature: float = 0.0, max_tokens: int = 512):
    def _call(prompt_value: Any) -> str:
        messages = prompt_value.to_messages()
        system = ""
        user_parts: list[str] = []
        for msg in messages:
            role = getattr(msg, "type", "human")
            content = str(getattr(msg, "content", ""))
            if role == "system":
                system = content
            else:
                user_parts.append(content)
        return chat_sync(
            system,
            "\n".join(user_parts),
            fast=fast,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    return _call


# ---------------------------------------------------------------------------
# Chain factories
# ---------------------------------------------------------------------------


def _chain(tmpl: str, model: type[BaseModel], fast: bool, temp: float, tokens: int):
    if not _LC_AVAILABLE:
        return None
    parser = PydanticOutputParser(pydantic_object=model)
    prompt = ChatPromptTemplate.from_template(tmpl).partial(
        format_instructions=parser.get_format_instructions()
    )
    llm = RunnableLambda(_make_llm(fast=fast, temperature=temp, max_tokens=tokens))
    return prompt | llm | parser


def make_router_chain() -> "RunnableSerializable | None":
    return _chain(_ROUTER_TMPL, RouterOutput, fast=True, temp=0.0, tokens=256)


def make_reflect_chain() -> "RunnableSerializable | None":
    return _chain(_REFLECT_TMPL, ReflectOutput, fast=True, temp=0.0, tokens=256)


def make_rewrite_chain() -> "RunnableSerializable | None":
    return _chain(_REWRITE_TMPL, RewriteOutput, fast=True, temp=0.3, tokens=128)


def make_synth_chain() -> "RunnableSerializable | None":
    return _chain(_SYNTH_TMPL, SynthesisOutput, fast=False, temp=0.1, tokens=2048)
