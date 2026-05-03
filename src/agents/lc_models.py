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
Classify the user query for a RAG system. Pick ONE intent:
- "general_chat": greetings, small talk, meta questions
- "vector_retrieval": factual lookup, definitions, descriptions
- "graph_retrieval": relational / multi-hop questions
- "hybrid_retrieval": mix of factual and relational

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
Answer using ONLY the provided context snippets. Cite inline as [1], [2] etc.
If the answer is not in the context, say so clearly. Do not invent facts.

{format_instructions}

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
    return _chain(_SYNTH_TMPL, SynthesisOutput, fast=False, temp=0.1, tokens=1024)
