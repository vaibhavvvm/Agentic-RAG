"""
RAG3 General Chat Agent
=========================
Handles greetings and non-retrieval queries with a fast Groq model.
Uses the session's memory context block so the LLM stays coherent
across small-talk turns without the full retrieval pipeline overhead.
"""

from __future__ import annotations

from src.monitoring.logger import get_logger
from src.retrieval.session import RAGSession
from src.utils.llm import chat_sync

log = get_logger(__name__)

SYSTEM_PROMPT = (
    "You are a concise, friendly assistant. The user is chatting casually; "
    "no retrieval is needed. Keep answers under three sentences unless asked."
)


class GeneralAgent:
    """Non-retrieval conversational responder."""

    async def run(self, query: str, session: RAGSession) -> str:
        context = await session.build_context(query, top_k_episodic=2)
        user_msg = (
            f"{context}\n\n[CURRENT TURN]\nUSER: {query}" if context else query
        )
        return chat_sync(
            SYSTEM_PROMPT,
            user_msg,
            fast=True,
            temperature=0.4,
            max_tokens=300,
        )
