"""
RAG Turn Summariser
=====================
Condenses batches of old conversation turns into a compact paragraph
that can be stored in FAISS / graph memory without blowing the context
budget.

Called by ``RAGSession`` when the sliding window exceeds its
``summarise_after_turns`` threshold.
"""

from __future__ import annotations

from src.monitoring.logger import get_logger
from src.storage.base import MemoryEntry
from src.utils.llm import chat_sync

log = get_logger(__name__)

SYSTEM_PROMPT = (
    "You compress a chunk of dialogue into 3–5 sentences that preserve "
    "all named entities, numbers, and decisions. Use past tense. Do not "
    "add interpretation — just summarise facts and outcomes."
)


def summarise_turns(turns: list[MemoryEntry], max_tokens: int = 300) -> MemoryEntry:
    """
    Summarise a batch of ``MemoryEntry`` turns into a single summary entry.

    Args:
        turns:       Chronological list of user/assistant turns.
        max_tokens:  Completion cap for the summary.

    Returns:
        A single ``MemoryEntry`` with ``is_summary=True`` carrying the
        compressed text and turn-index range in its metadata.
    """
    if not turns:
        return MemoryEntry(role="system", content="", is_summary=True)

    body_lines = [f"{t.role.upper()}: {t.content}" for t in turns]
    prompt = "Summarise this conversation excerpt:\n\n" + "\n".join(body_lines)

    try:
        summary = chat_sync(
            SYSTEM_PROMPT,
            prompt,
            fast=True,
            temperature=0.0,
            max_tokens=max_tokens,
        )
    except Exception as exc:
        log.warning(
            "Summariser fell back to concatenation", extra={"error": str(exc)}
        )
        summary = " | ".join(body_lines)[: max_tokens * 4]

    first, last = turns[0], turns[-1]
    return MemoryEntry(
        role="system",
        content=summary,
        is_summary=True,
        turn_index=last.turn_index,
        metadata={
            "summarised_turns": len(turns),
            "turn_range": [first.turn_index, last.turn_index],
        },
    )
