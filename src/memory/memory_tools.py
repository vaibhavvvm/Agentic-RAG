"""
RAG Memory Context Builders
==============================
Helper functions that format ``MemoryEntry`` lists into prompt-ready
strings for the LLM.  Kept separate from the store classes so prompt
layout can evolve without touching persistence logic.
"""

from __future__ import annotations

from src.storage.base import MemoryEntry


def format_window(entries: list[MemoryEntry], max_chars: int = 3000) -> str:
    """
    Render the sliding window as a plain-text transcript, newest last.

    Truncates from the front if the transcript exceeds ``max_chars``.
    """
    if not entries:
        return ""
    lines = [f"{e.role.upper()}: {e.content}" for e in entries]
    text = "\n".join(lines)
    if len(text) <= max_chars:
        return text
    return "...\n" + text[-max_chars:]


def format_episodic(entries: list[MemoryEntry], max_entries: int = 5) -> str:
    """Render episodic recall as a bulleted list of summary sentences."""
    if not entries:
        return ""
    picked = entries[:max_entries]
    return "\n".join(f"- {e.content}" for e in picked)


def build_context_block(
    window: list[MemoryEntry],
    episodic: list[MemoryEntry],
    *,
    window_chars: int = 3000,
    max_episodic: int = 5,
) -> str:
    """
    Assemble the full memory context block used as a system-prompt prefix.

    Layout:
        [RECENT CONVERSATION]
        <window>

        [RELEVANT HISTORY]
        - <episodic 1>
        - <episodic 2>
    """
    parts: list[str] = []
    if window:
        parts.append("[RECENT CONVERSATION]\n" + format_window(window, window_chars))
    if episodic:
        parts.append(
            "[RELEVANT HISTORY]\n" + format_episodic(episodic, max_episodic)
        )
    return "\n\n".join(parts)
