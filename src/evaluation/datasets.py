"""
RAG Evaluation Datasets
==========================
Minimal helpers to load / persist evaluation datasets in a JSONL format:

    {"question": "...", "answer": "...", "contexts": ["...", ...],
     "ground_truth": "..."}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from src.evaluation.metrics import EvalCase


def load_jsonl(path: str | Path) -> list[EvalCase]:
    """Load an evaluation dataset from a JSONL file."""
    p = Path(path)
    cases: list[EvalCase] = []
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            cases.append(
                EvalCase(
                    question=obj.get("question", ""),
                    answer=obj.get("answer", ""),
                    contexts=list(obj.get("contexts", []) or []),
                    ground_truth=obj.get("ground_truth", ""),
                )
            )
    return cases


def save_jsonl(cases: Iterable[EvalCase], path: str | Path) -> None:
    """Persist evaluation cases to a JSONL file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        for c in cases:
            fh.write(json.dumps(c.to_ragas_row()) + "\n")


def example_dataset() -> list[EvalCase]:
    """A tiny built-in dataset useful for smoke tests."""
    return [
        EvalCase(
            question="What is RAG built on?",
            answer="RAG is built on Haystack 2.x with pgvector and Neo4j.",
            contexts=[
                "RAG uses Haystack 2.x pipelines, pgvector for dense search, and Neo4j for graph storage."
            ],
            ground_truth="Haystack 2.x + pgvector + Neo4j",
        ),
    ]
