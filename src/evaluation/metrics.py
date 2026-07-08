"""
RAG Evaluation Metrics
=========================
Wraps RAGAS (when available) and provides a lightweight local fallback
for the four core RAG metrics:

    faithfulness       — does the answer stick to the retrieved context?
    answer_relevancy   — does the answer actually address the question?
    context_precision  — are the retrieved chunks on-topic?
    context_recall     — are the ground-truth facts represented?

Local fallback is deterministic and depends only on string/embedding
similarity so CI can run it offline.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from src.ingestion.embedder import CachedOllamaEmbedder
from src.monitoring.logger import get_logger

log = get_logger(__name__)

try:  # pragma: no cover
    from ragas import evaluate as _ragas_evaluate  # type: ignore
    from ragas.metrics import (  # type: ignore
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )
    from datasets import Dataset  # type: ignore
    _RAGAS_AVAILABLE = True
except Exception:
    _RAGAS_AVAILABLE = False


@dataclass
class EvalCase:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str = ""

    def to_ragas_row(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "contexts": self.contexts,
            "ground_truth": self.ground_truth,
        }


@dataclass
class EvalReport:
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0
    backend: str = "fallback"
    per_case: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Local fallback
# ---------------------------------------------------------------------------


def _cosine(a: list[float], b: list[float]) -> float:
    import math
    num = sum(x * y for x, y in zip(a, b, strict=False))
    da = math.sqrt(sum(x * x for x in a))
    db = math.sqrt(sum(y * y for y in b))
    if da == 0 or db == 0:
        return 0.0
    return num / (da * db)


def _fallback_eval(
    cases: list[EvalCase], embedder: CachedOllamaEmbedder
) -> EvalReport:
    report = EvalReport(backend="local_fallback")
    if not cases:
        return report

    per_case: list[dict[str, float]] = []
    for case in cases:
        texts = [case.question, case.answer, case.ground_truth or "", *case.contexts]
        vecs = embedder.run(texts=texts)["embeddings"]
        q_vec, a_vec, gt_vec, *ctx_vecs = vecs

        relevancy = _cosine(q_vec, a_vec)

        if ctx_vecs:
            faith_scores = [_cosine(a_vec, cv) for cv in ctx_vecs]
            faith = max(faith_scores)
            precision = sum(1 for s in faith_scores if s >= 0.6) / len(faith_scores)
        else:
            faith = 0.0
            precision = 0.0

        recall = _cosine(gt_vec, a_vec) if case.ground_truth else 0.0

        per_case.append({
            "faithfulness": round(faith, 4),
            "answer_relevancy": round(relevancy, 4),
            "context_precision": round(precision, 4),
            "context_recall": round(recall, 4),
        })

    def _avg(k: str) -> float:
        return sum(c[k] for c in per_case) / len(per_case)

    report.faithfulness = round(_avg("faithfulness"), 4)
    report.answer_relevancy = round(_avg("answer_relevancy"), 4)
    report.context_precision = round(_avg("context_precision"), 4)
    report.context_recall = round(_avg("context_recall"), 4)
    report.per_case = per_case  # type: ignore[assignment]
    return report


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def evaluate(
    cases: list[EvalCase],
    embedder: CachedOllamaEmbedder | None = None,
    use_ragas: bool = True,
) -> EvalReport:
    """
    Evaluate a batch of ``EvalCase`` objects.

    Uses RAGAS when installed and ``use_ragas=True``; otherwise falls
    back to a local embedding-similarity estimator.
    """
    if use_ragas and _RAGAS_AVAILABLE:
        try:  # pragma: no cover
            ds = Dataset.from_list([c.to_ragas_row() for c in cases])
            results = _ragas_evaluate(
                ds,
                metrics=[
                    faithfulness,
                    answer_relevancy,
                    context_precision,
                    context_recall,
                ],
            )
            return EvalReport(
                faithfulness=float(results.get("faithfulness", 0.0)),
                answer_relevancy=float(results.get("answer_relevancy", 0.0)),
                context_precision=float(results.get("context_precision", 0.0)),
                context_recall=float(results.get("context_recall", 0.0)),
                backend="ragas",
            )
        except Exception as exc:
            log.warning(
                "RAGAS evaluation failed; falling back",
                extra={"error": str(exc)},
            )

    return _fallback_eval(cases, embedder or CachedOllamaEmbedder())
