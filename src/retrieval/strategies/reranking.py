"""
RAG Cross-Encoder Reranker
============================
Haystack 2.x ``@component`` that rescores ``(query, passage)`` pairs
using the local **bge-reranker-v2-m3** checkpoint at
``D:\\MODELS\\bge-reranker-v2-m3`` (HuggingFace ``AutoModelForSequence
Classification``) and returns the top-N documents.

Two backends are supported (selected by ``RERANKER__BACKEND`` env or
config):

  * ``local_hf``  — loads the HF checkpoint once per process; fastest
                    option when the model is on local disk.  Uses GPU if
                    ``RERANKER__DEVICE=cuda`` or auto-detect succeeds.
  * ``ollama``    — legacy path; issues an HTTP call per pair against a
                    running Ollama server that has pulled the reranker.

If neither backend is available the component falls back to a simple
lexical overlap score so the pipeline keeps working in stripped-down
environments / CI.
"""

from __future__ import annotations

import asyncio
from typing import Any, ClassVar

import httpx
from haystack import component
from haystack.dataclasses import Document

from src.config import get_settings
from src.monitoring.logger import get_logger, timed_operation
from src.monitoring.metrics import MetricsCollector

log = get_logger(__name__)


@component
class OllamaRanker:  # name preserved for back-compat across callers
    """
    Cross-encoder reranker. Despite the legacy class name, it now defaults
    to a **local HuggingFace** backend using the bge-reranker-v2-m3
    checkpoint on disk. Pass ``backend="ollama"`` to use the HTTP path.
    """

    OUTPUT_TYPES: ClassVar[dict[str, type]] = {"documents": list}

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        top_k: int | None = None,
        concurrency: int = 4,
        score_key: str = "rerank_score",
        backend: str | None = None,
    ) -> None:
        cfg = get_settings()
        self.backend = (backend or cfg.reranker.backend).lower()
        self.model: str = model or cfg.ollama.reranker_model
        self.base_url: str = (base_url or str(cfg.ollama.base_url)).rstrip("/")
        self.top_k: int = top_k or cfg.retrieval.top_k_rerank
        self.concurrency: int = max(1, int(concurrency))
        self.score_key: str = score_key
        self._timeout: float = float(cfg.ollama.timeout)
        self._metrics = MetricsCollector.get_instance()

        self._hf_model = None
        self._hf_tokenizer = None
        self._hf_device = "cpu"

        if self.backend == "local_hf":
            self._load_hf(cfg)

    # ------------------------------------------------------------------
    # Backend loaders
    # ------------------------------------------------------------------

    def _load_hf(self, cfg) -> None:
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except Exception as exc:
            log.warning(
                "transformers/torch not installed — reranker will use lexical fallback",
                extra={"error": str(exc)},
            )
            self.backend = "lexical"
            return

        path = str(cfg.reranker.model_path)
        device = cfg.reranker.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            self._hf_tokenizer = AutoTokenizer.from_pretrained(path)
            self._hf_model = (
                AutoModelForSequenceClassification.from_pretrained(path).to(device).eval()
            )
            self._hf_device = device
            log.info(
                "local HF reranker loaded",
                extra={"path": path, "device": device},
            )
        except Exception as exc:
            log.error(
                "failed to load local HF reranker — falling back to lexical",
                extra={"path": path, "error": str(exc)},
            )
            self.backend = "lexical"

    # ------------------------------------------------------------------
    # Haystack run
    # ------------------------------------------------------------------

    @component.output_types(documents=list)
    def run(
        self,
        query: str,
        documents: list[Document],
        top_k: int | None = None,
    ) -> dict[str, list[Document]]:
        if not documents:
            return {"documents": []}
        k = top_k or self.top_k

        with timed_operation(log, "reranker.score", count=len(documents)):
            if self.backend == "local_hf" and self._hf_model is not None:
                scores = self._score_hf(query, [d.content or "" for d in documents])
            elif self.backend == "ollama":
                scores = asyncio.run(self._score_all_ollama(query, documents))
            else:
                scores = [self._lexical(query, d.content or "") for d in documents]

        for doc, score in zip(documents, scores, strict=False):
            meta = dict(doc.meta or {})
            meta[self.score_key] = float(score)
            doc.meta = meta

        ranked = sorted(
            documents, key=lambda d: d.meta.get(self.score_key, 0.0), reverse=True
        )[:k]

        self._metrics.record_event("reranker.calls", count=len(documents))
        return {"documents": ranked}

    # ------------------------------------------------------------------
    # Local HF scoring (batched)
    # ------------------------------------------------------------------

    def _score_hf(self, query: str, passages: list[str]) -> list[float]:
        import torch
        cfg = get_settings().reranker
        pairs = [(query, p) for p in passages]
        all_scores: list[float] = []

        for i in range(0, len(pairs), cfg.batch_size):
            batch = pairs[i : i + cfg.batch_size]
            enc = self._hf_tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=cfg.max_length,
                return_tensors="pt",
            ).to(self._hf_device)
            with torch.no_grad():
                logits = self._hf_model(**enc, return_dict=True).logits.view(-1)
                # BGE reranker outputs a single relevance logit; sigmoid squashes to [0,1]
                scores = torch.sigmoid(logits).detach().cpu().tolist()
            all_scores.extend(float(s) for s in scores)
        return all_scores

    # ------------------------------------------------------------------
    # Legacy Ollama scoring
    # ------------------------------------------------------------------

    async def _score_all_ollama(
        self, query: str, documents: list[Document]
    ) -> list[float]:
        sem = asyncio.Semaphore(self.concurrency)
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            async def one(doc: Document) -> float:
                async with sem:
                    return await self._score_pair_ollama(client, query, doc.content or "")
            return await asyncio.gather(*(one(d) for d in documents))

    async def _score_pair_ollama(
        self, client: httpx.AsyncClient, query: str, passage: str
    ) -> float:
        payload = {
            "model": self.model,
            "prompt": f"Query: {query}\nPassage: {passage}\nRelevance:",
            "stream": False,
            "options": {"temperature": 0.0},
        }
        try:
            r = await client.post(f"{self.base_url}/api/generate", json=payload)
            r.raise_for_status()
            data = r.json()
        except Exception as exc:
            log.warning("Ollama reranker call failed", extra={"error": str(exc)})
            self._metrics.record_event("reranker.errors")
            return 0.0
        raw = (data.get("response") or "").strip()
        for tok in raw.replace(",", " ").split():
            try:
                return max(0.0, min(1.0, float(tok)))
            except ValueError:
                continue
        return 0.0

    # ------------------------------------------------------------------
    # Fallback: lexical overlap
    # ------------------------------------------------------------------

    @staticmethod
    def _lexical(query: str, passage: str) -> float:
        q = {t.lower() for t in query.split() if len(t) > 2}
        p = {t.lower() for t in passage.split() if len(t) > 2}
        if not q or not p:
            return 0.0
        return len(q & p) / max(1, len(q))
