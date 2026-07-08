"""
RAG LangGraph Ingestion Pipeline
===================================
Stateful, agentic ingestion workflow implemented as a LangGraph state
machine. Every node is a pure function that mutates a shared
``IngestionState``; the edges encode the step order.

State machine::

    parse ──► crop_images ──► vlm_describe ──► structure_text
       │                                             │
       │                                             ▼
       ▼                                     push_to_minio
    chunk_text                                      │
       │                                             ▼
       └───────────────► embed_and_store ──► push_to_graph ──► END

Falls back to a plain sequential implementation when ``langgraph`` is
not installed, preserving identical semantics.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from haystack.dataclasses import Document

from src.ingestion.docling_parser import DoclingParser, ParsedDocument, ParsedImage
from src.ingestion.vision import VisionProcessor
from src.monitoring.logger import get_logger
from src.storage.base import BaseGraphStore, BaseVectorStore
from src.storage.object_store.minio_store import MinioObjectStore
from src.utils.er_extractor import extract_triples
from src.utils.llm import chat_sync

log = get_logger(__name__)

try:  # pragma: no cover - optional
    from langgraph.graph import END, StateGraph
    _LANGGRAPH_AVAILABLE = True
except Exception:
    StateGraph = None  # type: ignore
    END = "__end__"  # type: ignore
    _LANGGRAPH_AVAILABLE = False


_STRUCTURE_PROMPT = (
    "You are a document-cleanup assistant. The user will paste raw Markdown "
    "extracted from a document. Return a CLEANED version: fix broken headings, "
    "merge split paragraphs, keep tables as valid Markdown. Do not add new "
    "information. Reply with Markdown only."
)


# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------


@dataclass
class IngestionState:
    file_path: Path
    parsed: ParsedDocument | None = None
    markdown_clean: str = ""
    image_urls: list[str] = field(default_factory=list)       # MinIO URLs
    image_summaries: list[str] = field(default_factory=list)  # VLM outputs
    docs_for_vector: list[Document] = field(default_factory=list)
    docs_for_graph: list[Document] = field(default_factory=list)
    stats: dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Dependencies (resolved at construction time so nodes stay pure)
# ---------------------------------------------------------------------------


class _Deps:
    def __init__(
        self,
        parser: DoclingParser,
        vision: VisionProcessor,
        minio: MinioObjectStore,
        embedder: Any,
        vector_store: BaseVectorStore,
        graph_store: BaseGraphStore,
    ) -> None:
        self.parser = parser
        self.vision = vision
        self.minio = minio
        self.embedder = embedder
        self.vector_store = vector_store
        self.graph_store = graph_store


# ---------------------------------------------------------------------------
# Async Node implementations
# ---------------------------------------------------------------------------


def _parse(deps: _Deps):
    async def _node(state: IngestionState) -> IngestionState:
        # docling blocks, so run it in a thread
        state.parsed = await asyncio.to_thread(deps.parser.run, state.file_path)
        state.stats["images_found"] = len(state.parsed.images)
        state.stats["tables_found"] = len(state.parsed.tables)
        return state
    return _node


def _vlm_describe(deps: _Deps):
    async def _node(state: IngestionState) -> IngestionState:
        assert state.parsed is not None
        if not state.parsed.images:
            return state
        png_bytes = [img.bytes_png for img in state.parsed.images]
        state.image_summaries = await asyncio.to_thread(deps.vision.describe_bytes, png_bytes)
        return state
    return _node


def _push_images_to_minio(deps: _Deps):
    async def _node(state: IngestionState) -> IngestionState:
        assert state.parsed is not None
        if not state.parsed.images:
            return state
        await deps.minio.ensure_bucket()
        urls: list[str] = []
        for img in state.parsed.images:
            url = await deps.minio.put_bytes(
                img.bytes_png,
                content_type="image/png",
                metadata={
                    "page": str(img.page),
                    "source": str(state.file_path.name),
                },
            )
            urls.append(url)
        state.image_urls = urls
        return state
    return _node


def _structure_text():
    async def _node(state: IngestionState) -> IngestionState:
        assert state.parsed is not None
        raw_md = state.parsed.markdown
        if not raw_md.strip():
            state.markdown_clean = ""
            return state
        head = raw_md[:8000]
        tail = raw_md[8000:]
        cleaned = await asyncio.to_thread(
            chat_sync,
            _STRUCTURE_PROMPT, head, fast=True, temperature=0.0, max_tokens=2048
        )
        cleaned = cleaned or head
        state.markdown_clean = cleaned + ("\n\n" + tail if tail else "")
        return state
    return _node


def _prepare_docs(deps: _Deps):
    async def _node(state: IngestionState) -> IngestionState:
        assert state.parsed is not None
        source = str(state.file_path.name)
        docs: list[Document] = []

        md = state.markdown_clean or state.parsed.markdown
        paragraphs = [p.strip() for p in md.split("\n\n") if p.strip()]
        for i, para in enumerate(paragraphs):
            docs.append(Document(
                content=para,
                meta={"source_doc_id": source, "chunk_index": i, "kind": "text"},
            ))

        for i, tbl in enumerate(state.parsed.tables):
            docs.append(Document(
                content=tbl,
                meta={"source_doc_id": source, "chunk_index": 10_000 + i, "kind": "table"},
            ))

        for i, (img, summary) in enumerate(
            zip(state.parsed.images, state.image_summaries, strict=False)
        ):
            url = state.image_urls[i] if i < len(state.image_urls) else ""
            docs.append(Document(
                content=summary,
                meta={
                    "source_doc_id": source,
                    "chunk_index": 20_000 + i,
                    "kind": "image_summary",
                    "image_url": url,
                    "page": img.page,
                    "caption": img.caption,
                },
            ))

        state.docs_for_vector = docs
        state.docs_for_graph = [d for d in docs if d.meta.get("kind") == "text"][:50]
        return state
    return _node


def _embed_and_store(deps: _Deps):
    async def _node(state: IngestionState) -> IngestionState:
        docs = state.docs_for_vector
        if not docs:
            return state
        texts = [d.content or "" for d in docs]
        vecs = await asyncio.to_thread(deps.embedder.run, texts=texts)
        vecs = vecs["embeddings"]
        for d, v in zip(docs, vecs, strict=False):
            d.embedding = v
        await deps.vector_store.upsert_documents(docs)
        state.stats["vectors_written"] = len(docs)
        return state
    return _node


def _push_to_graph(deps: _Deps):
    async def _node(state: IngestionState) -> IngestionState:
        n_ep = 0
        n_tri = 0
        for d in state.docs_for_graph:
            episode_id = f"{d.meta.get('source_doc_id','doc')}:{d.meta.get('chunk_index',0)}"
            content = d.content or ""
            try:
                await deps.graph_store.add_episode(
                    content=content,
                    episode_id=episode_id,
                    metadata=dict(d.meta or {}),
                )
                n_ep += 1
            except Exception as exc:
                log.debug("graph episode add failed", extra={"err": str(exc)})

            try:
                triples = await asyncio.to_thread(extract_triples, content)
                if triples:
                    written = await deps.graph_store.add_triples(
                        [t.as_dict() for t in triples],
                        episode_id=episode_id,
                    )
                    n_tri += written
            except Exception as exc:
                log.debug("ER extraction/write failed", extra={"err": str(exc)})

        state.stats["graph_episodes"] = n_ep
        state.stats["graph_triples"] = n_tri
        return state
    return _node


# ---------------------------------------------------------------------------
# Public pipeline
# ---------------------------------------------------------------------------


class LangGraphIngestionPipeline:
    """
    LangGraph-based ingestion orchestrator natively asynchronous.
    """

    def __init__(
        self,
        embedder: Any,
        vector_store: BaseVectorStore,
        graph_store: BaseGraphStore,
        parser: DoclingParser | None = None,
        vision: VisionProcessor | None = None,
        minio: MinioObjectStore | None = None,
    ) -> None:
        self._deps = _Deps(
            parser=parser or DoclingParser(),
            vision=vision or VisionProcessor(),
            minio=minio or MinioObjectStore(),
            embedder=embedder,
            vector_store=vector_store,
            graph_store=graph_store,
        )
        self._graph = self._build() if _LANGGRAPH_AVAILABLE else None
        if not _LANGGRAPH_AVAILABLE:
            log.warning("langgraph not installed — ingestion uses sequential fallback")

    def _build(self):
        g = StateGraph(IngestionState)
        g.add_node("parse", _parse(self._deps))
        g.add_node("vlm_describe", _vlm_describe(self._deps))
        g.add_node("push_images", _push_images_to_minio(self._deps))
        g.add_node("structure", _structure_text())
        g.add_node("prepare_docs", _prepare_docs(self._deps))
        g.add_node("embed_and_store", _embed_and_store(self._deps))
        g.add_node("push_to_graph", _push_to_graph(self._deps))

        g.set_entry_point("parse")
        g.add_edge("parse", "vlm_describe")
        g.add_edge("vlm_describe", "push_images")
        g.add_edge("push_images", "structure")
        g.add_edge("structure", "prepare_docs")
        g.add_edge("prepare_docs", "embed_and_store")
        g.add_edge("embed_and_store", "push_to_graph")
        g.add_edge("push_to_graph", END)
        return g.compile()

    async def arun(self, file_path: Path) -> dict[str, Any]:
        state = IngestionState(file_path=Path(file_path))
        if self._graph is not None:
            out = await self._graph.ainvoke(state)
            stats = (out.get("stats") if isinstance(out, dict) else getattr(out, "stats", {})) or {}
            return dict(stats)

        # Sequential fallback
        state = await _parse(self._deps)(state)
        state = await _vlm_describe(self._deps)(state)
        state = await _push_images_to_minio(self._deps)(state)
        state = await _structure_text()(state)
        state = await _prepare_docs(self._deps)(state)
        state = await _embed_and_store(self._deps)(state)
        state = await _push_to_graph(self._deps)(state)
        return dict(state.stats)
