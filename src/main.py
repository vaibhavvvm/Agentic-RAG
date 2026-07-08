"""
RAG CLI + System Bootstrap
=============================
Entry point that wires together every layer into a ``RAGSystem`` and
exposes ``chat``, ``ingest``, ``evaluate``, and ``healthcheck`` commands
via Typer. Pluggable backends are selected by command-line flags:

    --graph-backend   neo4j | falkor | pggraph | none   (default: neo4j)
    --vector-backend  pgvector | memory | auto          (default: auto)

The ingest command runs the **LangGraph ingestion pipeline**
(docling → VLM → MinIO → Postgres + graph store).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.markdown import Markdown

from src.agents.orchestrator import Orchestrator
from src.config import get_settings
from src.evaluation.datasets import load_jsonl
from src.evaluation.metrics import evaluate as run_eval
from src.ingestion.embedder import CachedOllamaEmbedder
from src.ingestion.langgraph_pipeline import LangGraphIngestionPipeline
from src.monitoring.langsmith import setup_langsmith
from src.monitoring.logger import get_logger
from src.retrieval.session import RAGSession
from src.storage.base import BaseGraphStore, BaseVectorStore
from src.storage.graph.factory import build_graph_store
from src.storage.object_store.minio_store import MinioObjectStore
from src.storage.postgres.summary_store import PgSummaryStore
from src.storage.vector.factory import build_vector_store

log = get_logger(__name__)
setup_langsmith()  # no-op when LANGSMITH_TRACING_ENABLED=false (default)
app = typer.Typer(add_completion=False, help="RAG — Agentic Retrieval-Augmented Generation")
console = Console()


# ---------------------------------------------------------------------------
# RAGSystem — one-stop initialisation
# ---------------------------------------------------------------------------


class RAGSystem:
    """
    Owns all persistent resources. Backends are chosen at construction
    time; ``startup()`` honours the fallback chains in the factories.
    """

    def __init__(
        self,
        graph_backend: str | None = None,
        vector_backend: str | None = None,
    ) -> None:
        self.settings = get_settings()
        self._graph_pref = graph_backend
        self._vector_pref = vector_backend

        self.embedder = CachedOllamaEmbedder()
        self.summary_store = PgSummaryStore()
        self.minio = MinioObjectStore()

        self.vector_store: BaseVectorStore | None = None
        self.graph_store: BaseGraphStore | None = None
        self._orchestrator: Orchestrator | None = None

    async def startup(self) -> None:
        self.vector_store = await build_vector_store(self._vector_pref)
        self.graph_store = await build_graph_store(self._graph_pref)
        await self.summary_store.initialise()
        await self.minio.ensure_bucket()
        self._orchestrator = Orchestrator(
            vector_store=self.vector_store,
            graph_store=self.graph_store,
            embedder=self.embedder,
        )
        log.info(
            "RAGSystem startup complete",
            extra={
                "graph_backend": type(self.graph_store).__name__,
                "vector_backend": type(self.vector_store).__name__,
            },
        )

    async def shutdown(self) -> None:
        coros = [self.summary_store.close(), self.minio.close()]
        if self.vector_store:
            coros.append(self.vector_store.close())
        if self.graph_store:
            coros.append(self.graph_store.close())
        await asyncio.gather(*coros, return_exceptions=True)

    @property
    def orchestrator(self) -> Orchestrator:
        if self._orchestrator is None:
            raise RuntimeError("Call startup() first.")
        return self._orchestrator

    async def ingest_file(self, file_path: Path) -> dict[str, int]:
        """Run the LangGraph ingestion pipeline for a single file."""
        assert self.vector_store is not None and self.graph_store is not None
        pipeline = LangGraphIngestionPipeline(
            embedder=self.embedder,
            vector_store=self.vector_store,
            graph_store=self.graph_store,
            minio=self.minio,
        )
        return await pipeline.arun(file_path)


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------

# Shared options (replicated on each command — Typer doesn't support
# global options easily for multi-command apps).
_GraphOpt = typer.Option(
    None,
    "--graph-backend", "-g",
    help="Graph backend: neo4j | falkor | pggraph | none (default from env/config).",
)
_VectorOpt = typer.Option(
    None,
    "--vector-backend", "-v",
    help="Vector backend: pgvector | memory | auto (default from env/config).",
)


@app.command()
def chat(
    session_id: Optional[str] = typer.Option(None, help="Reuse a session id"),
    graph_backend: Optional[str] = _GraphOpt,
    vector_backend: Optional[str] = _VectorOpt,
) -> None:
    """Interactive REPL chat."""
    asyncio.run(_chat_loop(session_id, graph_backend, vector_backend))


async def _chat_loop(
    session_id: str | None,
    graph_backend: str | None,
    vector_backend: str | None,
) -> None:
    system = RAGSystem(graph_backend=graph_backend, vector_backend=vector_backend)
    await system.startup()
    session = RAGSession(
        session_id=session_id,
        embedder=system.embedder,
        graph_store=system.graph_store,
    )

    console.print(
        f"[bold green]RAG session {session.session_id} "
        f"(graph={type(system.graph_store).__name__}, "
        f"vector={type(system.vector_store).__name__}) — Ctrl+C to exit[/bold green]"
    )
    try:
        while True:
            try:
                query = console.input("[bold cyan]you >[/bold cyan] ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not query:
                continue
            response = await system.orchestrator.ask(query, session)
            console.print("[bold magenta]rag >[/bold magenta]")
            console.print(Markdown(response.answer or "_(no answer)_"))
            if response.sources:
                img_urls = [
                    s.get("image_url") for s in response.sources
                    if isinstance(s, dict) and s.get("image_url")
                ]
                if img_urls:
                    console.print(f"[dim]images: {img_urls}[/dim]")
                console.print(
                    f"[dim]sources: {len(response.sources)} — intent: "
                    f"{response.metadata.get('intent')}[/dim]"
                )
    finally:
        await session.flush()
        await system.shutdown()


@app.command()
def ingest(
    path: Path = typer.Argument(..., exists=True, readable=True),
    graph_backend: Optional[str] = _GraphOpt,
    vector_backend: Optional[str] = _VectorOpt,
    recursive: bool = typer.Option(False, "--recursive", "-r"),
) -> None:
    """Ingest a document (or directory) via the LangGraph pipeline."""
    asyncio.run(_ingest(path, graph_backend, vector_backend, recursive))


async def _ingest(
    path: Path, graph_backend: str | None, vector_backend: str | None, recursive: bool
) -> None:
    system = RAGSystem(graph_backend=graph_backend, vector_backend=vector_backend)
    await system.startup()
    try:
        files: list[Path]
        if path.is_dir():
            globber = path.rglob if recursive else path.glob
            files = [
                p for p in globber("*")
                if p.is_file() and p.suffix.lower() in {".pdf", ".md", ".txt", ".docx", ".html"}
            ]
        else:
            files = [path]

        for f in files:
            stats = system.ingest_file(f)
            console.print(f"[green]Ingested[/green] {f.name}: {stats}")
    finally:
        await system.shutdown()


@app.command()
def evaluate(
    dataset: Path = typer.Argument(..., exists=True, readable=True),
    output: Optional[Path] = typer.Option(None, help="Write JSON report to this path"),
) -> None:
    """Evaluate a JSONL dataset against the ingested corpus."""
    cases = load_jsonl(dataset)
    report = run_eval(cases)
    payload = report.to_dict()
    console.print_json(data=payload)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        console.print(f"[green]Report written to[/green] {output}")


@app.command()
def healthcheck(
    graph_backend: Optional[str] = _GraphOpt,
    vector_backend: Optional[str] = _VectorOpt,
) -> None:
    """Verify config + service connectivity."""
    asyncio.run(_healthcheck(graph_backend, vector_backend))


async def _healthcheck(graph_backend: str | None, vector_backend: str | None) -> None:
    cfg = get_settings()
    console.print(f"[bold]Config[/bold] — app={cfg.app_name} v{cfg.app_version}")
    console.print(f"  graph_backend (cfg):  {cfg.graph_backend}")
    console.print(f"  vector_backend (cfg): {cfg.vector_backend}")
    console.print(f"  groq keys:            {len(cfg.groq.api_keys)}")
    console.print(f"  openrouter configured: {cfg.openrouter.api_key is not None}")
    console.print(f"  postgres:             {cfg.postgres.host}:{cfg.postgres.port}/{cfg.postgres.db}")
    console.print(f"  neo4j:                {cfg.neo4j.uri}")
    console.print(f"  falkor:               {cfg.falkor.host}:{cfg.falkor.port}")
    console.print(f"  minio:                {cfg.minio.endpoint} / bucket={cfg.minio.bucket}")
    console.print(f"  reranker backend:     {cfg.reranker.backend} @ {cfg.reranker.model_path}")

    system = RAGSystem(graph_backend=graph_backend, vector_backend=vector_backend)
    try:
        await system.startup()
        console.print(
            f"[green]Services up[/green] — "
            f"vector={type(system.vector_store).__name__}, "
            f"graph={type(system.graph_store).__name__}"
        )
    except Exception as exc:
        console.print(f"[red]Startup failed:[/red] {exc}")
    finally:
        await system.shutdown()


if __name__ == "__main__":
    app()
