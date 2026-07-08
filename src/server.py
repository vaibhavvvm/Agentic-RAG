import os
import shutil
import tempfile
import uuid
import json
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional, Dict, List

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.main import RAGSystem
from src.retrieval.session import RAGSession
from src.evaluation.metrics import evaluate as run_eval, EvalCase
from src.evaluation.datasets import load_jsonl
from src.config import get_settings
from src.monitoring.logger import get_logger

log = get_logger(__name__)

# Global instances
system: Optional[RAGSystem] = None
active_sessions: Dict[str, Dict[str, Any]] = {}  # {id: {session, name, turns}}

# Resolve frontend dist path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_FRONTEND_DIST = _PROJECT_ROOT / "frontend" / "dist"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global system
    system = RAGSystem()
    log.info("Starting up RAGSystem inside FastAPI lifespan...")
    await system.startup()
    yield
    log.info("Shutting down RAGSystem inside FastAPI lifespan...")
    for entry in active_sessions.values():
        await entry["session"].flush()
    await system.shutdown()


app = FastAPI(
    title="Agentic RAG API Server",
    description="Backend API for hybrid vector-graph Agentic RAG system",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_headers=["*"],
    allow_methods=["*"],
)

# Mount static assets from Vite build (JS, CSS bundles)
if (_FRONTEND_DIST / "assets").exists():
    app.mount(
        "/assets",
        StaticFiles(directory=str(_FRONTEND_DIST / "assets")),
        name="static-assets",
    )


# ── Pydantic models ──────────────────────────────────────────────────────


class ChatRequest(BaseModel):
    session_id: str
    query: str


class SessionCreateRequest(BaseModel):
    name: Optional[str] = None
    session_id: Optional[str] = None


# ── Frontend serving ─────────────────────────────────────────────────────


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """Serve the React SPA index.html."""
    index_path = _FRONTEND_DIST / "index.html"
    if index_path.exists():
        return HTMLResponse(content=index_path.read_text(encoding="utf-8"))
    # Fallback to docs/index.html
    fallback = _PROJECT_ROOT / "docs" / "index.html"
    if fallback.exists():
        return HTMLResponse(content=fallback.read_text(encoding="utf-8"))
    raise HTTPException(status_code=404, detail="index.html not found. Run 'npm run build' in frontend/.")


# ── Health ───────────────────────────────────────────────────────────────


@app.get("/api/health")
async def health_check():
    """Service status and configuration info."""
    global system
    if not system:
        return {"status": "offline", "error": "RAGSystem not initialized"}

    cfg = get_settings()
    services = {"postgres": "offline", "neo4j": "offline", "minio": "offline"}

    # Verify PostgreSQL
    if system.vector_store and hasattr(system.vector_store, "_pool") and system.vector_store._pool:
        try:
            async with system.vector_store._pool.acquire() as conn:
                await conn.execute("SELECT 1;")
            services["postgres"] = "online"
        except Exception as e:
            services["postgres"] = f"error: {str(e)[:80]}"

    # Verify Neo4j
    if system.graph_store and hasattr(system.graph_store, "_driver") and system.graph_store._driver:
        try:
            async with system.graph_store._driver.session(database=system.graph_store._database) as session:
                await session.run("RETURN 1;")
            services["neo4j"] = "online"
        except Exception as e:
            services["neo4j"] = f"error: {str(e)[:80]}"

    # Verify MinIO
    if system.minio:
        try:
            services["minio"] = "online"
        except Exception as e:
            services["minio"] = f"error: {str(e)[:80]}"

    # Dynamic counts
    counts = {"documents": 0, "chunks": 0, "episodes": 0, "triples": 0}
    try:
        if services["postgres"] == "online":
            counts["chunks"] = await system.vector_store.count_documents()
            async with system.vector_store._pool.acquire() as conn:
                row_docs = await conn.fetchval(
                    "SELECT COUNT(DISTINCT metadata->>'source_doc_id') FROM documents;"
                )
                counts["documents"] = row_docs or 0
        if services["neo4j"] == "online":
            async with system.graph_store._driver.session(database=system.graph_store._database) as session:
                res_ep = await session.run("MATCH (e:Episode) RETURN count(e) as c")
                rec_ep = await res_ep.single()
                counts["episodes"] = rec_ep["c"] if rec_ep else 0
                res_rel = await session.run("MATCH ()-[r:RELATES_TO]->() RETURN count(r) as c")
                rec_rel = await res_rel.single()
                counts["triples"] = rec_rel["c"] if rec_rel else 0
    except Exception as e:
        log.warning(f"Error fetching counts: {e}")

    return {
        "status": "online",
        "config": {
            "app_name": cfg.app_name,
            "app_version": cfg.app_version,
            "environment": cfg.environment,
            "debug": cfg.debug,
        },
        "services": services,
        "counts": counts,
    }


# ── Sessions ─────────────────────────────────────────────────────────────


@app.post("/api/sessions")
async def create_session(req: SessionCreateRequest):
    """Create a new chat session with memory container."""
    global system
    if not system:
        raise HTTPException(status_code=500, detail="System not initialized")

    session_id = req.session_id or str(uuid.uuid4())
    name = req.name or f"Chat {len(active_sessions) + 1}"

    session = RAGSession(
        session_id=session_id,
        embedder=system.embedder,
        graph_store=system.graph_store,
    )
    active_sessions[session_id] = {
        "session": session,
        "name": name,
        "turns": 0,
    }

    return {
        "session_id": session_id,
        "name": name,
        "turns": 0,
    }


@app.get("/api/sessions")
async def list_sessions():
    """List all active chat sessions."""
    return [
        {
            "id": sid,
            "name": entry["name"],
            "turns": entry["turns"],
        }
        for sid, entry in active_sessions.items()
    ]


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    """Flush and destroy a chat session."""
    if session_id in active_sessions:
        entry = active_sessions.pop(session_id)
        await entry["session"].flush()
        return {"status": "deleted"}
    raise HTTPException(status_code=404, detail="Session not found")


# ── Chat ─────────────────────────────────────────────────────────────────


@app.post("/api/chat")
async def chat_completion(req: ChatRequest):
    """Route query through Orchestrator state machine."""
    global system
    if not system:
        raise HTTPException(status_code=500, detail="System not initialized")

    entry = active_sessions.get(req.session_id)
    if not entry:
        # Auto-create session
        session = RAGSession(
            session_id=req.session_id,
            embedder=system.embedder,
            graph_store=system.graph_store,
        )
        entry = {"session": session, "name": f"Chat {req.session_id[:8]}", "turns": 0}
        active_sessions[req.session_id] = entry

    try:
        response = await system.orchestrator.ask(req.query, entry["session"])
        entry["turns"] += 1
        return response.to_dict()
    except Exception as e:
        log.exception("Chat routing failed")
        raise HTTPException(status_code=500, detail=str(e))


# ── Ingestion ────────────────────────────────────────────────────────────


@app.post("/api/ingest")
async def ingest_document(
    file: UploadFile = File(...),
    chunk_strategy: str = Form("semantic"),
):
    """Upload a file and run the ingestion pipeline."""
    global system
    if not system:
        raise HTTPException(status_code=500, detail="System not initialized")

    temp_dir = tempfile.mkdtemp()
    temp_path = Path(temp_dir) / file.filename
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Run async ingestion
        stats = await system.ingest_file(temp_path)
        return {
            "filename": file.filename,
            "status": "success",
            "stats": stats,
        }
    except Exception as e:
        log.exception("Ingestion failed")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_path.exists():
            os.remove(temp_path)
        shutil.rmtree(temp_dir, ignore_errors=True)


# ── Documents ────────────────────────────────────────────────────────────


@app.get("/api/documents")
async def list_documents():
    """List ingested documents and chunk counts."""
    global system
    if not system:
        raise HTTPException(status_code=500, detail="System not initialized")

    docs = []
    try:
        async with system.vector_store._pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT metadata->>'source_doc_id' as name, COUNT(*) as chunks
                FROM documents
                GROUP BY 1
                ORDER BY 1;
            """)
            for r in rows:
                name = r["name"] or "unknown"
                ext = name.split(".")[-1] if "." in name else "txt"
                docs.append({
                    "name": name,
                    "type": ext,
                    "chunks": r["chunks"],
                    "episodes": 0,
                    "triples": 0,
                    "status": "complete",
                })

        # Fetch graph counts
        if system.graph_store and hasattr(system.graph_store, "_driver") and system.graph_store._driver:
            async with system.graph_store._driver.session(database=system.graph_store._database) as session:
                for doc in docs:
                    res_ep = await session.run(
                        "MATCH (e:Episode) WHERE e.id STARTS WITH $prefix RETURN count(e) as c",
                        {"prefix": f"{doc['name']}:"},
                    )
                    rec_ep = await res_ep.single()
                    doc["episodes"] = rec_ep["c"] if rec_ep else 0
    except Exception as e:
        log.warning(f"Failed to fetch documents: {e}")
    return docs


@app.delete("/api/documents/{name}")
async def delete_document(name: str):
    """Remove a document from vector and graph stores."""
    global system
    if not system:
        raise HTTPException(status_code=500, detail="System not initialized")

    try:
        async with system.vector_store._pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM documents WHERE metadata->>'source_doc_id' = $1;", name
            )

        if system.graph_store and hasattr(system.graph_store, "_driver") and system.graph_store._driver:
            async with system.graph_store._driver.session(database=system.graph_store._database) as session:
                await session.run(
                    "MATCH (e:Episode) WHERE e.id STARTS WITH $prefix DETACH DELETE e;",
                    {"prefix": f"{name}:"},
                )
                await session.run(
                    "MATCH (n:Entity) WHERE n.source_doc_id = $name DETACH DELETE n;",
                    {"name": name},
                )

        return {"status": "success", "message": f"Document '{name}' deleted"}
    except Exception as e:
        log.exception(f"Failed to delete document: {name}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Evaluation ───────────────────────────────────────────────────────────


@app.post("/api/evaluate")
async def run_evaluation_api(
    dataset: UploadFile = File(...),
    backend: str = Form("auto"),
):
    """Run RAGAS / Local fallback evaluation."""
    global system
    if not system:
        raise HTTPException(status_code=500, detail="System not initialized")

    temp_dir = tempfile.mkdtemp()
    temp_path = Path(temp_dir) / dataset.filename
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(dataset.file, buffer)

        cases = load_jsonl(temp_path)
        eval_cases = [
            EvalCase(
                question=c.get("question", ""),
                answer=c.get("answer", ""),
                contexts=c.get("contexts", []),
                ground_truth=c.get("ground_truth", ""),
            )
            for c in cases
        ]

        use_ragas = backend in {"auto", "ragas"}
        report = await asyncio.to_thread(
            run_eval, eval_cases, embedder=system.embedder, use_ragas=use_ragas
        )
        return report.to_dict()
    except Exception as e:
        log.exception("Evaluation run failed")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_path.exists():
            os.remove(temp_path)
        shutil.rmtree(temp_dir, ignore_errors=True)


# ── SPA Fallback (must be last) ──────────────────────────────────────────


@app.get("/{full_path:path}")
async def spa_fallback(full_path: str):
    """Serve static files or fall back to index.html for client-side routing."""
    # Try to serve the file directly from dist
    file_path = _FRONTEND_DIST / full_path
    if file_path.is_file():
        return FileResponse(str(file_path))
    # Fall back to index.html for SPA routes
    index_path = _FRONTEND_DIST / "index.html"
    if index_path.exists():
        return HTMLResponse(content=index_path.read_text(encoding="utf-8"))
    raise HTTPException(status_code=404, detail="Not found")
