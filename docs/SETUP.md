# RAG — Setup Guide & Pipeline Flow

> **Version:** 1.0.0 &nbsp;|&nbsp; **Python:** 3.11+ &nbsp;|&nbsp; **Framework:** Haystack 2.x + LangGraph + custom ReAct

A production-grade **agentic Retrieval-Augmented Generation** system combining hybrid vector search (pgvector + BM25), a Neo4j/Graphiti knowledge graph, LangGraph state-machine workers, a Haystack-style ReAct retrieval planner, episodic FAISS memory, and CRAG-style self-reflection.

---

## Table of Contents

1. [Architecture at a Glance](#1-architecture-at-a-glance)
2. [Pipeline Flow Diagram](#2-pipeline-flow-diagram)
3. [Prerequisites](#3-prerequisites)
4. [Installation](#4-installation)
5. [External Services](#5-external-services)
6. [Environment Configuration](#6-environment-configuration)
7. [Phase-by-Phase Module Map](#7-phase-by-phase-module-map)
8. [Pipeline Flow — Narrative Walkthrough](#8-pipeline-flow--narrative-walkthrough)
9. [Running the System](#9-running-the-system)
10. [Evaluation](#10-evaluation)
11. [Operational Tips](#11-operational-tips)

---

## 1. Architecture at a Glance

| Layer | Technology | Role |
|-------|-----------|------|
| Orchestration | Haystack 2.x + LangGraph | Pipelines, state-machine workers |
| Primary LLM | Groq (Llama 3 70B / 8B) | Generation + routing + grading |
| Embeddings | Ollama `nomic-embed-text` | 768-d dense vectors |
| Reranker | Ollama `bge-reranker-v2-m3` | Cross-encoder precision boost |
| Vision | Ollama `llava:13b` | Image → text for multimodal docs |
| Vector Store | PostgreSQL 16 + pgvector (HNSW) | Hybrid BM25 + dense search |
| Graph Store | Neo4j 5 + Graphiti | Entity/relation KG, multi-hop |
| Memory | FAISS IndexFlatIP + Postgres | Sliding window + episodic recall |
| Parsing | `unstructured` hi_res | PDF/DOCX/HTML/images with OCR |
| Evaluation | RAGAS (+ local fallback) | Faithfulness, relevancy, recall |
| CLI | Typer + Rich | `chat` / `ingest` / `evaluate` / `healthcheck` |

---

## 2. Pipeline Flow Diagram

The full pipeline is visualised in `docs/pipeline_flow.svg`:

![RAG Pipeline](./pipeline_flow.svg)

The SVG is divided into four stages:

1. **Ingestion Pipeline** — parse → chunk → embed → write to pgvector + Neo4j + summary store.
2. **Session + Routing** — memory lookup → 3-tier intent router → worker dispatch.
3. **Retrieval Strategies** — query expansion → hybrid search → rerank → self-reflect.
4. **Output** — synthesise response → persist turn + episodic memory → emit metrics.

Colour legend (from the SVG):

| Colour | Type |
|--------|------|
| Slate  | Component / agent |
| Blue   | Retrieval tool |
| Purple | LLM call |
| Amber  | Vector/summary store |
| Emerald| Graph store |
| Pink   | Memory |
| Red    | Decision / router |

---

## 3. Prerequisites

| Requirement | Version |
|-------------|---------|
| Python | 3.11+ |
| PostgreSQL | 16+ with `pgvector` 0.7+ |
| Neo4j | 5.x with APOC |
| Ollama | latest (for embed/rerank/vision) |
| Groq API key | required (1 or more keys for rotation) |
| Optional: Docker | for Postgres/Neo4j containers |

---

## 4. Installation

```bash
# 1. Clone & enter
cd D:/Vaibhav/RAG

# 2. Create venv
python -m venv .venv
.venv\Scripts\activate       # Windows
# source .venv/bin/activate   # Linux/macOS

# 3. Install project (editable)
pip install -e .

# Optional extras:
pip install langgraph>=0.2.0      # GraphAgent state machine
pip install graphiti-core          # automated entity/relation extraction
pip install faiss-cpu              # episodic memory
pip install ragas                  # evaluation metrics
```

Every optional dep is guarded by a try/except in code — the system degrades gracefully (sequential fallback for LangGraph, regex fallback for Graphiti, numpy fallback for FAISS, local-eval fallback for RAGAS).

---

## 5. External Services

### 5.1 Ollama

```bash
ollama pull nomic-embed-text
ollama pull bge-reranker-v2-m3
ollama pull llava:13b
ollama serve                       # http://localhost:11434
```

### 5.2 PostgreSQL + pgvector

```sql
CREATE DATABASE rag;
\c rag
CREATE EXTENSION IF NOT EXISTS vector;
-- schema is auto-created by PgVectorStore.initialize()
```

### 5.3 Neo4j

```bash
# docker-compose or desktop; enable APOC plugin
# default bolt://localhost:7687, user=neo4j
```

---

## 6. Environment Configuration

Copy `.env.example` → `.env` and fill in:

```ini
# --- LLM ---
GROQ_API_KEYS=gsk_...,gsk_...       # comma-separated for rotation
LLM__PRIMARY_MODEL=llama-3.3-70b-versatile
LLM__FAST_MODEL=llama-3.1-8b-instant

# --- Embedding / Rerank / Vision ---
OLLAMA__BASE_URL=http://localhost:11434
OLLAMA__EMBED_MODEL=nomic-embed-text
OLLAMA__RERANK_MODEL=bge-reranker-v2-m3
OLLAMA__VISION_MODEL=llava:13b

# --- Postgres ---
POSTGRES__HOST=localhost
POSTGRES__PORT=5432
POSTGRES__DB=rag
POSTGRES__USER=postgres
POSTGRES__PASSWORD=postgres

# --- Neo4j ---
NEO4J__URI=bolt://localhost:7687
NEO4J__USER=neo4j
NEO4J__PASSWORD=neo4j

# --- Retrieval ---
RETRIEVAL__TOP_K=20
RETRIEVAL__RERANK_TOP_K=8
RETRIEVAL__VECTOR_WEIGHT=0.6
RETRIEVAL__MAX_REFLECTION_ROUNDS=1
```

Nested settings use the `__` delimiter (Pydantic v2 `BaseSettings`).

---

## 7. Phase-by-Phase Module Map

### Phase 1 — Foundation
- `src/config.py` — Pydantic settings + `get_settings()`
- `src/monitoring/logger.py` — structlog + request/session context vars
- `src/monitoring/metrics.py` — `MetricsCollector` (latency, token, cache)
- `src/utils/groq_client.py` — `RotatableGroqGenerator` with key rotation + backoff
- `src/utils/llm.py` — `chat_sync()` / `chat_json()` helpers (fast vs primary)
- `src/storage/base.py` — abstract `BaseVectorStore` / `BaseGraphStore` / `BaseSummaryStore`

### Phase 2 — Ingestion
- `src/ingestion/parser.py` — `DocumentParser` (unstructured hi_res)
- `src/ingestion/vision.py` — `VisionDescriber` (llava)
- `src/ingestion/tables.py` — `TableExtractor` (camelot/pdfplumber)
- `src/ingestion/chunking/{semantic,contextual,hierarchical}.py`
- `src/ingestion/embedders.py` — Ollama embedder wrapper
- `src/ingestion/pipeline.py` — Haystack `Pipeline` wiring

### Phase 3 — Storage & Retrieval Primitives
- `src/storage/postgres/vector_store.py` — hybrid search with `_rrf_fuse()`
- `src/storage/postgres/summary_store.py` — doc/section summaries
- `src/storage/graph/neo4j_store.py` — Graphiti + BFS subgraph expansion
- `src/retrieval/strategies/reranking.py` — `OllamaRanker` cross-encoder

### Phase 4 — Strategies, Tools, Session
- `src/memory/{summarizer,vector_store,memory_tools}.py`
- `src/retrieval/cache.py` — `TTLCache` (OrderedDict LRU + TTL)
- `src/retrieval/strategies/query_expansion.py` — `QueryExpander`
- `src/retrieval/strategies/self_reflection.py` — CRAG-style grader
- `src/retrieval/strategies/query_router.py` — regex+LLM router
- `src/retrieval/strategies/summary_index.py` — summary-first retrieval
- `src/retrieval/strategies/graph_fusion.py` — parallel vector+graph RRF
- `src/retrieval/fallback.py` — `ProgressiveFallback` stages + gates
- `src/retrieval/tools/vector_tool.py` — `VectorSearchTool` facade
- `src/retrieval/tools/graph_search_tool.py` — `GraphSearchTool` facade
- `src/retrieval/session.py` — `RAGSession` (window + FAISS write-back)
- `src/retrieval/agent.py` — Haystack-style **ReAct** retrieval planner

### Phase 5 — Agents, Synthesis, CLI, Evaluation
- `src/agents/router.py` — 3-tier `IntentRouter` (regex → keyword → LLM)
- `src/agents/synthesizer.py` — intent-specific response composer
- `src/agents/workers/general_agent.py` — small-talk fast path
- `src/agents/workers/vector_agent.py` — wraps `AdvancedRAGAgent` (inner loop)
- `src/agents/workers/graph_agent.py` — **LangGraph** state machine
- `src/agents/orchestrator.py` — owns stores, builds tools, dispatches
- `src/evaluation/{metrics,datasets}.py` — RAGAS + local fallback
- `src/main.py` — Typer CLI: `chat / ingest / evaluate / healthcheck`

---

## 8. Pipeline Flow — Narrative Walkthrough

### Stage ① Ingestion

```
file → DocumentParser ─┬─► VisionDescriber (images)
                       ├─► TableExtractor  (tables)
                       └─► text blocks
                              │
                              ▼
         SemanticChunker → ContextualChunker → HierarchicalChunker
                              │
          ┌───────────────────┼──────────────────┐
          ▼                   ▼                  ▼
    PgVectorStore      PgSummaryStore     Neo4jGraphStore
    (HNSW + tsv)      (doc/section)     (entities + facts)
```

### Stage ② Session + Routing

```
User Query
    │
    ▼
RAGSession.build_context()       (sliding window + FAISS recall)
    │
    ▼
IntentRouter (Tier 1: regex) ──► general_chat / graph / vector
    │ (low-confidence)
    ▼
IntentRouter (Tier 2: keywords)
    │ (still uncertain)
    ▼
IntentRouter (Tier 3: LLM arbiter)
    │
    ▼
Dispatch ─┬─► GeneralAgent
          ├─► VectorAgent  (AdvancedRAGAgent loop)
          ├─► GraphAgent   (LangGraph state machine)
          └─► RetrievalAgent (ReAct planner for mixed queries)
```

### Stage ③ Retrieval Strategies

**VectorAgent path:**
```
QueryExpander → [original, paraphrase, HyDE]
      │
      ▼
PgVectorStore.hybrid_search (dense + BM25, RRF fused)
      │
      ▼
OllamaRanker (cross-encoder rerank, top_k)
      │
      ▼
SelfReflection (relevance / sufficiency / faithfulness)
      │              └─► if reject & retries left: broaden & retry
      ▼
TTLCache.put  →  Documents to LLM answer prompt
```

**GraphAgent path (LangGraph):**
```
entity_extract → graph_search → grade ─accept─► answer → END
                                   │
                                   └─expand─► fact_expand ─► grade
```

**RetrievalAgent (ReAct):**
```
LLM plan →  ACTION: vector_search | graph_search | fusion_search
         →  OBSERVATION: top snippets
         →  (loop ≤ max_steps)  → FINAL: done
```

### Stage ④ Output

```
answer + docs
    │
    ▼
Synthesiser (intent-specific)
  ├ general_chat      → {answer}
  ├ vector_retrieval  → {answer, sources[1..N] with snippet+page+score}
  ├ graph_retrieval   → {answer, facts[], entities[]}
  └ hybrid_retrieval  → {answer, sources[], facts[], entities[]}
    │
    ▼
RAGSession.add_turn() → episodic FAISS write-back
    │
    ▼
MetricsCollector (latency, tokens, cache hit, route)
```

---

## 9. Running the System

```bash
# health check (pings Postgres, Neo4j, Ollama, Groq)
python -m src.main healthcheck

# ingest one file
python -m src.main ingest D:/docs/paper.pdf

# ingest a directory
python -m src.main ingest D:/docs --recursive

# interactive chat (Rich UI)
python -m src.main chat

# one-shot chat
python -m src.main chat --query "Explain HNSW vs IVF-Flat"

# evaluate on a JSONL eval set
python -m src.main evaluate eval/questions.jsonl
```

---

## 10. Evaluation

`src/evaluation/metrics.py` runs each `EvalCase` through the orchestrator and scores:

- **faithfulness** — answer grounded in retrieved docs
- **answer_relevancy** — answer addresses the question
- **context_precision / recall** — retrieval quality

If `ragas` is installed it's used end-to-end; otherwise `_fallback_eval()` computes cosine-similarity-based approximations via the Ollama embedder.

Dataset format (`eval/questions.jsonl`):

```json
{"question": "...", "ground_truth": "...", "contexts": ["..."]}
```

---

## 11. Operational Tips

- **Groq rate limits:** set multiple keys in `GROQ_API_KEYS`; `RotatableGroqGenerator` rotates on 429.
- **Cache warmup:** `TTLCache` is per-process; stable prompts benefit most.
- **Reflection loop:** set `RETRIEVAL__MAX_REFLECTION_ROUNDS=0` to disable CRAG retries during latency testing.
- **LangGraph missing:** `GraphAgent` auto-falls back to a sequential implementation with identical semantics.
- **Graphiti missing:** `Neo4jGraphStore` falls back to regex keyword extraction for entity creation.
- **FAISS missing:** `EpisodicMemoryStore` falls back to numpy brute-force cosine search.
- **Inspect a run:** every component emits structured logs with `request_id` + `session_id` context vars; pipe `logs/rag.jsonl` into `jq` or Loki.

---

**See also:** `docs/RAG_SETUP_AND_IMPLEMENTATION.md` for the deep-dive reference on Phases 1–2 internals, and `docs/pipeline_flow.svg` for the full visual pipeline.
