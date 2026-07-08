# RAG — Complete Setup, Architecture & Implementation Reference

> **Version:** 1.0.0 | **Python:** 3.11+ | **Framework:** Haystack 2.x

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Full Architecture Diagram](#2-full-architecture-diagram)
3. [Directory Structure](#3-directory-structure)
4. [Prerequisites & Dependency Matrix](#4-prerequisites--dependency-matrix)
5. [Phase 1 — Foundation Layer](#5-phase-1--foundation-layer)
6. [Phase 2 — Ingestion Pipeline](#6-phase-2--ingestion-pipeline)
7. [Environment Configuration](#7-environment-configuration)
8. [Installation Guide](#8-installation-guide)
9. [External Service Setup](#9-external-service-setup)
10. [Ingestion Pipeline Deep-Dive](#10-ingestion-pipeline-deep-dive)
11. [Component Interface Reference](#11-component-interface-reference)
12. [Testing Strategy](#12-testing-strategy)
13. [Performance Tuning](#13-performance-tuning)
14. [Troubleshooting](#14-troubleshooting)

---

## 1. System Overview

RAG is a **production-grade, agentic Retrieval-Augmented Generation system** built on:

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Orchestration | Haystack 2.x Pipelines + ReAct Agent | Composable, inspectable pipelines |
| Primary LLM | Groq (Llama 3 70B / 8B) | Inference with key-rotation & rate-limit handling |
| Embeddings | Ollama `nomic-embed-text` | Local, privacy-preserving vectors |
| Reranking | Ollama `bge-reranker-v2-m3` | Cross-encoder precision boost |
| Vector Store | PostgreSQL 16 + pgvector | Hybrid BM25+vector search, HNSW indexing |
| Graph Store | Neo4j 5 + Graphiti | Entity-relation knowledge graph, multi-hop |
| Memory | FAISS + Neo4j episodic | Sliding window + long-term episodic recall |
| Document Parsing | `unstructured` hi_res | PDF, DOCX, HTML, images with OCR |
| Vision | Ollama `llava:13b` | Image → natural language for multimodal RAG |

### Data Flow (End-to-End)

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  RAGSession (sliding window + FAISS memory lookup)  │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  IntentRouter (Regex → Keywords → LLM Arbiter)      │
│  Classifies: GeneralChat / VectorRetrieval /        │
│              GraphRetrieval / HybridRetrieval        │
└─────────────────────────────────────────────────────┘
    │
    ├──── GeneralChat ──────► GeneralAgent (fast LLM)
    │
    ├──── VectorRetrieval ──► VectorAgent
    │                              │
    │                    QueryExpansion → pgvector hybrid search
    │                    → Reranker → SelfReflection → LLM
    │
    ├──── GraphRetrieval ───► GraphAgent
    │                              │
    │                    Graphiti semantic search → Neo4j subgraph
    │                    → Fact extraction → LLM
    │
    └──── HybridRetrieval ──► GraphVectorFusion
                                   │
                         Both paths in parallel → RRF fusion
                         → Reranker → LLM

    ▼
Synthesizer (formats response + attaches metadata)
    ▼
User Response
```

---

## 2. Full Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│                         INGESTION PIPELINE                        │
│                                                                    │
│  File ──► DocumentParser ──► ParsedElement[]                      │
│              │                    │                                │
│              │             ┌──────┴──────────────┐                │
│              │             │                     │                 │
│              ▼             ▼                     ▼                 │
│         VisionProcessor  TableReformatter   (text/title)          │
│              │                 │                 │                 │
│              └──────────────────────────────────►│                 │
│                                                  ▼                 │
│                                    ┌─────────────────────┐        │
│                                    │   SemanticChunker   │        │
│                                    └──────────┬──────────┘        │
│                                               │                    │
│                                    ┌──────────▼──────────┐        │
│                                    │  ContextualChunker  │        │
│                                    └──────────┬──────────┘        │
│                                               │                    │
│                             ┌─────────────────▼───────────────┐   │
│                             │       HierarchicalChunker        │   │
│                             │   parent_docs   child_docs       │   │
│                             └────────┬──────────────┬──────────┘   │
│                                      │              │               │
│                             ┌────────▼──┐    ┌─────▼──────────┐   │
│                             │ Summary   │    │ pgvector Store  │   │
│                             │ Index     │    │ (HNSW + BM25)  │   │
│                             └───────────┘    └────────────────┘   │
│                                                      │             │
│                             ┌────────────────────────▼──────────┐ │
│                             │         Neo4j + Graphiti           │ │
│                             │   (entity/relation extraction)     │ │
│                             └────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│                         RETRIEVAL PIPELINE                        │
│                                                                    │
│  ┌─────────────┐    ┌──────────────────┐    ┌─────────────────┐  │
│  │ QueryRouter │───►│ QueryExpansion   │───►│ VectorSearch    │  │
│  └─────────────┘    └──────────────────┘    └────────┬────────┘  │
│                                                       │           │
│  ┌─────────────┐                            ┌────────▼────────┐  │
│  │ GraphSearch │◄───────────────────────────┤   RRF Fusion    │  │
│  └─────────────┘                            └────────┬────────┘  │
│                                                       │           │
│                                             ┌────────▼────────┐  │
│                                             │   OllamaRanker  │  │
│                                             └────────┬────────┘  │
│                                                       │           │
│                                             ┌────────▼────────┐  │
│                                             │ SelfReflection  │  │
│                                             └────────┬────────┘  │
│                                                       │           │
│                                             ┌────────▼────────┐  │
│                                             │   LLM Answer    │  │
│                                             └─────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

---

## 3. Directory Structure

```
RAG/
├── .env                          ← Your secrets (copy from .env.example)
├── .env.example                  ← Template with all variables documented
├── pyproject.toml                ← Dependencies + tool config
├── docs/
│   └── RAG_SETUP_AND_IMPLEMENTATION.md  ← This file
├── data/
│   └── faiss_episodic/           ← Auto-created: FAISS index storage
└── src/
    ├── __init__.py
    ├── config.py                 ← [Phase 1] All settings via Pydantic
    ├── main.py                   ← [Phase 5] CLI entry + RAGSystem
    │
    ├── monitoring/
    │   ├── logger.py             ← [Phase 1] Structured JSON logger
    │   └── metrics.py            ← [Phase 1] P50/P90/P99 latency tracker
    │
    ├── utils/
    │   ├── llm.py                ← [Phase 3] chat_sync() helper
    │   └── groq_client.py        ← [Phase 1] RotatableGroqGenerator
    │
    ├── storage/
    │   ├── base.py               ← [Phase 1] Abstract interfaces
    │   ├── postgres/
    │   │   ├── vector_store.py   ← [Phase 3] pgvector + hybrid + RRF
    │   │   └── summary_store.py  ← [Phase 3] Summary index
    │   └── graph/
    │       └── neo4j_store.py    ← [Phase 3] Neo4j + Graphiti
    │
    ├── ingestion/                ← [Phase 2] ← YOU ARE HERE
    │   ├── embedder.py           ✓ CachedOllamaEmbedder (L1+L2 cache)
    │   ├── parser.py             ✓ DocumentParser (hi_res OCR)
    │   ├── vision.py             ✓ VisionProcessor (LLaVA)
    │   ├── tables.py             ✓ TableReformatter (rule + LLM)
    │   ├── semantic_chunker.py   ✓ Embedding-based breakpoints
    │   ├── contextual_chunker.py ✓ LLM context enrichment
    │   └── hierarchical_chunker.py ✓ Parent/child hierarchy
    │
    ├── memory/
    │   ├── memory_tools.py       ← [Phase 4] Context builders
    │   ├── summarizer.py         ← [Phase 4] Turn summariser
    │   └── vector_store.py       ← [Phase 4] FAISS + GraphMemoryManager
    │
    ├── retrieval/
    │   ├── agent.py              ← [Phase 4] Haystack ReAct Agent
    │   ├── cache.py              ← [Phase 4] Multi-level LRU cache
    │   ├── fallback.py           ← [Phase 4] Progressive fallback
    │   ├── session.py            ← [Phase 4] RAGSession
    │   ├── tools/
    │   │   ├── vector_tool.py    ← [Phase 4] Vector search as Tool
    │   │   └── graph_search_tool.py ← [Phase 4] Graph search as Tool
    │   └── strategies/
    │       ├── reranking.py      ← [Phase 4] OllamaRanker
    │       ├── query_expansion.py ← [Phase 4] LLM query reformulation
    │       ├── self_reflection.py ← [Phase 4] Quality grading + retry
    │       ├── query_router.py   ← [Phase 4] Query classification
    │       ├── summary_index.py  ← [Phase 4] Summary generation
    │       └── graph_fusion.py   ← [Phase 4] Hybrid fusion
    │
    ├── agents/
    │   ├── orchestrator.py       ← [Phase 5] Central supervisor
    │   ├── router.py             ← [Phase 5] 3-Tier intent router
    │   ├── synthesizer.py        ← [Phase 5] Output formatter
    │   └── workers/
    │       ├── general_agent.py  ← [Phase 5] Fast conversational LLM
    │       ├── vector_agent.py   ← [Phase 5] Vector query wrapper
    │       └── graph_agent.py    ← [Phase 5] Graph agent
    │
    └── evaluation/
        ├── metrics.py            ← [Phase 5] RAGAS + fallback metrics
        └── datasets.py           ← [Phase 5] Dataset management
```

---

## 4. Prerequisites & Dependency Matrix

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python | 3.11 | 3.12 |
| RAM | 8 GB | 16 GB |
| Disk (models) | 20 GB | 50 GB |
| GPU (optional) | — | NVIDIA 8GB VRAM for Ollama |
| OS | Linux / macOS / Windows (WSL2) | Ubuntu 22.04 |

### External Services

| Service | Version | Purpose | Required? |
|---------|---------|---------|-----------|
| PostgreSQL | 16+ | Vector + BM25 storage | Yes |
| pgvector extension | 0.7+ | HNSW index + cosine ops | Yes |
| Neo4j | 5.21+ | Graph knowledge base | Yes |
| Ollama | 0.2+ | Local embeddings + vision | Yes |
| Groq API | — | Primary LLM inference | Yes |
| Tesseract OCR | 5.3+ | PDF hi_res OCR | Recommended |
| Poppler | 23+ | PDF → image conversion | Recommended |

### Python Dependencies (key packages)

```toml
# Core
haystack-ai          >= 2.7.0    # Pipeline framework (MUST be v2.x)
pydantic             >= 2.7.0    # Settings + data validation
pydantic-settings    >= 2.3.0    # BaseSettings with .env support

# LLM
groq                 >= 0.9.0    # Groq Python SDK
ollama               >= 0.2.1    # Ollama Python client

# Storage
asyncpg              >= 0.29.0   # Async PostgreSQL driver
pgvector             >= 0.3.2    # pgvector Python bindings
psycopg[binary]      >= 3.1.19   # Sync PostgreSQL (Haystack compat)
neo4j                >= 5.21.0   # Neo4j Python driver
graphiti-core        >= 0.3.0    # Graphiti knowledge graph

# Embeddings & Retrieval
faiss-cpu            >= 1.8.0    # FAISS episodic memory index
numpy                >= 1.26.0   # Vector math
scikit-learn         >= 1.5.0    # Cosine similarity helpers

# Document Parsing
unstructured[all-docs] >= 0.14.0  # Multi-format parser
pillow               >= 10.4.0   # Image processing

# Evaluation
ragas                >= 0.1.14   # RAG evaluation metrics
datasets             >= 2.20.0   # HuggingFace datasets

# Utilities
python-dotenv        >= 1.0.1    # .env file loading
httpx                >= 0.27.0   # Async HTTP (Ollama calls)
tenacity             >= 8.5.0    # Retry logic
rich                 >= 13.7.0   # CLI formatting
typer                >= 0.12.0   # CLI framework
```

---

## 5. Phase 1 — Foundation Layer

### Files Delivered

| File | Class / Function | Description |
|------|-----------------|-------------|
| `src/config.py` | `Settings`, `get_settings()` | Pydantic BaseSettings with 9 nested groups |
| `src/monitoring/logger.py` | `get_logger()`, `timed_operation()` | JSON structured logger + context vars |
| `src/monitoring/metrics.py` | `MetricsCollector` | P50/P90/P99 + counters + snapshots |
| `src/utils/groq_client.py` | `RotatableGroqGenerator` | Haystack 2.x component, key rotation |
| `src/storage/base.py` | 4 ABCs | Contracts for vector/graph/memory/summary stores |

### Key Design Decisions

**`config.py`** — Uses `env_nested_delimiter="__"` so `POSTGRES__HOST=myhost` overrides `postgres.host`. The `@lru_cache` singleton means `.env` is read exactly once. Call `get_settings.cache_clear()` in tests after patching env vars.

**`logger.py`** — ContextVars (`request_id`, `session_id`) are automatically included in every log line without passing the logger around. Set them once per request in the orchestrator.

**`groq_client.py`** — The `_KeyPool` uses `time.monotonic()` for per-key cooldowns. Each key has an *independent* cooldown so a key that was rate-limited rejoins the rotation automatically after the penalty window expires. No shared back-off state between keys.

---

## 6. Phase 2 — Ingestion Pipeline

### Files Delivered

| File | Class | Strategy | Haystack? |
|------|-------|---------|-----------|
| `src/ingestion/embedder.py` | `CachedOllamaEmbedder` | L1 LRU + L2 shelve, batched, normalised | `@component` |
| `src/ingestion/parser.py` | `DocumentParser` | unstructured hi_res/fast/auto | `@component` |
| `src/ingestion/vision.py` | `VisionProcessor` | Ollama LLaVA, async batch, cached | `@component` |
| `src/ingestion/tables.py` | `TableReformatter` | Rule-based + LLM for complex tables | `@component` |
| `src/ingestion/semantic_chunker.py` | `SemanticChunker` | Embedding breakpoints + size constraints | `@component` |
| `src/ingestion/contextual_chunker.py` | `ContextualChunker` | LLM context prefix, cached | `@component` |
| `src/ingestion/hierarchical_chunker.py` | `HierarchicalChunker` | Parent/child split, deterministic IDs | `@component` |

### Ingestion Pipeline Assembly (Haystack)

```python
from haystack import Pipeline
from src.ingestion.parser import DocumentParser
from src.ingestion.vision import VisionProcessor
from src.ingestion.tables import TableReformatter
from src.ingestion.semantic_chunker import SemanticChunker
from src.ingestion.contextual_chunker import ContextualChunker
from src.ingestion.hierarchical_chunker import HierarchicalChunker
from src.ingestion.embedder import CachedOllamaEmbedder

pipeline = Pipeline()
pipeline.add_component("parser",       DocumentParser(strategy="hi_res"))
pipeline.add_component("embedder",     CachedOllamaEmbedder())
pipeline.add_component("sem_chunker",  SemanticChunker())
pipeline.add_component("ctx_chunker",  ContextualChunker())
pipeline.add_component("hier_chunker", HierarchicalChunker())

# Connect: parser → semantic chunker → contextual chunker → hierarchical
pipeline.connect("parser.elements",           "sem_chunker.elements")
pipeline.connect("sem_chunker.documents",     "ctx_chunker.documents")
# hier_chunker takes original elements for parent sizing
pipeline.connect("parser.elements",           "hier_chunker.elements")

result = pipeline.run({"parser": {"file_path": "docs/manual.pdf"}})
child_docs = result["hier_chunker"]["child_documents"]
parent_docs = result["hier_chunker"]["parent_documents"]
```

---

## 7. Environment Configuration

### Complete `.env` File

Copy `.env.example` to `.env` and populate:

```bash
cp .env.example .env
```

### Critical Variables (must be set)

```env
# Groq API (at least one key required)
GROQ_API_KEY=gsk_your_key_here
# OR for rotation:
# GROQ_API_KEYS=gsk_key1,gsk_key2,gsk_key3

# PostgreSQL
POSTGRES_PASSWORD=your_secure_password

# Neo4j
NEO4J_PASSWORD=your_secure_password
```

### Nested Override Syntax

You can override any nested setting with double-underscore notation:

```env
POSTGRES__HOST=prod-db.internal   # overrides postgres.host
GROQ__TEMPERATURE=0.0             # overrides groq.temperature
MEMORY__WINDOW_SIZE=20            # overrides memory.window_size
```

---

## 8. Installation Guide

### Step 1 — Python Environment

```bash
# Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate          # Windows

# Install dependencies
pip install -e ".[dev]"
```

### Step 2 — OCR Tools (for PDF hi_res)

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr tesseract-ocr-eng \
    poppler-utils libmagic1 libreoffice
```

**macOS:**
```bash
brew install tesseract poppler libmagic
```

**Windows (WSL2 recommended):**
```bash
# Inside WSL2 Ubuntu:
sudo apt-get install tesseract-ocr poppler-utils
```

### Step 3 — Ollama Setup

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull required models
ollama pull nomic-embed-text       # Embeddings (~274 MB)
ollama pull bge-reranker-v2-m3     # Reranker (~568 MB)
ollama pull llava:13b              # Vision (~8 GB)
ollama pull llama3:8b              # Local chat fallback (~4.7 GB)

# Verify
ollama list
```

### Step 4 — PostgreSQL + pgvector

```bash
# Ubuntu — install PostgreSQL 16 + pgvector
sudo apt-get install -y postgresql-16 postgresql-server-dev-16

# Install pgvector extension
git clone https://github.com/pgvector/pgvector.git
cd pgvector
make && sudo make install

# Create database
sudo -u postgres psql <<EOF
CREATE USER raguser WITH PASSWORD 'yourpassword';
CREATE DATABASE rag OWNER raguser;
\c rag
CREATE EXTENSION vector;
\q
EOF
```

**Docker alternative:**
```bash
docker run -d \
  --name rag-postgres \
  -e POSTGRES_USER=raguser \
  -e POSTGRES_PASSWORD=yourpassword \
  -e POSTGRES_DB=rag \
  -p 5432:5432 \
  ankane/pgvector:v0.7.4
```

### Step 5 — Neo4j

```bash
# Docker (recommended)
docker run -d \
  --name rag-neo4j \
  -e NEO4J_AUTH=neo4j/yourpassword \
  -e NEO4J_PLUGINS='["apoc"]' \
  -p 7687:7687 \
  -p 7474:7474 \
  neo4j:5.21-community
```

**Verify:** Open http://localhost:7474 in browser (neo4j / yourpassword).

### Step 6 — Graphiti

```bash
pip install graphiti-core
```

Graphiti initialises inside `src/storage/graph/neo4j_store.py` using the Neo4j connection from settings. No separate service needed.

### Step 7 — Environment File

```bash
cp .env.example .env
# Edit .env with your credentials
```

### Step 8 — Verify Installation

```bash
python -c "
from src.config import get_settings
cfg = get_settings()
print('Config OK:', cfg.app_name, cfg.app_version)
print('Groq keys:', len(cfg.groq.api_keys))
print('Postgres DSN:', cfg.postgres.dsn[:40], '...')
"
```

---

## 9. External Service Setup

### Groq API Keys

1. Create account at https://console.groq.com
2. Generate API key(s) under API Keys section
3. For rotation, create 3–5 keys and add them comma-separated:
   ```env
   GROQ_API_KEYS=gsk_key1,gsk_key2,gsk_key3
   ```
4. Free tier: 30 RPM, 6000 tokens/min (per key)
5. **Recommended models:**
   - `llama3-70b-8192` — primary reasoning (slower, better quality)
   - `llama3-8b-8192` — routing/grading (fast, cheap)

### Ollama Model Selection

```bash
# Embedding models (choose one)
ollama pull nomic-embed-text          # 768-dim, fast (RECOMMENDED)
ollama pull mxbai-embed-large         # 1024-dim, higher quality

# Vision models (choose one based on VRAM)
ollama pull llava:7b                  # 4 GB VRAM
ollama pull llava:13b                 # 8 GB VRAM (RECOMMENDED)
ollama pull llava:34b                 # 20 GB VRAM

# Reranker
ollama pull bge-reranker-v2-m3        # Cross-encoder, ~568 MB
```

Update `.env` if using non-default models:
```env
OLLAMA_EMBEDDING_MODEL=mxbai-embed-large
OLLAMA_VISION_MODEL=llava:7b
POSTGRES_VECTOR_DIM=1024   # MUST match embedding dimension!
```

### PostgreSQL Schema Initialisation

Phase 3 will create tables automatically via `initialise()`. The schema includes:
- `documents` table with `embedding vector(768)` column
- HNSW index: `CREATE INDEX ON documents USING hnsw (embedding vector_cosine_ops)`
- Full-text search: `tsvector` column with GIN index for BM25
- Hybrid search: Reciprocal Rank Fusion SQL function

### Neo4j Configuration

Enable APOC procedures (required by Graphiti):
```env
# neo4j.conf (or via Docker env)
NEO4J_PLUGINS=["apoc"]
dbms.security.procedures.allowlist=apoc.*
dbms.security.procedures.unrestricted=apoc.*
```

---

## 10. Ingestion Pipeline Deep-Dive

### DocumentParser — Supported Formats

| Format | Strategy | OCR | Tables | Images |
|--------|---------|-----|--------|--------|
| PDF | hi_res | ✓ Tesseract | ✓ detectron2 | ✓ base64 |
| DOCX | fast | ✗ | ✓ | ✓ |
| HTML | fast | ✗ | ✓ | ✗ |
| TXT/MD | fast | ✗ | ✗ | ✗ |
| PNG/JPG | n/a | ✗ | ✗ | ✓ direct |

### SemanticChunker — Algorithm Detail

```
Text
 │
 ▼  split_sentences()
["Sentence 1", "Sentence 2", ..., "Sentence N"]
 │
 ▼  CachedOllamaEmbedder.run(sentences)
[vec_1, vec_2, ..., vec_N]
 │
 ▼  cosine_similarity(vec_i, vec_{i+1}) for all i
[sim_1, sim_2, ..., sim_{N-1}]
 │
 ▼  percentile threshold = percentile(sims, 100 - breakpoint_pct)
     default breakpoint_pct=95 → threshold = 5th percentile of sims
 │
 ▼  breakpoint at i where sim_i < threshold
["S1 S2 S3" | "S4 S5" | "S6 S7 S8 S9"]
 │
 ▼  merge_short() + split_by_size()
Final chunks: 100–1500 chars each
```

**Tuning:**
- Higher `CHUNK_SEMANTIC_BREAKPOINT_PERCENTILE` (e.g. 99) → more, smaller chunks
- Lower (e.g. 80) → fewer, larger chunks
- Adjust `CHUNK_SEMANTIC_MIN/MAX_CHUNK_SIZE` for your embedding model's token limit

### ContextualChunker — LLM Prompt Flow

```
Document backdrop (first 3000 chars)
        +
Chunk content (up to 1500 chars)
        │
        ▼
  Groq llama3-8b-8192 (fast, low temperature)
        │
        ▼
"This chunk is from Section 4 of the API reference and describes
 the authentication flow for OAuth2 clients."
        │
        ▼
Prepended to chunk:
"This chunk is from Section 4... [context sentence]

[original chunk content]"
```

**Cost estimation:** ~200 tokens per chunk × N chunks.
- At Groq free tier (6K tokens/min): 30 chunks/min
- At paid tier: scales linearly

### HierarchicalChunker — ID Scheme

```
Parent:  hier_par_{index}_{sha256[:12]}
Child:   hier_chd_{parent_index}_{child_index}_{sha256[:12]}
Special: hier_spc_{index}_{sha256[:12]}_par / _chd
```

IDs are deterministic from content so re-ingesting the same document
produces identical IDs → upsert-safe in pgvector.

---

## 11. Component Interface Reference

### CachedOllamaEmbedder

```python
embedder = CachedOllamaEmbedder(
    model="nomic-embed-text",     # Ollama model tag
    base_url="http://localhost:11434",
    batch_size=32,                # texts per HTTP call
    l1_maxsize=4096,              # in-process LRU entries
    disk_cache_path=Path("data/embed_cache"),  # None to disable
    normalise=True,               # L2-normalise output
)
result = embedder.run(texts=["hello", "world"])
# {"embeddings": [[0.12, -0.34, ...], [...]]}
```

### DocumentParser

```python
parser = DocumentParser(
    strategy="hi_res",            # or "fast", "auto"
    extract_images=True,
    pdf_infer_table_structure=True,
    languages=["eng"],
)
result = parser.run(
    file_path="docs/report.pdf",
    metadata={"source": "upload", "doc_id": "rpt-001"},
)
# {"elements": [ParsedElement(element_type="text", text="...", page_number=1), ...]}
```

### VisionProcessor

```python
vp = VisionProcessor(
    model="llava:13b",
    concurrency=4,      # parallel Ollama requests
    cache_size=512,
)
result = vp.run(
    image_b64_list=[base64_string],
    extra_context="Figure 3 from Chapter 2 showing system architecture.",
)
# {"descriptions": ["The image shows a three-tier architecture diagram..."]}
```

### TableReformatter

```python
reformatter = TableReformatter(
    llm_threshold_cells=30,    # tables with >30 cells use LLM
    use_llm=True,
)
result = reformatter.run(
    html_tables=["<table><tr><th>Col1</th>...</table>"],
    context="From the financial report Q3 2024.",
)
tbl = result["tables"][0]
print(tbl.best_text)        # LLM summary or rule-based NL
print(tbl.strategy_used)    # "rule_based" or "llm"
```

### SemanticChunker

```python
chunker = SemanticChunker(
    breakpoint_percentile=95.0,
    min_chunk_size=100,
    max_chunk_size=1500,
    chunk_overlap_sentences=1,
)
result = chunker.run(elements=parsed_elements, doc_id_prefix="pdf001")
docs = result["documents"]
```

### ContextualChunker

```python
enricher = ContextualChunker(
    model="llama3-8b-8192",
    backdrop_chars=3000,
    cache_enabled=True,
)
result = enricher.run(
    documents=semantic_chunks,
    full_text=raw_document_text,
    source_id="doc-001",
)
enriched_docs = result["documents"]
```

### HierarchicalChunker

```python
hier = HierarchicalChunker(
    parent_chunk_size=2000,
    child_chunk_size=400,
    child_chunk_overlap=80,
)
result = hier.run(elements=parsed_elements)
parents = result["parent_documents"]   # store in summary index
children = result["child_documents"]   # embed + store in pgvector
```

---

## 12. Testing Strategy

### Unit Tests

```bash
# Run Phase 2 unit tests
pytest tests/ingestion/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=term-missing
```

### Test Structure

```
tests/
├── conftest.py                    # fixtures: mock Ollama, mock Groq
├── ingestion/
│   ├── test_embedder.py           # cache hit/miss, normalisation
│   ├── test_parser.py             # element type detection
│   ├── test_vision.py             # fallback behaviour
│   ├── test_tables.py             # HTML parsing, NL output
│   ├── test_semantic_chunker.py   # breakpoint detection
│   ├── test_contextual_chunker.py # cache, LLM call count
│   └── test_hierarchical_chunker.py  # parent/child linkage
└── monitoring/
    ├── test_logger.py
    └── test_metrics.py
```

### Key Test Fixtures

```python
# conftest.py
import pytest
from unittest.mock import MagicMock, patch

@pytest.fixture
def mock_embedder():
    """Returns deterministic embeddings without calling Ollama."""
    embedder = MagicMock()
    embedder.run.return_value = {
        "embeddings": [[0.1] * 768, [0.2] * 768]
    }
    return embedder

@pytest.fixture(autouse=True)
def patch_settings(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_key")
    monkeypatch.setenv("POSTGRES_PASSWORD", "test")
    monkeypatch.setenv("NEO4J_PASSWORD", "test")
    from src.config import get_settings
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
```

---

## 13. Performance Tuning

### Embedding Throughput

| Setting | Value | Effect |
|---------|-------|--------|
| `OLLAMA_EMBEDDING_MODEL` | `nomic-embed-text` | 768-dim, ~200 texts/sec (GPU) |
| `batch_size` | 32 | Texts per Ollama call |
| `l1_maxsize` | 4096 | In-process cache entries |
| `disk_cache_path` | `data/embed_cache` | Survives restarts |

**Expected throughput (CPU-only):**
- nomic-embed-text: ~50 texts/sec, batch=32
- With GPU: ~200 texts/sec

### Chunking Performance

| Strategy | Tokens/sec | Cost | Quality |
|----------|-----------|------|---------|
| Hierarchical only | High | Free | Good |
| Semantic | Medium | Free (local) | Better |
| Semantic + Contextual | Low | ~200 tok/chunk | Best |

**Recommendation:** Use hierarchical for bulk ingestion, add contextual for critical documents.

### PDF Parsing Speed

| Strategy | Speed | Quality |
|---------|-------|---------|
| `fast` | ~10 pages/sec | Text layer only |
| `auto` | ~5 pages/sec | Adaptive |
| `hi_res` | ~0.5 pages/sec | OCR + layout |

Set `strategy="fast"` for initial ingestion, re-ingest with `hi_res` for important documents.

---

## 14. Troubleshooting

### Ollama connection refused

```bash
# Check Ollama is running
ollama serve                    # Start if not running
curl http://localhost:11434/api/tags  # Should return model list
```

### Groq rate limit errors

```
AllKeysExhaustedException: All N Groq API keys are currently rate-limited.
```
- Add more API keys via `GROQ_API_KEYS`
- Reduce `RETRIEVAL_NUM_EXPANDED_QUERIES` to lower LLM call volume
- Increase `GROQ_RETRY_MAX_DELAY` to allow longer back-off

### unstructured import errors

```bash
# Install all document type extras
pip install 'unstructured[all-docs]'
# May also need system packages:
sudo apt-get install libmagic1 libreoffice-core
```

### pgvector dimension mismatch

```
ERROR: different vector dimensions 768 and 1024
```
- Ensure `POSTGRES_VECTOR_DIM` matches your Ollama embedding model's output dimension
- `nomic-embed-text` → 768
- `mxbai-embed-large` → 1024
- Drop and recreate the table if changing models on existing data

### FAISS index not found

The FAISS index is created automatically at the path in `MEMORY_FAISS_INDEX_PATH`.
If the path is read-only:
```env
MEMORY_FAISS_INDEX_PATH=/tmp/rag_faiss
```

### Tesseract not found (hi_res PDF)

```bash
# Ubuntu
sudo apt-get install tesseract-ocr
# Verify
tesseract --version

# Set language data path if needed
export TESSDATA_PREFIX=/usr/share/tesseract-ocr/5/tessdata/
```

---

## Coming in Phase 3

- `src/storage/postgres/vector_store.py` — Full pgvector hybrid store with RRF
- `src/storage/postgres/summary_store.py` — Summary index implementation
- `src/storage/graph/neo4j_store.py` — Neo4j + Graphiti episode ingestion
- `src/utils/llm.py` — `chat_sync()` helper pipeline
- `src/retrieval/strategies/reranking.py` — OllamaRanker (bge-reranker-v2-m3)

---

*Generated for RAG v1.0.0 — 2026-04-20*
