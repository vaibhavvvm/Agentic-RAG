# Agentic RAG — Intelligent Document Intelligence 🧠

![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111.0-009688.svg)
![React](https://img.shields.io/badge/React-18.3.1-61DAFB.svg)
![Neo4j](https://img.shields.io/badge/Neo4j-Graph-blue)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-pgvector-336791)

Agentic RAG is a state-of-the-art Document Intelligence and Retrieval-Augmented Generation (RAG) system. It goes beyond simple vector search by combining **Knowledge Graphs**, **Vision-Language Models (VLMs)**, and **Agentic Orchestration** to understand, synthesize, and reason across complex documents.

## ✨ Key Features

*   **Hybrid Orchestration:** Uses a LangGraph-based intent router that dynamically decides whether a query requires simple vector lookups, multi-hop graph traversal, or a hybrid fusion of both.
*   **Multi-Modal Ingestion:** Parses unstructured PDFs, tables, and images using `Docling`. Extracts visual contexts using `LLaVA` (Ollama) and stores them seamlessly.
*   **Knowledge Graph Extraction:** Automatically extracts entities and relationships (triples) from documents and stores them in Neo4j for deep, relational querying.
*   **Vector Search & Reranking:** Uses `pgvector` for semantic search and `Ollama` cross-encoders to re-rank chunks for maximum relevance.
*   **Episodic Memory:** Maintains long-term and short-term session memories to allow for continuous, contextual conversations.
*   **Beautiful UI:** A modern, dark-mode React frontend to manage your knowledge base, chat with the agent, and view source citations.

---

## 🏗️ Architecture

```mermaid
flowchart TD
    %% Styling
    classDef file fill:#f8fafc,stroke:#334155,stroke-width:1.5px
    classDef agent fill:#ede9fe,stroke:#7c3aed,stroke-width:1.5px
    classDef tool fill:#dbeafe,stroke:#2563eb,stroke-width:1.5px
    classDef store fill:#fef3c7,stroke:#ca8a04,stroke-width:1.5px
    classDef graph fill:#d1fae5,stroke:#059669,stroke-width:1.5px
    classDef intent fill:#fee2e2,stroke:#dc2626,stroke-width:1.5px

    %% INGESTION PIPELINE
    subgraph Ingestion["Ingestion Pipeline (LangGraph)"]
        direction TB
        doc[📄 Source File]:::file --> parse[Docling Parser]:::tool
        parse --> vlm[LLaVA Vision]:::agent
        vlm --> clean[Groq Text Cleaner]:::agent
        clean --> chunk[Semantic Chunker]:::tool
        chunk --> embed[Ollama Embedder]:::tool
        
        embed --> pg[(Postgres pgvector)]:::store
        embed --> neo[(Neo4j GraphStore)]:::graph
    end

    %% RETRIEVAL / ORCHESTRATOR
    subgraph Retrieval["RAG Orchestrator (LangGraph)"]
        direction TB
        user((👤 User Query)):::file --> session[RAG Session Memory]:::store
        session --> router{LLM Intent Router}:::intent
        
        router -- "General" --> gen[General Agent]:::agent
        router -- "Vector" --> vec[Vector Agent]:::agent
        router -- "Graph" --> gra[Graph Agent]:::agent
        router -- "Hybrid" --> hyb[Hybrid Agent]:::agent
        
        gen & vec & gra & hyb --> reflect{CRAG Reflection}:::intent
        reflect -- "Insufficient" --> rewrite[Query Rewriter]:::agent
        rewrite --> router
        
        reflect -- "Accept" --> synth[LLM Synthesizer]:::agent
        synth --> out((💬 Final Answer)):::file
    end

    %% Database Links
    vec -.-> pg
    gra -.-> neo
    hyb -.-> pg
    hyb -.-> neo
```

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have the following installed on your machine:
*   [Docker](https://docs.docker.com/get-docker/) & Docker Compose
*   [Python 3.12+](https://www.python.org/downloads/)
*   [Node.js 18+](https://nodejs.org/)
*   [Ollama](https://ollama.com/) (Running locally for embeddings and reranking)

### 2. Environment Setup
Clone the repository and set up your `.env` file:
```bash
cp .env.example .env
```
Add your required API keys (e.g., `GROQ_API_KEYS`) in the `.env` file.

### 3. Start Infrastructure Services
Start the required databases and object storage (PostgreSQL, Neo4j, MinIO):
```bash
docker-compose up -d
```

### 4. Install Dependencies
**Backend:**
```bash
python3 -m venv src/.venv
source src/.venv/bin/activate
pip install -r requirements.txt
```

**Frontend:**
```bash
cd frontend
npm install
```

---

## 💻 Running the Application

To run the full stack locally:

1.  **Start the Backend API:**
    ```bash
    # From the project root
    src/.venv/bin/python -m uvicorn src.server:app --port 8000 --host 0.0.0.0
    ```

2.  **Start the Frontend UI:**
    ```bash
    cd frontend
    npm run dev
    ```
    Navigate to `http://localhost:5173` to use the application! (Alternatively, the built UI is served on port 8000 by the backend).

---

## 🛠️ CLI Utilities

Agentic RAG also comes with a powerful CLI for headless operations:

```bash
# Check system health (Databases, LLMs)
python -m src.main healthcheck

# Ingest a single document
python -m src.main ingest path/to/document.pdf

# Ingest an entire directory recursively
python -m src.main ingest path/to/directory --recursive

# Run an interactive terminal chat (Rich UI)
python -m src.main chat

# Run evaluations on a dataset
python -m src.main evaluate eval/questions.jsonl
```

---

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request or open an Issue to discuss improvements, bug fixes, or new features.

## 📄 License
This project is licensed under the MIT License.
