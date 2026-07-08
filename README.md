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

1.  **Ingestion Pipeline (`Haystack` + `LangGraph`)**:
    *   **Parse**: Extracts text, tables, and images (`Docling`).
    *   **Vision**: Describes images using local VLMs (`LLaVA`).
    *   **Structure**: Cleans up formatting using Groq LLMs.
    *   **Embed & Store**: Chunks text, embeds via Ollama, and stores in Postgres (`pgvector`).
    *   **Graph Extraction**: Extracts triples and stores them in Neo4j.
2.  **Retrieval & Orchestration (`LangGraph`)**:
    *   **Router**: LLM-based intent routing determines the complexity of the query.
    *   **General/Vector/Graph/Hybrid Tools**: Executes the appropriate search strategy.
    *   **Reflection**: Grades the retrieved context. If insufficient, it rewrites the query and retries.
    *   **Synthesis**: Generates a grounded, multi-paragraph response with inline citations.

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
