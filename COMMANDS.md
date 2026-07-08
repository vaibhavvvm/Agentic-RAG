# Agentic RAG — Command Reference

This file outlines all the commands required to run, develop, and maintain the Agentic RAG system.

---

## 1. Prerequisites (Docker Services)
The system requires PostgreSQL, Neo4j, MinIO, and Ollama to be running.
```bash
# Start all external services in the background
docker-compose up -d

# Check the status of the containers
docker-compose ps

# View logs for a specific service (e.g., neo4j)
docker-compose logs -f neo4j

# Stop all services
docker-compose down
```

## 2. Backend Server (FastAPI)
Run the core API that serves both the `/api` endpoints and the static frontend built files.

```bash
# Start the backend server on port 8000
src/.venv/bin/python -m uvicorn src.server:app --port 8000 --host 0.0.0.0

# Or, if your virtual environment is activated:
uvicorn src.server:app --port 8000 --host 0.0.0.0
```

## 3. Frontend (React / Vite)
If you are developing the UI, you can run the Vite dev server.

```bash
cd frontend

# Install dependencies (only needed once)
npm install

# Start the Vite development server (usually on port 5173)
npm run dev

# Build the frontend for production (outputs to frontend/dist)
npm run build
```
*(Note: The backend server serves the production UI from `frontend/dist` automatically, so you must run `npm run build` after making UI changes if you are accessing it via port 8000).*

## 4. CLI Utilities
You can use the built-in CLI to perform system tasks without the UI.

```bash
# 1. Healthcheck (pings Postgres, Neo4j, Ollama, Groq, MinIO)
src/.venv/bin/python -m src.main healthcheck

# 2. Ingest a document (PDF, MD, etc.)
src/.venv/bin/python -m src.main ingest path/to/document.pdf

# 3. Ingest a directory of documents recursively
src/.venv/bin/python -m src.main ingest path/to/directory --recursive

# 4. Run an interactive CLI chat session (Rich UI)
src/.venv/bin/python -m src.main chat

# 5. One-shot chat query from the terminal
src/.venv/bin/python -m src.main chat --query "Explain HNSW vs IVF-Flat"

# 6. Evaluate the system on a JSONL evaluation dataset
src/.venv/bin/python -m src.main evaluate eval/questions.jsonl
```

## 5. Virtual Environment Management
If you need to install or update dependencies:
```bash
# Create the virtual environment (if not already created)
python3 -m venv src/.venv

# Activate the virtual environment
source src/.venv/bin/activate

# Install requirements
pip install -r requirements.txt
```
