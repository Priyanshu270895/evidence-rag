# EvidenceRAG

A local-first RAG service that answers questions from PDFs with page-level citations. It uses
hybrid retrieval (semantic + SQLite FTS5), reciprocal-rank fusion, a local embedding model,
and an Ollama-hosted language model.

## Current phase

EvidenceRAG is now in the local production baseline phase:

- Phase 0 complete: FastAPI, PDF ingestion, chunking, embeddings, SQLite FTS5, vector search,
  reciprocal-rank fusion, Ollama generation, grounded fallback, and tests.
- Phase 1 complete: document IDs, document metadata, duplicate detection by file hash, document
  deletion, reindexing, chunk inspection, upload size limits, and richer health checks.
- Phase 2 in progress: retrieval evaluation dataset and ranking metrics.

## Why this is not a toy chatbot

- Separates ingestion, retrieval, and generation
- Combines dense and keyword retrieval
- Returns inspectable evidence even when generation is unavailable
- Refuses unsupported questions
- Persists data locally without a paid API
- Includes typed API contracts and automated tests

## Laptop-friendly setup (Windows, 8 GB RAM)

Install Python 3.12, Git, and Ollama. Docker is **not required** for Phase 1.

```powershell
git clone https://github.com/Priyanshu270895/evidence-rag.git
cd evidence-rag
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
Copy-Item .env.example .env
python scripts/cache_embedding_model.py
ollama pull qwen2.5:1.5b
ollama serve
```

In a second PowerShell window:

```powershell
.venv\Scripts\Activate.ps1
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000/docs`, upload a PDF with `POST /documents`, then ask a question
with `POST /ask`.

## API overview

```text
GET    /health
POST   /documents
GET    /documents
GET    /documents/{document_id}
DELETE /documents/{document_id}
POST   /documents/{document_id}/reindex
GET    /documents/{document_id}/chunks
POST   /ask
```

Documents are tracked as indexed assets. Each PDF gets a stable `document_id`, SHA-256
`file_hash`, original filename, page count, chunk count, indexing status, upload path, and the
chunking settings used to build its index. If the same PDF is uploaded again, EvidenceRAG returns
the existing document and skips duplicate embedding work.

Chunks belong to a document ID and keep page-level metadata. Answers return citations containing
`document_id`, filename, page number, chunk ID, excerpt, and retrieval score.

## Architecture

```text
FastAPI routes
  -> Pydantic request/response schemas
  -> RAG service layer
  -> PDF parsing and chunking
  -> Sentence Transformer embeddings
  -> SQLite document/chunk store
  -> vector search + SQLite FTS5 keyword search
  -> reciprocal-rank fusion
  -> Ollama grounded generation
  -> cited answer response
```

The project intentionally uses transparent Python components before adding orchestration
frameworks. This makes the full RAG pipeline explainable in interviews and keeps the first local
version lightweight enough for an 8 GB Windows laptop.

## Tests

```powershell
ruff format --check .
ruff check .
pytest
```

## Planned production increments

1. Retrieval evaluation set with Recall@K, Precision@K, MRR, nDCG, and per-query reports
2. Cross-encoder reranking and chunking experiments guided by evaluation metrics
3. Stronger refusal behavior, citation correctness checks, and prompt-injection defenses
4. Streamlit UI, structured logs, request IDs, latency metrics, and GitHub Actions CI
5. Docker Compose with Qdrant and Azure deployment after the local version is reliable

## Safety

Use public or personally owned documents only. Uploaded PDFs and embeddings stay on the local
machine in this version.
