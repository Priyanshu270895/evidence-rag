# EvidenceRAG

A local-first RAG service that answers questions from PDFs with page-level citations. It uses
hybrid retrieval (semantic + SQLite FTS5), reciprocal-rank fusion, a local embedding model,
and an Ollama-hosted language model.

## Current phase

EvidenceRAG is now in the local production portfolio phase:

- Phase 0 complete: FastAPI, PDF ingestion, chunking, embeddings, SQLite FTS5, vector search,
  reciprocal-rank fusion, Ollama generation, grounded fallback, and tests.
- Phase 1 complete: document IDs, document metadata, duplicate detection by file hash, document
  deletion, reindexing, chunk inspection, upload size limits, and richer health checks.
- Phase 2 complete: retrieval evaluation dataset, Recall@K, Precision@K, MRR, nDCG, and a CLI
  report for measuring retrieval quality before adding reranking.
- Phase 3 complete: grounded-answer metadata, citation validation, prompt-injection checks,
  structured logs, request IDs, latency reporting, retry settings, Streamlit UI, and GitHub Actions
  CI.
- Phase 4 complete: optional Docker Compose, optional Qdrant vector backend, Docker build assets,
  and Azure deployment design.

## Why this is not a toy chatbot

- Separates ingestion, retrieval, and generation
- Combines dense and keyword retrieval
- Returns inspectable evidence even when generation is unavailable
- Refuses unsupported questions
- Adds grounding metadata, request IDs, latency measurements, and structured logs
- Includes retrieval evaluation before retrieval optimizations
- Persists data locally without a paid API
- Includes typed API contracts and automated tests

## Laptop-friendly setup (Windows, 8 GB RAM)

Install Python 3.12, Git, and Ollama. Docker is optional and not required for normal local use.

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

For a tiny smoke-test PDF, you can use the public W3C dummy PDF and upload it with the filename
`evidence-rag-smoke.pdf`.

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
`document_id`, filename, page number, chunk ID, excerpt, and retrieval score. Ask responses also
include grounding metadata with answer status, cited source indexes, invalid source indexes, prompt
injection risk, warnings, endpoint latency, and request ID.

## Architecture

```text
FastAPI routes
  -> request-ID middleware and structured JSON logs
  -> Pydantic request/response schemas
  -> RAG service layer
  -> PDF parsing and chunking
  -> Sentence Transformer embeddings
  -> SQLite document/chunk store
  -> vector search + SQLite FTS5 keyword search
  -> reciprocal-rank fusion
  -> Ollama grounded generation with retries
  -> citation and support checks
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

## Retrieval evaluation

Phase 2 adds a lightweight retrieval evaluator:

```powershell
python scripts/evaluate_retrieval.py --dataset eval/retrieval_eval.example.jsonl --k 5
```

The dataset is JSONL. Each line contains a `question` and a list of `relevant` evidence labels.
Labels can identify evidence by `chunk_id`, `document_id`, `document` or `filename`, `page`, and
optional `contains` text.

The evaluator reports:

- `Precision@K`: how much of the top K retrieved evidence is relevant.
- `Recall@K`: how much of the expected evidence was found in the top K.
- `MRR`: whether the first relevant result appears early.
- `nDCG@K`: whether relevant results are ranked near the top.

This gives EvidenceRAG a measurable baseline. Reranking, query rewriting, and chunking changes
should improve these numbers before they are considered useful.

## Streamlit UI

The UI is optional:

```powershell
pip install -e ".[dev,ui]"
streamlit run ui/streamlit_app.py
```

Keep the FastAPI server running separately. The UI lets you upload PDFs, inspect indexed documents,
ask questions, and review citations.

## Optional Docker and Qdrant

The Python local workflow remains the default. After Docker Desktop is installed, you can run the
containerized stack:

```powershell
docker compose up --build
docker compose exec ollama ollama pull qwen2.5:1.5b
```

Docker Compose starts:

- `api`: FastAPI EvidenceRAG container
- `qdrant`: optional vector database backend
- `ollama`: local model serving container

The local `.env.example` defaults to `VECTOR_BACKEND=sqlite`. Docker Compose sets
`VECTOR_BACKEND=qdrant` so vector search can move from SQLite brute force to Qdrant without changing
the API contract.

## Azure deployment

See [Azure Deployment Track](docs/AZURE_DEPLOYMENT.md) for the Phase 4 deployment design using
Azure Container Registry, Azure Container Apps, persistent storage, Qdrant planning, and Ollama
deployment constraints.

## Planned production increments

1. Cross-encoder reranking and chunking experiments guided by evaluation metrics
2. Larger evaluation dataset with answer faithfulness and citation-correctness scoring
3. Authentication, rate limits, and stronger adversarial PDF testing
4. Azure implementation with real credentials, cost controls, image pinning, and monitoring

## Safety

Use public or personally owned documents only. Uploaded PDFs, embeddings, databases, secrets,
virtual environments, caches, and generated model files must not be committed.
