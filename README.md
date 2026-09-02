# EvidenceRAG

A local-first RAG service that answers questions from PDFs with page-level citations. It uses
hybrid retrieval (semantic + SQLite FTS5), reciprocal-rank fusion, a local embedding model,
and an Ollama-hosted language model.

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

## Tests

```powershell
pytest
ruff check .
```

## Planned production increments

1. Evaluation set with retrieval Recall@K, MRR, faithfulness, citation accuracy, and latency
2. Cross-encoder reranking, document lifecycle, deduplication, and incremental indexing
3. Streamlit UI, tracing, structured logs, and prompt-injection defenses
4. Docker Compose with Qdrant, CI/CD, and Azure deployment

## Safety

Use public or personally owned documents only. Uploaded PDFs and embeddings stay on the local
machine in this version.
