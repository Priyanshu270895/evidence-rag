# Non-Azure Deployment Options

EvidenceRAG does not require Azure. The strongest path right now is a local full demo plus optional
Qdrant Cloud for managed vector search.

## Recommended Path

```text
Local laptop
  -> FastAPI backend
  -> Streamlit UI
  -> local Ollama generation
  -> SQLite metadata and fallback vector search
  -> optional Qdrant Cloud vector search
```

This keeps the complete RAG system working without a paid LLM API or cloud compute bill.

## Local Streamlit Demo

Install the optional UI dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[ui]"
```

Start the API:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

Start the UI in another terminal:

```powershell
.\.venv\Scripts\streamlit.exe run ui\streamlit_app.py
```

Open the URL Streamlit prints, usually `http://localhost:8501`.

## Qdrant Cloud Free Tier

Qdrant Cloud is the best non-Azure vector-store upgrade for this project. You need to create the
account and cluster because it requires your email and API key.

After creating a free cluster, put these values in your local `.env`:

```text
VECTOR_BACKEND=qdrant
QDRANT_URL=https://your-cluster-url
QDRANT_API_KEY=your-api-key
QDRANT_COLLECTION=evidence_rag_chunks
```

Then run:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[qdrant]"
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

Check:

```text
GET /health
```

Expected:

```json
{
  "vector_backend": "qdrant",
  "vector_store": true
}
```

## Streamlit Community Cloud

This can host the UI, but the full local RAG backend still needs to run somewhere else. If you use
Streamlit Community Cloud later:

- app path: `ui/streamlit_app.py`
- install file: `requirements.txt`
- environment variable: `EVIDENCE_RAG_API_URL=https://your-backend-url`

For now, the most reliable demo is local Streamlit plus local FastAPI.

## Why Not Render Or Railway For Full RAG Yet

Render and Railway can host lightweight web services, but the full EvidenceRAG stack includes
Ollama, embeddings, uploads, SQLite/Qdrant, and longer request times. Free or trial tiers may sleep,
have limited RAM, or lose local filesystem state. They are useful later for a lightweight API demo,
not for the strongest full RAG demo today.
