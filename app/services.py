import hashlib
from functools import lru_cache
from pathlib import Path

import httpx
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

from app.chunking import chunk_text
from app.config import settings
from app.retrieval import reciprocal_rank_fusion
from app.store import ChunkStore


@lru_cache
def embedder() -> SentenceTransformer:
    return SentenceTransformer(settings.embedding_model)


@lru_cache
def store() -> ChunkStore:
    return ChunkStore(settings.database_path)


def ingest_pdf(path: Path) -> int:
    reader = PdfReader(path)
    records = []
    for page_number, page in enumerate(reader.pages, start=1):
        for chunk in chunk_text(page.extract_text() or "", page_number):
            digest = hashlib.sha256(f"{path.name}:{page_number}:{chunk.index}:{chunk.text}".encode()).hexdigest()[:20]
            records.append({"id": digest, "document": path.name, "page": page_number, "text": chunk.text})
    if not records:
        return 0
    vectors = embedder().encode([r["text"] for r in records], normalize_embeddings=True).tolist()
    for record, vector in zip(records, vectors, strict=True):
        record["embedding"] = vector
    store().add(records)
    return len(records)


def retrieve(question: str, limit: int) -> list[dict]:
    query_vector = embedder().encode(question, normalize_embeddings=True).tolist()
    expanded_limit = max(limit * 4, 20)
    vector_ids = store().vector_search(query_vector, expanded_limit)
    keyword_ids = store().keyword_search(question, expanded_limit)
    fused = reciprocal_rank_fusion([vector_ids, keyword_ids])[:limit]
    rows = store().get([item_id for item_id, _ in fused])
    return [{**dict(rows[item_id]), "score": score} for item_id, score in fused if item_id in rows]


def generate_answer(question: str, evidence: list[dict]) -> str:
    if not evidence:
        return "I don't have enough evidence in the indexed documents to answer that question."
    context = "\n\n".join(
        f"[SOURCE {index}: {row['document']}, page {row['page']}]\n{row['text']}"
        for index, row in enumerate(evidence, start=1)
    )
    prompt = f"""Answer only from the supplied evidence. Cite claims using [SOURCE N].
If the evidence is insufficient, say so clearly. Do not invent facts.

Question: {question}

Evidence:
{context}
"""
    try:
        response = httpx.post(
            f"{settings.ollama_base_url}/api/generate",
            json={"model": settings.ollama_model, "prompt": prompt, "stream": False},
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["response"].strip()
    except httpx.HTTPError:
        return "Relevant evidence was retrieved, but the local language model is unavailable. Start Ollama and try again."

