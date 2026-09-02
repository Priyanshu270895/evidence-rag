import hashlib
import re
from functools import lru_cache
from pathlib import Path

import httpx
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

from app.chunking import chunk_text
from app.config import settings
from app.retrieval import reciprocal_rank_fusion
from app.store import ChunkStore

SOURCE_CITATION_PATTERN = re.compile(r"\[SOURCE (?P<index>\d+)\]")


class EmbeddingModelUnavailableError(RuntimeError):
    pass


@lru_cache
def embedder() -> SentenceTransformer:
    try:
        return SentenceTransformer(
            settings.embedding_model,
            local_files_only=settings.embedding_model_local_files_only,
        )
    except (OSError, RuntimeError) as exc:
        raise EmbeddingModelUnavailableError(
            "Embedding model is unavailable. Run `python scripts/cache_embedding_model.py` "
            "from the project virtual environment, then retry."
        ) from exc


@lru_cache
def store() -> ChunkStore:
    return ChunkStore(settings.database_path)


def ingest_pdf(path: Path) -> int:
    reader = PdfReader(path)
    records = []
    for page_number, page in enumerate(reader.pages, start=1):
        for chunk in chunk_text(page.extract_text() or "", page_number):
            digest = hashlib.sha256(
                f"{path.name}:{page_number}:{chunk.index}:{chunk.text}".encode()
            ).hexdigest()[:20]
            records.append(
                {"id": digest, "document": path.name, "page": page_number, "text": chunk.text}
            )
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


def has_valid_source_citation(answer: str, evidence_count: int) -> bool:
    for match in SOURCE_CITATION_PATTERN.finditer(answer):
        if 1 <= int(match.group("index")) <= evidence_count:
            return True
    return False


def grounded_fallback(evidence: list[dict]) -> str:
    top = evidence[0]["text"].strip().replace("\n", " ")
    if len(top) > 500:
        top = f"{top[:497]}..."
    return f"The retrieved evidence says: {top} [SOURCE 1]"


def generate_answer(question: str, evidence: list[dict]) -> str:
    if not evidence:
        return "I don't have enough evidence in the indexed documents to answer that question."
    context = "\n\n".join(
        f"[SOURCE {index}: {row['document']}, page {row['page']}]\n{row['text']}"
        for index, row in enumerate(evidence, start=1)
    )
    prompt = f"""Answer only from the supplied evidence.
Every factual sentence must cite one supplied source using [SOURCE N].
If the evidence is insufficient, say so clearly and cite the closest relevant source.
Do not infer, generalize, or explain beyond the evidence.
If the evidence is only a short phrase, answer with only that phrase and its citation.

Question: {question}

Evidence:
{context}
"""
    try:
        response = httpx.post(
            f"{settings.ollama_base_url}/api/generate",
            json={
                "model": settings.ollama_model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0, "num_predict": 256},
            },
            timeout=120,
        )
        response.raise_for_status()
        answer = response.json()["response"].strip()
        if not has_valid_source_citation(answer, len(evidence)):
            return grounded_fallback(evidence)
        return answer
    except httpx.HTTPError:
        return "Relevant evidence was retrieved, but the local language model is unavailable. Start Ollama and try again."
