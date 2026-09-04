import hashlib
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from uuid import uuid4

import httpx
from pypdf import PdfReader
from pypdf.errors import PdfReadError
from sentence_transformers import SentenceTransformer

from app.chunking import chunk_text
from app.config import settings
from app.retrieval import reciprocal_rank_fusion
from app.store import ChunkStore, utc_now

SOURCE_CITATION_PATTERN = re.compile(r"\[SOURCE (?P<index>\d+)\]")


class DocumentIngestionError(RuntimeError):
    pass


class DocumentNotFoundError(RuntimeError):
    pass


class DocumentSourceUnavailableError(RuntimeError):
    pass


class EmbeddingModelUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True)
class PreparedChunks:
    page_count: int
    rows: list[dict]


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


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def row_to_dict(row) -> dict:
    return dict(row)


def list_documents() -> list[dict]:
    return [row_to_dict(row) for row in store().list_documents()]


def get_document(document_id: str) -> dict | None:
    row = store().get_document(document_id)
    return row_to_dict(row) if row is not None else None


def delete_document(document_id: str) -> bool:
    document = store().get_document(document_id)
    deleted = store().delete_document(document_id)
    if deleted and document and document["storage_path"]:
        Path(document["storage_path"]).unlink(missing_ok=True)
    return deleted


def get_document_chunks(document_id: str) -> list[dict]:
    if store().get_document(document_id) is None:
        raise DocumentNotFoundError(f"Document '{document_id}' was not found.")
    rows = store().chunks_for_document(document_id)
    return [
        {
            "chunk_id": row["id"],
            "document_id": row["document_id"],
            "document": row["document"],
            "page": row["page"],
            "chunk_index": row["chunk_index"],
            "text": row["text"],
        }
        for row in rows
    ]


def ingest_pdf(
    path: Path,
    original_filename: str | None = None,
    cleanup_source_on_duplicate: bool = False,
) -> dict:
    filename = Path(original_filename or path.name).name
    file_hash = file_sha256(path)
    existing = store().get_document_by_hash(file_hash)
    if existing is not None:
        if cleanup_source_on_duplicate:
            path.unlink(missing_ok=True)
        document = row_to_dict(existing)
        return {
            **document,
            "document": document["filename"],
            "chunks_created": 0,
            "duplicate": True,
        }

    document_id = uuid4().hex
    final_path = settings.upload_dir / f"{document_id}.pdf"
    prepared = prepare_chunks(path, document_id, filename)
    timestamp = utc_now()
    status = "indexed" if prepared.rows else "empty"
    document = {
        "document_id": document_id,
        "filename": filename,
        "file_hash": file_hash,
        "storage_path": str(final_path),
        "size_bytes": path.stat().st_size,
        "page_count": prepared.page_count,
        "chunk_count": len(prepared.rows),
        "status": status,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "created_at": timestamp,
        "updated_at": timestamp,
    }

    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    path.replace(final_path)
    try:
        store().save_document(document, prepared.rows)
    except Exception:
        final_path.unlink(missing_ok=True)
        raise
    return {
        **document,
        "document": filename,
        "chunks_created": len(prepared.rows),
        "duplicate": False,
    }


def reindex_document(document_id: str) -> dict:
    existing = store().get_document(document_id)
    if existing is None:
        raise DocumentNotFoundError(f"Document '{document_id}' was not found.")
    source_path = Path(existing["storage_path"])
    if not source_path.exists():
        raise DocumentSourceUnavailableError(
            f"Stored PDF for document '{document_id}' is missing from disk."
        )

    prepared = prepare_chunks(source_path, document_id, existing["filename"])
    timestamp = utc_now()
    document = {
        **row_to_dict(existing),
        "page_count": prepared.page_count,
        "chunk_count": len(prepared.rows),
        "status": "indexed" if prepared.rows else "empty",
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "updated_at": timestamp,
    }
    store().save_document(document, prepared.rows)
    return {
        **document,
        "document": document["filename"],
        "chunks_created": len(prepared.rows),
        "duplicate": False,
    }


def prepare_chunks(path: Path, document_id: str, filename: str) -> PreparedChunks:
    try:
        reader = PdfReader(path)
    except PdfReadError as exc:
        raise DocumentIngestionError("The uploaded file could not be read as a PDF.") from exc

    records = []
    for page_number, page in enumerate(reader.pages, start=1):
        for chunk in chunk_text(
            page.extract_text() or "",
            page_number,
            chunk_size=settings.chunk_size,
            overlap=settings.chunk_overlap,
        ):
            digest = hashlib.sha256(
                f"{document_id}:{page_number}:{chunk.index}:{chunk.text}".encode()
            ).hexdigest()[:20]
            records.append(
                {
                    "id": digest,
                    "document_id": document_id,
                    "document": filename,
                    "page": page_number,
                    "chunk_index": chunk.index,
                    "text": chunk.text,
                }
            )
    if not records:
        return PreparedChunks(page_count=len(reader.pages), rows=[])

    vectors = embedder().encode([r["text"] for r in records], normalize_embeddings=True).tolist()
    for record, vector in zip(records, vectors, strict=True):
        record["embedding"] = vector
    return PreparedChunks(page_count=len(reader.pages), rows=records)


def retrieve(question: str, limit: int, document_ids: list[str] | None = None) -> list[dict]:
    query_vector = embedder().encode(question, normalize_embeddings=True).tolist()
    expanded_limit = max(limit * 4, 20)
    vector_ids = store().vector_search(query_vector, expanded_limit, document_ids=document_ids)
    keyword_ids = store().keyword_search(question, expanded_limit, document_ids=document_ids)
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
