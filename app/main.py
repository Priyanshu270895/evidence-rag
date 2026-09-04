import sqlite3
from pathlib import Path
from typing import Annotated
from uuid import uuid4

import httpx
from fastapi import FastAPI, File, HTTPException, UploadFile

from app.config import settings
from app.schemas import (
    AskRequest,
    AskResponse,
    ChunkResponse,
    Citation,
    DeleteDocumentResponse,
    DocumentSummary,
    IngestResponse,
)
from app.services import (
    DocumentIngestionError,
    DocumentNotFoundError,
    DocumentSourceUnavailableError,
    EmbeddingModelUnavailableError,
    delete_document,
    generate_answer,
    get_document,
    get_document_chunks,
    ingest_pdf,
    list_documents,
    reindex_document,
    retrieve,
    store,
)

app = FastAPI(title="EvidenceRAG", version="0.2.0")


@app.get("/health")
def health() -> dict[str, object]:
    sqlite_ok = False
    ollama_ok = False
    try:
        sqlite_ok = store().health_check()
    except (OSError, sqlite3.Error):
        sqlite_ok = False
    try:
        response = httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=1.5)
        ollama_ok = response.status_code == 200
    except httpx.HTTPError:
        ollama_ok = False

    return {
        "status": "ok" if sqlite_ok else "degraded",
        "sqlite": sqlite_ok,
        "ollama": ollama_ok,
        "embedding_model": settings.embedding_model,
        "generation_model": settings.ollama_model,
    }


@app.post("/documents", response_model=IngestResponse)
def add_document(file: Annotated[UploadFile, File(...)]) -> IngestResponse:
    if file.content_type != "application/pdf" or not file.filename:
        raise HTTPException(status_code=415, detail="Only PDF files are supported")

    temp_path = save_upload(file)
    try:
        result = ingest_pdf(
            temp_path,
            original_filename=file.filename,
            cleanup_source_on_duplicate=True,
        )
    except EmbeddingModelUnavailableError as exc:
        temp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except DocumentIngestionError as exc:
        temp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return IngestResponse.model_validate(result)


@app.get("/documents", response_model=list[DocumentSummary])
def list_indexed_documents() -> list[DocumentSummary]:
    return [DocumentSummary.model_validate(document) for document in list_documents()]


@app.get("/documents/{document_id}", response_model=DocumentSummary)
def get_indexed_document(document_id: str) -> DocumentSummary:
    document = get_document(document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return DocumentSummary.model_validate(document)


@app.delete("/documents/{document_id}", response_model=DeleteDocumentResponse)
def remove_document(document_id: str) -> DeleteDocumentResponse:
    deleted = delete_document(document_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found")
    return DeleteDocumentResponse(document_id=document_id, deleted=True)


@app.post("/documents/{document_id}/reindex", response_model=IngestResponse)
def reindex_indexed_document(document_id: str) -> IngestResponse:
    try:
        result = reindex_document(document_id)
    except DocumentNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except DocumentSourceUnavailableError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except EmbeddingModelUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return IngestResponse.model_validate(result)


@app.get("/documents/{document_id}/chunks", response_model=list[ChunkResponse])
def list_document_chunks(document_id: str) -> list[ChunkResponse]:
    try:
        chunks = get_document_chunks(document_id)
    except DocumentNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return [ChunkResponse.model_validate(chunk) for chunk in chunks]


@app.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest) -> AskResponse:
    try:
        evidence = retrieve(payload.question, settings.top_k, document_ids=payload.document_ids)
    except EmbeddingModelUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    answer = generate_answer(payload.question, evidence)
    citations = [
        Citation(
            document_id=row["document_id"],
            document=row["document"],
            page=row["page"],
            chunk_id=row["id"],
            excerpt=row["text"][:300],
            score=row["score"],
        )
        for row in evidence
    ]
    return AskResponse(answer=answer, citations=citations)


def save_upload(file: UploadFile) -> Path:
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    destination = settings.upload_dir / f"{uuid4().hex}.upload.pdf"
    size = 0
    try:
        with destination.open("wb") as target:
            while chunk := file.file.read(1024 * 1024):
                size += len(chunk)
                if size > settings.max_upload_bytes:
                    raise HTTPException(status_code=413, detail="Uploaded PDF is too large")
                target.write(chunk)
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    return destination
