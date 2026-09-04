from datetime import datetime

from pydantic import BaseModel, Field


class DocumentSummary(BaseModel):
    document_id: str
    filename: str
    file_hash: str
    size_bytes: int
    page_count: int
    chunk_count: int
    status: str
    chunk_size: int
    chunk_overlap: int
    created_at: datetime
    updated_at: datetime


class Citation(BaseModel):
    document_id: str
    document: str
    page: int
    chunk_id: str
    excerpt: str
    score: float


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    document_ids: list[str] | None = None


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]


class IngestResponse(DocumentSummary):
    document: str
    chunks_created: int
    duplicate: bool


class DeleteDocumentResponse(BaseModel):
    document_id: str
    deleted: bool


class ChunkResponse(BaseModel):
    chunk_id: str
    document_id: str
    document: str
    page: int
    chunk_index: int
    text: str
