from pydantic import BaseModel, Field


class Citation(BaseModel):
    document: str
    page: int
    chunk_id: str
    excerpt: str
    score: float


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]


class IngestResponse(BaseModel):
    document: str
    chunks_created: int

