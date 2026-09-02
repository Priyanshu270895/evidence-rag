import shutil
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile

from app.config import settings
from app.schemas import AskRequest, AskResponse, Citation, IngestResponse
from app.services import generate_answer, ingest_pdf, retrieve

app = FastAPI(title="EvidenceRAG", version="0.1.0")
UPLOAD_DIR = Path("data/uploads")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/documents", response_model=IngestResponse)
def add_document(file: Annotated[UploadFile, File(...)]) -> IngestResponse:
    if file.content_type != "application/pdf" or not file.filename:
        raise HTTPException(status_code=415, detail="Only PDF files are supported")
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    destination = UPLOAD_DIR / Path(file.filename).name
    with destination.open("wb") as target:
        shutil.copyfileobj(file.file, target)
    chunks = ingest_pdf(destination)
    return IngestResponse(document=destination.name, chunks_created=chunks)


@app.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest) -> AskResponse:
    evidence = retrieve(payload.question, settings.top_k)
    answer = generate_answer(payload.question, evidence)
    citations = [
        Citation(
            document=row["document"],
            page=row["page"],
            chunk_id=row["id"],
            excerpt=row["text"][:300],
            score=row["score"],
        )
        for row in evidence
    ]
    return AskResponse(answer=answer, citations=citations)
