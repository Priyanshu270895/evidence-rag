from fastapi.testclient import TestClient

from app import main
from app.services import EmbeddingModelUnavailableError


def test_add_document_returns_503_when_embedding_model_is_unavailable(monkeypatch, tmp_path):
    def fail_ingest(_path):
        raise EmbeddingModelUnavailableError("Embedding model is unavailable.")

    monkeypatch.setattr(main, "UPLOAD_DIR", tmp_path)
    monkeypatch.setattr(main, "ingest_pdf", fail_ingest)

    client = TestClient(main.app)
    response = client.post(
        "/documents",
        files={"file": ("sample.pdf", b"%PDF-1.4\n%%EOF", "application/pdf")},
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "Embedding model is unavailable."


def test_ask_returns_503_when_embedding_model_is_unavailable(monkeypatch):
    def fail_retrieve(_question, _limit):
        raise EmbeddingModelUnavailableError("Embedding model is unavailable.")

    monkeypatch.setattr(main, "retrieve", fail_retrieve)

    client = TestClient(main.app)
    response = client.post("/ask", json={"question": "What is in the document?"})

    assert response.status_code == 503
    assert response.json()["detail"] == "Embedding model is unavailable."
