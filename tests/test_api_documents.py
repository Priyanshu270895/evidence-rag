from fastapi.testclient import TestClient

from app import main
from app.store import utc_now


def api_document(document_id: str = "doc-1") -> dict:
    timestamp = utc_now()
    return {
        "document_id": document_id,
        "filename": "sample.pdf",
        "file_hash": "hash-1",
        "storage_path": "data/uploads/doc-1.pdf",
        "size_bytes": 42,
        "page_count": 1,
        "chunk_count": 1,
        "status": "indexed",
        "chunk_size": 900,
        "chunk_overlap": 120,
        "created_at": timestamp,
        "updated_at": timestamp,
    }


def test_upload_document_returns_lifecycle_metadata(monkeypatch, tmp_path):
    def fake_ingest(*_args, **_kwargs):
        document = api_document()
        return {
            **document,
            "document": document["filename"],
            "chunks_created": 1,
            "duplicate": False,
        }

    monkeypatch.setattr(main.settings, "upload_dir", tmp_path)
    monkeypatch.setattr(main, "ingest_pdf", fake_ingest)

    client = TestClient(main.app)
    response = client.post(
        "/documents",
        files={"file": ("sample.pdf", b"%PDF-1.4\n%%EOF", "application/pdf")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["document_id"] == "doc-1"
    assert payload["document"] == "sample.pdf"
    assert payload["chunks_created"] == 1
    assert payload["duplicate"] is False


def test_document_lifecycle_endpoints(monkeypatch):
    monkeypatch.setattr(main, "list_documents", lambda: [api_document()])
    monkeypatch.setattr(main, "get_document", lambda document_id: api_document(document_id))
    monkeypatch.setattr(main, "delete_document", lambda _document_id: True)
    monkeypatch.setattr(
        main,
        "get_document_chunks",
        lambda document_id: [
            {
                "chunk_id": "chunk-1",
                "document_id": document_id,
                "document": "sample.pdf",
                "page": 1,
                "chunk_index": 0,
                "text": "alpha evidence",
            }
        ],
    )

    client = TestClient(main.app)

    assert client.get("/documents").json()[0]["document_id"] == "doc-1"
    assert client.get("/documents/doc-1").json()["filename"] == "sample.pdf"
    assert client.get("/documents/doc-1/chunks").json()[0]["chunk_id"] == "chunk-1"
    assert client.delete("/documents/doc-1").json() == {"document_id": "doc-1", "deleted": True}
