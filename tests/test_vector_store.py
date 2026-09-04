from uuid import UUID

from pydantic import SecretStr

from app import vector_store
from app.store import ChunkStore
from app.vector_store import (
    QdrantVectorIndex,
    SQLiteVectorIndex,
    _qdrant_point_id,
    qdrant_client_kwargs,
)


def document_record() -> dict:
    return {
        "document_id": "doc-1",
        "filename": "sample.pdf",
        "file_hash": "hash-1",
        "storage_path": "data/uploads/doc-1.pdf",
        "size_bytes": 42,
        "page_count": 1,
        "chunk_count": 1,
        "status": "indexed",
        "chunk_size": 900,
        "chunk_overlap": 120,
        "created_at": "2026-09-04T00:00:00Z",
        "updated_at": "2026-09-04T00:00:00Z",
    }


def chunk_record() -> dict:
    return {
        "id": "chunk-1",
        "document_id": "doc-1",
        "document": "sample.pdf",
        "page": 1,
        "chunk_index": 0,
        "text": "alpha evidence",
        "embedding": [1.0, 0.0],
    }


def test_sqlite_vector_index_delegates_to_chunk_store(tmp_path):
    store = ChunkStore(tmp_path / "chunks.db")
    store.save_document(document_record(), [chunk_record()])
    index = SQLiteVectorIndex(store)

    assert index.health_check() is True
    assert index.search([1.0, 0.0], limit=5) == ["chunk-1"]


def test_qdrant_point_id_is_stable_uuid() -> None:
    first = _qdrant_point_id("chunk-1")
    second = _qdrant_point_id("chunk-1")

    assert first == second
    assert UUID(first)


def test_qdrant_client_kwargs_include_api_key_when_configured(monkeypatch) -> None:
    monkeypatch.setattr(vector_store.settings, "qdrant_timeout_seconds", 3.5)
    monkeypatch.setattr(vector_store.settings, "qdrant_api_key", SecretStr("secret-key"))

    kwargs = qdrant_client_kwargs("https://example.cloud.qdrant.io")

    assert kwargs == {
        "url": "https://example.cloud.qdrant.io",
        "timeout": 3.5,
        "api_key": "secret-key",
    }


def test_qdrant_client_kwargs_omit_empty_api_key(monkeypatch) -> None:
    monkeypatch.setattr(vector_store.settings, "qdrant_timeout_seconds", 10.0)
    monkeypatch.setattr(vector_store.settings, "qdrant_api_key", SecretStr(""))

    kwargs = qdrant_client_kwargs("http://localhost:6333")

    assert kwargs == {"url": "http://localhost:6333", "timeout": 10.0}


def test_qdrant_vector_index_uses_query_points(monkeypatch) -> None:
    class FakePoint:
        def __init__(self, payload: dict):
            self.payload = payload

    class FakeQueryResponse:
        def __init__(self):
            self.points = [FakePoint({"chunk_id": "chunk-1"})]

    class FakeClient:
        query_call: dict | None = None

        def __init__(self, **_kwargs):
            return None

        def collection_exists(self, collection_name: str) -> bool:
            return collection_name == "chunks"

        def query_points(self, **kwargs):
            FakeClient.query_call = kwargs
            return FakeQueryResponse()

    import qdrant_client

    monkeypatch.setattr(qdrant_client, "QdrantClient", FakeClient)
    index = QdrantVectorIndex("https://example.cloud.qdrant.io", "chunks")

    assert index.search([1.0, 0.0], limit=3) == ["chunk-1"]
    assert FakeClient.query_call is not None
    assert FakeClient.query_call["collection_name"] == "chunks"
    assert FakeClient.query_call["query"] == [1.0, 0.0]
    assert FakeClient.query_call["limit"] == 3
