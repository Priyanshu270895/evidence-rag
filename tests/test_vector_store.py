from uuid import UUID

from app.store import ChunkStore
from app.vector_store import SQLiteVectorIndex, _qdrant_point_id


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
