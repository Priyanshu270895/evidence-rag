import sqlite3

from app.store import ChunkStore, utc_now


def document_record(document_id: str = "doc-1") -> dict:
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


def chunk_record(document_id: str = "doc-1", text: str = "alpha evidence") -> dict:
    return {
        "id": "chunk-1",
        "document_id": document_id,
        "document": "sample.pdf",
        "page": 1,
        "chunk_index": 0,
        "text": text,
        "embedding": [1.0, 0.0],
    }


def test_save_document_lists_metadata_and_chunks(tmp_path):
    store = ChunkStore(tmp_path / "chunks.db")
    store.save_document(document_record(), [chunk_record()])

    documents = store.list_documents()
    assert len(documents) == 1
    assert documents[0]["document_id"] == "doc-1"
    assert documents[0]["file_hash"] == "hash-1"
    assert store.get_document_by_hash("hash-1")["filename"] == "sample.pdf"

    chunks = store.chunks_for_document("doc-1")
    assert len(chunks) == 1
    assert chunks[0]["id"] == "chunk-1"


def test_delete_document_removes_chunks_and_keyword_index(tmp_path):
    store = ChunkStore(tmp_path / "chunks.db")
    store.save_document(document_record(), [chunk_record()])

    assert store.keyword_search("alpha", limit=5) == ["chunk-1"]
    assert store.delete_document("doc-1") is True

    assert store.get_document("doc-1") is None
    assert store.chunks_for_document("doc-1") == []
    assert store.keyword_search("alpha", limit=5) == []


def test_replacing_document_rebuilds_document_chunks(tmp_path):
    store = ChunkStore(tmp_path / "chunks.db")
    store.save_document(document_record(), [chunk_record(text="alpha evidence")])
    updated_document = {**document_record(), "chunk_count": 1}
    store.save_document(updated_document, [chunk_record(text="beta evidence")])

    assert store.keyword_search("alpha", limit=5) == []
    assert store.keyword_search("beta", limit=5) == ["chunk-1"]


def test_migrates_phase_zero_chunk_schema_before_creating_indexes(tmp_path):
    database_path = tmp_path / "legacy.db"
    connection = sqlite3.connect(database_path)
    connection.execute(
        """
        CREATE TABLE chunks (
            id TEXT PRIMARY KEY,
            document TEXT NOT NULL,
            page INTEGER NOT NULL,
            text TEXT NOT NULL,
            embedding TEXT NOT NULL
        )
        """
    )
    connection.execute(
        "INSERT INTO chunks(id, document, page, text, embedding) VALUES(?,?,?,?,?)",
        ("legacy-chunk", "legacy.pdf", 2, "legacy alpha text", "[1.0, 0.0]"),
    )
    connection.commit()
    connection.close()

    store = ChunkStore(database_path)

    documents = store.list_documents()
    assert len(documents) == 1
    assert documents[0]["filename"] == "legacy.pdf"
    assert documents[0]["status"] == "legacy"
    assert store.keyword_search("legacy", limit=5) == ["legacy-chunk"]
