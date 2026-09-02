from app.retrieval import reciprocal_rank_fusion
from app.store import ChunkStore


def test_rrf_rewards_documents_found_by_both_retrievers():
    result = reciprocal_rank_fusion([["a", "b"], ["b", "c"]])
    assert result[0][0] == "b"


def test_keyword_search_handles_natural_language_punctuation(tmp_path):
    store = ChunkStore(tmp_path / "chunks.db")
    store.add(
        [
            {
                "id": "chunk-1",
                "document": "sample.pdf",
                "page": 1,
                "text": "This is a dummy PDF file.",
                "embedding": [1.0, 0.0],
            }
        ]
    )

    assert store.keyword_search("Dummy PDF?", limit=5) == ["chunk-1"]
    assert store.keyword_search("What does the document say?", limit=5) == []


def test_keyword_index_updates_when_chunk_is_replaced(tmp_path):
    store = ChunkStore(tmp_path / "chunks.db")
    base_row = {
        "id": "chunk-1",
        "document": "sample.pdf",
        "page": 1,
        "embedding": [1.0, 0.0],
    }

    store.add([{**base_row, "text": "alpha evidence"}])
    store.add([{**base_row, "text": "beta evidence"}])

    assert store.keyword_search("alpha", limit=5) == []
    assert store.keyword_search("beta", limit=5) == ["chunk-1"]
