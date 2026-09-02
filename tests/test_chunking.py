import pytest

from app.chunking import chunk_text


def test_chunking_retains_page_and_respects_size():
    chunks = chunk_text("First sentence. " * 100, page=3, chunk_size=120, overlap=20)
    assert len(chunks) > 1
    assert all(chunk.page == 3 for chunk in chunks)
    assert all(len(chunk.text) <= 120 for chunk in chunks)


def test_overlap_must_be_smaller_than_chunk():
    with pytest.raises(ValueError):
        chunk_text("content", page=1, chunk_size=10, overlap=10)

