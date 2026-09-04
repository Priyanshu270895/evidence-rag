import pytest

from app.evaluation import (
    evaluate_retrieved_items,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    relevance_for_targets,
    retrieved_item_matches,
)


def test_ranking_metrics_at_k() -> None:
    relevance = [1, 0, 1]

    assert precision_at_k(relevance, 2) == 0.5
    assert recall_at_k(relevance, relevant_count=2, k=3) == 1.0
    assert reciprocal_rank([0, 0, 1]) == pytest.approx(1 / 3)
    assert ndcg_at_k(relevance, relevant_count=2, k=3) == pytest.approx(0.91972, rel=1e-4)


def test_target_matching_supports_common_evidence_labels() -> None:
    item = {
        "id": "chunk-1",
        "document_id": "doc-1",
        "document": "guide.pdf",
        "page": 3,
        "text": "The refund window is 30 days.",
    }

    assert retrieved_item_matches(item, {"chunk_id": "chunk-1"})
    assert retrieved_item_matches(item, {"filename": "guide.pdf", "page": 3})
    assert retrieved_item_matches(item, {"document_id": "doc-1", "contains": "refund"})
    assert not retrieved_item_matches(item, {"document_id": "doc-2"})


def test_relevance_deduplicates_targets() -> None:
    retrieved = [
        {
            "id": "chunk-1",
            "document_id": "doc-1",
            "document": "guide.pdf",
            "page": 2,
            "text": "OAuth setup instructions.",
        },
        {
            "id": "chunk-2",
            "document_id": "doc-1",
            "document": "guide.pdf",
            "page": 2,
            "text": "More OAuth setup instructions.",
        },
    ]
    targets = [{"document_id": "doc-1", "page": 2, "contains": "OAuth"}]

    assert relevance_for_targets(retrieved, targets) == [1, 0]
    metrics = evaluate_retrieved_items(retrieved, targets, k=2)
    assert metrics["recall_at_k"] == 1.0
    assert metrics["precision_at_k"] == 0.5
