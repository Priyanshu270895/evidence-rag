from app.retrieval import reciprocal_rank_fusion


def test_rrf_rewards_documents_found_by_both_retrievers():
    result = reciprocal_rank_fusion([["a", "b"], ["b", "c"]])
    assert result[0][0] == "b"
