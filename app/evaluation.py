import math
from collections.abc import Mapping, Sequence
from statistics import mean
from typing import Any

MetricValue = int | float


def precision_at_k(relevance: Sequence[MetricValue], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be greater than zero")
    return sum(1 for score in relevance[:k] if score > 0) / k


def recall_at_k(relevance: Sequence[MetricValue], relevant_count: int, k: int) -> float:
    if k <= 0:
        raise ValueError("k must be greater than zero")
    if relevant_count <= 0:
        return 0.0
    hits = sum(1 for score in relevance[:k] if score > 0)
    return min(hits, relevant_count) / relevant_count


def reciprocal_rank(relevance: Sequence[MetricValue]) -> float:
    for rank, score in enumerate(relevance, start=1):
        if score > 0:
            return 1 / rank
    return 0.0


def dcg_at_k(relevance: Sequence[MetricValue], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be greater than zero")
    return sum((2**score - 1) / math.log2(rank + 1) for rank, score in enumerate(relevance[:k], 1))


def ndcg_at_k(relevance: Sequence[MetricValue], relevant_count: int, k: int) -> float:
    if k <= 0:
        raise ValueError("k must be greater than zero")
    if relevant_count <= 0:
        return 0.0
    ideal_relevance = [1] * min(relevant_count, k)
    ideal_score = dcg_at_k(ideal_relevance, k)
    if ideal_score == 0:
        return 0.0
    return dcg_at_k(relevance, k) / ideal_score


def ranking_metrics(
    relevance: Sequence[MetricValue],
    relevant_count: int,
    k: int,
) -> dict[str, float | int]:
    top_k_relevance = list(relevance[:k])
    return {
        "k": k,
        "relevant_count": relevant_count,
        "hits_at_k": sum(1 for score in top_k_relevance if score > 0),
        "precision_at_k": precision_at_k(top_k_relevance, k),
        "recall_at_k": recall_at_k(top_k_relevance, relevant_count, k),
        "mrr": reciprocal_rank(relevance),
        "ndcg_at_k": ndcg_at_k(top_k_relevance, relevant_count, k),
    }


def summarize_metrics(results: Sequence[Mapping[str, Any]]) -> dict[str, float | int]:
    if not results:
        return {
            "cases": 0,
            "mean_precision_at_k": 0.0,
            "mean_recall_at_k": 0.0,
            "mean_mrr": 0.0,
            "mean_ndcg_at_k": 0.0,
        }
    return {
        "cases": len(results),
        "mean_precision_at_k": mean(float(result["precision_at_k"]) for result in results),
        "mean_recall_at_k": mean(float(result["recall_at_k"]) for result in results),
        "mean_mrr": mean(float(result["mrr"]) for result in results),
        "mean_ndcg_at_k": mean(float(result["ndcg_at_k"]) for result in results),
    }


def relevance_for_targets(
    retrieved_items: Sequence[Mapping[str, Any]],
    targets: Sequence[Mapping[str, Any]],
) -> list[int]:
    matched_targets: set[int] = set()
    relevance: list[int] = []
    for item in retrieved_items:
        matched_index = next(
            (
                index
                for index, target in enumerate(targets)
                if index not in matched_targets and retrieved_item_matches(item, target)
            ),
            None,
        )
        if matched_index is None:
            relevance.append(0)
            continue
        matched_targets.add(matched_index)
        relevance.append(1)
    return relevance


def evaluate_retrieved_items(
    retrieved_items: Sequence[Mapping[str, Any]],
    targets: Sequence[Mapping[str, Any]],
    k: int,
) -> dict[str, float | int]:
    relevance = relevance_for_targets(retrieved_items, targets)
    return ranking_metrics(relevance, relevant_count=len(targets), k=k)


def retrieved_item_matches(item: Mapping[str, Any], target: Mapping[str, Any]) -> bool:
    checks = 0
    for target_key, item_keys in (
        ("chunk_id", ("chunk_id", "id")),
        ("id", ("id", "chunk_id")),
        ("document_id", ("document_id",)),
        ("document", ("document", "filename")),
        ("filename", ("filename", "document")),
    ):
        if target_key in target:
            checks += 1
            if _string_value(target[target_key]) != _first_string_value(item, item_keys):
                return False

    if "page" in target:
        checks += 1
        try:
            if int(item.get("page", -1)) != int(target["page"]):
                return False
        except (TypeError, ValueError):
            return False

    contains = target.get("contains") or target.get("text_contains")
    if contains:
        checks += 1
        if _string_value(contains) not in _string_value(item.get("text", "")):
            return False

    return checks > 0


def _first_string_value(item: Mapping[str, Any], keys: Sequence[str]) -> str:
    for key in keys:
        if key in item and item[key] is not None:
            return _string_value(item[key])
    return ""


def _string_value(value: Any) -> str:
    return str(value).strip().casefold()
