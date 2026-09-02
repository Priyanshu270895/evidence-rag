from collections import defaultdict


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, item_id in enumerate(ranking, start=1):
            scores[item_id] += 1 / (k + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))

