from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from math import log2


@dataclass(frozen=True)
class UserRankingMetrics:
    recall: float
    precision: float
    hit_rate: float
    ndcg: float
    map: float
    mrr: float
    relevant_count: int
    recommended_count: int
    hits: int

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


def _deduplicate_preserve_order(items: Iterable[int]) -> list[int]:
    return list(dict.fromkeys(int(item) for item in items))


def ranking_metrics(
    recommendations: Sequence[int] | Iterable[int],
    relevant_items: set[int] | Sequence[int] | Iterable[int],
    k: int,
) -> UserRankingMetrics:
    """Compute standard binary-relevance Top-K metrics for one user.

    Duplicate recommendations are removed while preserving their first position. Ground truth is
    treated as a set. MAP@K uses min(|relevant|, K) as the denominator, which is the common Top-K
    recommendation convention when more relevant items exist than can be returned.
    """
    if k <= 0:
        raise ValueError("k must be > 0")

    recs = _deduplicate_preserve_order(recommendations)[:k]
    relevant = {int(item) for item in relevant_items}
    if not relevant:
        return UserRankingMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, len(recs), 0)

    hit_flags = [1 if item in relevant else 0 for item in recs]
    hits = sum(hit_flags)

    recall = hits / len(relevant)
    precision = hits / len(recs) if recs else 0.0
    hit_rate = float(hits > 0)

    dcg = sum(flag / log2(rank + 1) for rank, flag in enumerate(hit_flags, start=1))
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / log2(rank + 1) for rank in range(1, ideal_hits + 1))
    ndcg = dcg / idcg if idcg > 0 else 0.0

    running_hits = 0
    precision_sum = 0.0
    first_hit_rank: int | None = None
    for rank, flag in enumerate(hit_flags, start=1):
        if flag:
            running_hits += 1
            precision_sum += running_hits / rank
            if first_hit_rank is None:
                first_hit_rank = rank
    average_precision = precision_sum / min(len(relevant), k)
    mrr = 1.0 / first_hit_rank if first_hit_rank is not None else 0.0

    return UserRankingMetrics(
        recall=float(recall),
        precision=float(precision),
        hit_rate=float(hit_rate),
        ndcg=float(ndcg),
        map=float(average_precision),
        mrr=float(mrr),
        relevant_count=len(relevant),
        recommended_count=len(recs),
        hits=int(hits),
    )
