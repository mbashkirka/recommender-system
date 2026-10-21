from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from recsys_lab.models.hybrid import MetadataIndex


def deduplicate_preserve_order(items: Sequence[int]) -> list[int]:
    return list(dict.fromkeys(int(x) for x in items))


def genre_jaccard(metadata: MetadataIndex, item_a: int, item_b: int) -> float:
    ia = metadata.item_index.get(int(item_a))
    ib = metadata.item_index.get(int(item_b))
    if ia is None or ib is None:
        return 0.0
    a = metadata.features[ia] > 0
    b = metadata.features[ib] > 0
    union = int(np.logical_or(a, b).sum())
    if union == 0:
        return 0.0
    return float(np.logical_and(a, b).sum() / union)


def intra_list_diversity(metadata: MetadataIndex, items: Sequence[int]) -> float:
    unique = deduplicate_preserve_order(items)
    if len(unique) < 2:
        return 0.0
    values: list[float] = []
    for i in range(len(unique)):
        for j in range(i + 1, len(unique)):
            values.append(1.0 - genre_jaccard(metadata, unique[i], unique[j]))
    return float(np.mean(values)) if values else 0.0


def novelty_bits(
    items: Sequence[int],
    popularity_counts: Mapping[int, int],
    catalog_size: int,
) -> float:
    unique = deduplicate_preserve_order(items)
    if not unique:
        return 0.0
    total_positive = float(sum(int(v) for v in popularity_counts.values()))
    denominator = total_positive + max(int(catalog_size), 1)
    values = [
        -np.log2((float(popularity_counts.get(int(item), 0)) + 1.0) / denominator)
        for item in unique
    ]
    return float(np.mean(values))


@dataclass(frozen=True)
class DiversityNoveltyReranker:
    """Greedy MMR-like re-ranker over an already ordered candidate list.

    The original candidate order is the relevance prior. Diversity penalizes maximum genre overlap
    with already selected items; novelty rewards low train-positive popularity. A zero/zero weight
    pair is an exact no-regression anchor and returns the original order unchanged.
    """

    metadata: MetadataIndex
    popularity_counts: Mapping[int, int]
    diversity_weight: float
    novelty_weight: float
    pool_k: int = 50
    max_per_genre: int = 4

    def __post_init__(self) -> None:
        if self.diversity_weight < 0 or self.novelty_weight < 0:
            raise ValueError("reranking weights must be >= 0")
        if self.pool_k <= 0:
            raise ValueError("pool_k must be > 0")
        if self.max_per_genre < 0:
            raise ValueError("max_per_genre must be >= 0")

    def _primary_genre(self, item_id: int) -> int | None:
        idx = self.metadata.item_index.get(int(item_id))
        if idx is None or self.metadata.features.shape[1] == 0:
            return None
        row = self.metadata.features[idx]
        nz = np.flatnonzero(row > 0)
        return int(nz[0]) if len(nz) else None

    def _novelty_vector(self, items: list[int]) -> np.ndarray:
        total_positive = float(sum(int(v) for v in self.popularity_counts.values()))
        denominator = total_positive + max(len(self.metadata.item_ids), 1)
        raw = np.asarray(
            [
                -np.log2((float(self.popularity_counts.get(int(item), 0)) + 1.0) / denominator)
                for item in items
            ],
            dtype=np.float64,
        )
        std = float(raw.std())
        if std <= 1e-12:
            return np.zeros_like(raw)
        return (raw - float(raw.mean())) / std

    def rerank(self, ordered_candidates: Sequence[int], k: int) -> list[int]:
        if k <= 0:
            return []
        ordered = deduplicate_preserve_order(ordered_candidates)
        if not ordered:
            return []
        if self.diversity_weight == 0.0 and self.novelty_weight == 0.0:
            return ordered

        pool = ordered[: min(self.pool_k, len(ordered))]
        tail = ordered[len(pool) :]
        n = len(pool)
        # Rank prior in [1, 0], so the original order remains the dominant relevance signal.
        relevance = np.ones(n, dtype=np.float64) if n == 1 else 1.0 - np.arange(n) / (n - 1)
        novelty_z = self._novelty_vector(pool)
        item_to_pos = {item: pos for pos, item in enumerate(pool)}

        selected: list[int] = []
        genre_counts: dict[int, int] = {}
        remaining = set(pool)
        target = min(k, len(pool))
        while remaining and len(selected) < target:
            scored: list[tuple[float, int, int, bool]] = []
            for item in remaining:
                pos = item_to_pos[item]
                max_sim = max(
                    (genre_jaccard(self.metadata, item, chosen) for chosen in selected),
                    default=0.0,
                )
                primary = self._primary_genre(item)
                violates = bool(
                    self.max_per_genre > 0
                    and primary is not None
                    and genre_counts.get(primary, 0) >= self.max_per_genre
                )
                utility = (
                    relevance[pos]
                    - self.diversity_weight * max_sim
                    + self.novelty_weight * novelty_z[pos]
                )
                scored.append((float(utility), -pos, int(item), violates))

            feasible = [x for x in scored if not x[3]]
            choices = feasible if feasible else scored  # relax constraint rather than return <K
            best = max(choices, key=lambda x: (x[0], x[1], -x[2]))
            item = int(best[2])
            selected.append(item)
            remaining.remove(item)
            primary = self._primary_genre(item)
            if primary is not None:
                genre_counts[primary] = genre_counts.get(primary, 0) + 1

        # Preserve every candidate for downstream audit/serving; only the prefix is diversified.
        selected_set = set(selected)
        remainder = [item for item in pool if item not in selected_set] + tail
        return selected + remainder
