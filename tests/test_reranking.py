from __future__ import annotations

import pandas as pd
import pytest

from recsys_lab.models.hybrid import build_metadata_index
from recsys_lab.models.reranking import (
    DiversityNoveltyReranker,
    deduplicate_preserve_order,
    genre_jaccard,
    intra_list_diversity,
    novelty_bits,
)


def _metadata():
    movies = pd.DataFrame(
        [
            (1, "A", "Action"),
            (2, "B", "Action"),
            (3, "C", "Comedy"),
            (4, "D", "Drama"),
            (5, "E", "Romance"),
            (6, "F", "Comedy|Drama"),
        ],
        columns=["item_id", "title", "genres"],
    )
    return build_metadata_index(movies)


def test_deduplicate_preserves_first_occurrence() -> None:
    assert deduplicate_preserve_order([3, 1, 3, 2, 1]) == [3, 1, 2]


def test_genre_jaccard_detects_same_and_disjoint_genres() -> None:
    metadata = _metadata()
    assert genre_jaccard(metadata, 1, 2) == pytest.approx(1.0)
    assert genre_jaccard(metadata, 1, 3) == pytest.approx(0.0)


def test_intra_list_diversity_is_higher_for_mixed_genres() -> None:
    metadata = _metadata()
    same = intra_list_diversity(metadata, [1, 2])
    mixed = intra_list_diversity(metadata, [1, 3, 4])
    assert mixed > same


def test_novelty_rewards_less_popular_items() -> None:
    popularity = {1: 1000, 2: 500, 3: 2, 4: 1}
    popular = novelty_bits([1, 2], popularity, catalog_size=4)
    rare = novelty_bits([3, 4], popularity, catalog_size=4)
    assert rare > popular


def test_zero_weights_are_exact_no_regression_anchor() -> None:
    reranker = DiversityNoveltyReranker(_metadata(), {i: 10 for i in range(1, 7)}, 0.0, 0.0)
    original = [1, 2, 3, 4, 5, 6]
    assert reranker.rerank(original, 4) == original


def test_diversity_weight_breaks_same_genre_cluster() -> None:
    metadata = _metadata()
    reranker = DiversityNoveltyReranker(
        metadata, {i: 10 for i in range(1, 7)}, diversity_weight=1.0, novelty_weight=0.0,
        pool_k=6, max_per_genre=4,
    )
    result = reranker.rerank([1, 2, 3, 4, 5, 6], 3)
    assert 2 not in result[:2]  # after Action item 1, another identical Action is penalized
    assert intra_list_diversity(metadata, result[:3]) > intra_list_diversity(metadata, [1, 2, 3])


def test_novelty_weight_can_promote_rare_candidate() -> None:
    metadata = _metadata()
    popularity = {1: 1000, 2: 900, 3: 800, 4: 1, 5: 1, 6: 1}
    reranker = DiversityNoveltyReranker(
        metadata, popularity, diversity_weight=0.0, novelty_weight=2.0, pool_k=6,
        max_per_genre=0,
    )
    result = reranker.rerank([1, 2, 3, 4, 5, 6], 3)
    assert any(item in {4, 5, 6} for item in result[:2])


def test_constraint_relaxes_instead_of_returning_short_list() -> None:
    metadata = build_metadata_index(
        pd.DataFrame(
            [(1, "A", "Action"), (2, "B", "Action"), (3, "C", "Action")],
            columns=["item_id", "title", "genres"],
        )
    )
    reranker = DiversityNoveltyReranker(
        metadata, {1: 3, 2: 2, 3: 1}, diversity_weight=0.2, novelty_weight=0.1,
        pool_k=3, max_per_genre=1,
    )
    assert len(reranker.rerank([1, 2, 3], 3)[:3]) == 3


def test_quality_guardrail_gate_accepts_and_rejects_expected_cases() -> None:
    from recsys_lab.evaluation.reranking import _passes_quality_guardrails

    baseline = {"ndcg": 0.20, "recall": 0.10}
    assert _passes_quality_guardrails({"ndcg": 0.195, "recall": 0.10}, baseline, 0.97, 0.95)
    assert not _passes_quality_guardrails({"ndcg": 0.18, "recall": 0.10}, baseline, 0.97, 0.95)
    assert not _passes_quality_guardrails({"ndcg": 0.20, "recall": 0.09}, baseline, 0.97, 0.95)
