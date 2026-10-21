from math import isclose, log2

from recsys_lab.evaluation.metrics import ranking_metrics


def test_ranking_metrics_known_example() -> None:
    metrics = ranking_metrics([1, 2, 3], {2, 4}, 3)
    assert metrics.hits == 1
    assert isclose(metrics.recall, 0.5)
    assert isclose(metrics.precision, 1 / 3)
    assert metrics.hit_rate == 1.0
    expected_ndcg = (1 / log2(3)) / (1 + 1 / log2(3))
    assert isclose(metrics.ndcg, expected_ndcg)
    assert isclose(metrics.map, 0.25)
    assert isclose(metrics.mrr, 0.5)


def test_ranking_metrics_perfect_top_two() -> None:
    metrics = ranking_metrics([2, 4, 1], {2, 4}, 3)
    assert metrics.recall == 1.0
    assert isclose(metrics.precision, 2 / 3)
    assert metrics.ndcg == 1.0
    assert metrics.map == 1.0
    assert metrics.mrr == 1.0


def test_ranking_metrics_deduplicates_recommendations() -> None:
    metrics = ranking_metrics([2, 2, 3], {2}, 3)
    assert metrics.hits == 1
    assert metrics.recommended_count == 2


def test_ranking_metrics_empty_ground_truth_is_zero() -> None:
    metrics = ranking_metrics([1, 2], set(), 2)
    assert metrics.recall == 0.0
    assert metrics.ndcg == 0.0
    assert metrics.relevant_count == 0
