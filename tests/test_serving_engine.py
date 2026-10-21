from __future__ import annotations

from pathlib import Path

import numpy as np

from recsys_lab.serving.engine import RecommendationEngine


def make_bundle(path: Path, challenger_eligible: bool = False) -> Path:
    all_item_ids = np.array([10, 11, 12, 13, 14], dtype=np.int64)
    np.savez_compressed(
        path,
        user_ids=np.array([1, 2], dtype=np.int64),
        svd_item_ids=all_item_ids,
        user_components=np.array([[1.0, 0.1], [0.2, 1.0]]),
        singular_values=np.array([1.0, 0.5]),
        item_components=np.array([[1.0, 0.0], [0.8, 0.1], [0.0, 1.0], [0.1, 0.8], [0.4, 0.4]]),
        svd_popularity_z=np.array([1.2, 0.8, 0.4, 0.1, -0.2]),
        all_item_ids=all_item_ids,
        metadata_features=np.array([[1,0,0],[1,0,0],[0,1,0],[0,1,0],[0,0,1]], dtype=np.float32),
        feature_names=np.array(["Action", "Drama", "Comedy"]),
        all_popularity_z=np.array([1.2, 0.8, 0.4, 0.1, -0.2]),
        seen_user_ids=np.array([1, 2], dtype=np.int64),
        seen_item_ids=np.array([10, 12], dtype=np.int64),
        positive_user_ids=np.array([1, 2], dtype=np.int64),
        positive_item_ids=np.array([10, 12], dtype=np.int64),
        rank=np.array([2], dtype=np.int32),
        popularity_blend=np.array([0.35]),
        metadata_blend=np.array([0.0]),
        sparse_positive_threshold=np.array([0], dtype=np.int32),
        sparse_metadata_multiplier=np.array([1.5]),
        cold_item_penalty=np.array([0.25]),
        movie_titles=np.array(["A", "B", "C", "D", "E"]),
        movie_genres=np.array(["Action", "Action", "Drama", "Drama", "Comedy"]),
        positive_popularity_counts=np.array([100, 80, 60, 40, 20], dtype=np.int64),
        ltr_feature_names=np.array([
            "retrieval_score", "reciprocal_rank", "warm_base_score", "metadata_score",
            "popularity_z", "log_positive_popularity", "log_user_seen_count",
            "log_user_positive_count", "genre_overlap", "is_cold_item",
        ]),
        ltr_scaler_mean=np.zeros(10),
        ltr_scaler_scale=np.ones(10),
        ltr_coefficients=np.zeros(10),
        ltr_intercept=np.array([0.0]),
        ltr_c=np.array([0.1]),
        ltr_blend=np.array([0.0]),
        m6_deployed_diversity_weight=np.array([0.0]),
        m6_deployed_novelty_weight=np.array([0.0]),
        m6_development_diversity_weight=np.array([0.5]),
        m6_development_novelty_weight=np.array([0.05]),
        m6_pool_k=np.array([5], dtype=np.int32),
        m6_primary_k=np.array([3], dtype=np.int32),
        m6_max_per_genre=np.array([2], dtype=np.int32),
        challenger_eligible=np.array([int(challenger_eligible)], dtype=np.int8),
    )
    return path


def test_engine_loads_and_fingerprints(tmp_path: Path) -> None:
    engine = RecommendationEngine(make_bundle(tmp_path / "bundle.npz"))
    assert len(engine.fingerprint) == 64
    assert len(engine.all_item_ids) == 5


def test_engine_excludes_seen_item(tmp_path: Path) -> None:
    engine = RecommendationEngine(make_bundle(tmp_path / "bundle.npz"))
    recs, meta = engine.recommend(1, 3)
    assert 10 not in [r.item_id for r in recs]
    assert meta["route"] == "warm_user_hybrid"


def test_cold_user_uses_popularity_route(tmp_path: Path) -> None:
    engine = RecommendationEngine(make_bundle(tmp_path / "bundle.npz"))
    recs, meta = engine.recommend(999, 3)
    assert meta["route"] == "cold_user_popularity"
    assert [r.item_id for r in recs][:2] == [10, 11]


def test_ineligible_treatment_falls_back_to_deployed_policy(tmp_path: Path) -> None:
    engine = RecommendationEngine(make_bundle(tmp_path / "bundle.npz", challenger_eligible=False))
    _, meta = engine.recommend(1, 3, policy="treatment")
    assert meta["effective_policy"] == "m6_deployed"


def test_eligible_treatment_uses_development_policy(tmp_path: Path) -> None:
    engine = RecommendationEngine(make_bundle(tmp_path / "bundle.npz", challenger_eligible=True))
    _, meta = engine.recommend(1, 3, policy="treatment")
    assert meta["effective_policy"] == "m6_development_selected"
