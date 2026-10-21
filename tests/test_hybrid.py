from __future__ import annotations

import numpy as np
import pandas as pd

from recsys_lab.models.collaborative import PureSVDRecommender, fit_puresvd
from recsys_lab.models.hybrid import HybridSVDMetadataRecommender, build_metadata_index


def _train() -> pd.DataFrame:
    rows = [
        (1, 10, 5, True), (1, 11, 5, True), (1, 15, 2, False),
        (2, 10, 5, True), (2, 11, 4, True), (2, 12, 5, True),
        (3, 12, 5, True), (3, 13, 5, True),
        (4, 12, 4, True), (4, 13, 5, True), (4, 14, 5, True),
        (5, 10, 4, True), (5, 14, 5, True),
        (6, 11, 5, True), (6, 14, 4, True),
    ]
    return pd.DataFrame(rows, columns=["user_id", "item_id", "rating", "is_positive"])


def _movies() -> pd.DataFrame:
    return pd.DataFrame(
        [
            (10, "A", "Action"),
            (11, "B", "Action|Comedy"),
            (12, "C", "Drama"),
            (13, "D", "Drama|Romance"),
            (14, "E", "Comedy"),
            (15, "F", "Horror"),
            (99, "Cold action", "Action"),
        ],
        columns=["item_id", "title", "genres"],
    )


def _model(metadata_blend: float = 0.2) -> HybridSVDMetadataRecommender:
    decomp = fit_puresvd(_train(), max_rank=2, seed=7)
    return HybridSVDMetadataRecommender(
        decomposition=decomp,
        movies=_movies(),
        rank=2,
        popularity_blend=0.1,
        metadata_blend=metadata_blend,
        sparse_positive_threshold=1,
        sparse_metadata_multiplier=1.5,
        cold_item_penalty=0.1,
    )


def test_metadata_index_contains_train_unseen_item() -> None:
    index = build_metadata_index(_movies())
    assert 99 in index.item_index
    assert index.features.shape[0] == 7
    assert "Action" in index.feature_names


def test_zero_metadata_blend_matches_puresvd_on_warm_catalog() -> None:
    train = _train()
    decomp = fit_puresvd(train, max_rank=2, seed=7)
    pure = PureSVDRecommender(decomp, rank=2, popularity_blend=0.1)
    hybrid = HybridSVDMetadataRecommender(
        decomposition=decomp,
        movies=_movies(),
        rank=2,
        popularity_blend=0.1,
        metadata_blend=0.0,
        sparse_positive_threshold=1,
    )
    assert hybrid.recommend(1, 4) == pure.recommend(1, 4)


def test_hybrid_excludes_seen_items() -> None:
    recs = _model().recommend(1, 10)
    assert set(recs).isdisjoint({10, 11, 15})


def test_metadata_channel_makes_cold_item_score_finite() -> None:
    model = _model(metadata_blend=0.5)
    scores = model.score_all(1)
    cold_idx = model.metadata.item_index[99]
    assert np.isfinite(scores[cold_idx])


def test_zero_metadata_blend_keeps_cold_item_unreachable_for_warm_user() -> None:
    model = _model(metadata_blend=0.0)
    scores = model.score_all(1)
    assert not np.isfinite(scores[model.metadata.item_index[99]])


def test_unknown_user_routes_to_popularity() -> None:
    model = _model()
    assert model.route_for_user(999) == "cold_user_popularity"
    recs = model.recommend(999, 3)
    assert len(recs) == 3
    assert len(set(recs)) == 3


def test_sparse_user_route_is_explicit() -> None:
    train = _train()
    decomp = fit_puresvd(train, max_rank=2, seed=7)
    model = HybridSVDMetadataRecommender(
        decomposition=decomp,
        movies=_movies(),
        rank=2,
        popularity_blend=0.1,
        metadata_blend=0.2,
        sparse_positive_threshold=2,
    )
    assert model.route_for_user(3) == "sparse_user_hybrid"


def test_dense_warm_user_route_is_explicit() -> None:
    train = _train()
    decomp = fit_puresvd(train, max_rank=2, seed=7)
    model = HybridSVDMetadataRecommender(
        decomposition=decomp,
        movies=_movies(),
        rank=2,
        popularity_blend=0.1,
        metadata_blend=0.2,
        sparse_positive_threshold=1,
    )
    assert model.route_for_user(2) == "warm_user_hybrid"


def test_cold_item_quota_injects_train_unseen_metadata_items() -> None:
    model = _model(metadata_blend=0.0)
    recs = model.recommend_with_cold_item_quota(1, k=6, cold_item_quota=2)
    assert 99 in recs
    assert len(recs) == len(set(recs))
    assert 1 <= len(recs) <= 6


def test_hybrid_artifact_save_contains_serving_state(tmp_path) -> None:
    model = _model(metadata_blend=0.2)
    path = tmp_path / "hybrid.npz"
    model.save(path)
    saved = np.load(path)
    assert int(saved["rank"][0]) == 2
    assert "metadata_features" in saved.files
    assert len(saved["seen_user_ids"]) == len(saved["seen_item_ids"])
