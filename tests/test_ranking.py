from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from recsys_lab.models.collaborative import fit_puresvd
from recsys_lab.models.hybrid import HybridSVDMetadataRecommender
from recsys_lab.models.ranking import FEATURE_NAMES, CandidateFeatureBuilder, PairwiseLinearRanker


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
        [(10, "A", "Action"), (11, "B", "Action|Comedy"), (12, "C", "Drama"),
         (13, "D", "Drama|Romance"), (14, "E", "Comedy"), (15, "F", "Horror"),
         (99, "Cold", "Action")],
        columns=["item_id", "title", "genres"],
    )


def _hybrid() -> HybridSVDMetadataRecommender:
    decomp = fit_puresvd(_train(), max_rank=2, seed=7)
    return HybridSVDMetadataRecommender(decomp, _movies(), rank=2, popularity_blend=0.1, metadata_blend=0.2)


def _rank_frame() -> pd.DataFrame:
    rows = []
    for user in [1, 2, 3]:
        for rank, item in enumerate([20, 21, 22, 23], 1):
            row = {"user_id": user, "item_id": item, "base_rank": rank, "label": int(item == 22)}
            for i, name in enumerate(FEATURE_NAMES):
                row[name] = float((1 if item == 22 else -1) * (i + 1) + rank * 0.01)
            rows.append(row)
    return pd.DataFrame(rows)


def test_feature_builder_returns_expected_columns_and_candidate_order() -> None:
    model = _hybrid()
    recs = model.recommend(1, 4)
    frame = CandidateFeatureBuilder(model).build(1, recs)
    assert list(frame["item_id"]) == recs
    assert tuple(name for name in FEATURE_NAMES if name in frame.columns) == FEATURE_NAMES
    assert list(frame["base_rank"]) == list(range(1, len(frame) + 1))


def test_feature_builder_uses_only_finite_numeric_features() -> None:
    frame = CandidateFeatureBuilder(_hybrid()).build(999, _hybrid().recommend(999, 4))
    assert np.isfinite(frame[list(FEATURE_NAMES)].to_numpy(dtype=float)).all()


def test_pairwise_training_builds_symmetric_classes() -> None:
    x, y, diag = PairwiseLinearRanker.build_pairwise_training_data(_rank_frame(), 2)
    assert x.shape[1] == len(FEATURE_NAMES)
    assert set(y.tolist()) == {0, 1}
    assert int((y == 0).sum()) == int((y == 1).sum())
    assert diag["users_with_pairs"] == 3


def test_pairwise_ranker_promotes_relevant_pattern() -> None:
    frame = _rank_frame()
    ranker = PairwiseLinearRanker(c=1.0, seed=1)
    ranker.fit(frame, negatives_per_positive=3)
    user = frame.loc[frame["user_id"].eq(1)]
    reranked = ranker.rerank(user, blend=1.0)
    assert reranked[0] == 22


def test_zero_blend_is_exact_no_regression_anchor() -> None:
    frame = _rank_frame()
    ranker = PairwiseLinearRanker(c=1.0, seed=1)
    ranker.fit(frame, negatives_per_positive=2)
    user = frame.loc[frame["user_id"].eq(1)]
    assert ranker.rerank(user, blend=0.0) == [20, 21, 22, 23]


def test_ranker_save_contains_serving_state(tmp_path) -> None:
    ranker = PairwiseLinearRanker(c=0.5, seed=1)
    ranker.fit(_rank_frame(), negatives_per_positive=2)
    path = tmp_path / "ranker.npz"
    ranker.save(path, blend=0.75)
    saved = np.load(path)
    assert saved["feature_names"].tolist() == list(FEATURE_NAMES)
    assert float(saved["blend"][0]) == pytest.approx(0.75)


def test_invalid_pairwise_settings_rejected() -> None:
    with pytest.raises(ValueError):
        PairwiseLinearRanker(c=0.0)
    with pytest.raises(ValueError):
        PairwiseLinearRanker.build_pairwise_training_data(_rank_frame(), 0)
