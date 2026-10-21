from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from recsys_lab.models.collaborative import (
    PureSVDRecommender,
    build_positive_matrix,
    fit_puresvd,
)


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


def test_positive_matrix_is_binary_and_includes_train_catalog() -> None:
    matrix, users, items = build_positive_matrix(_train())
    assert matrix.shape == (6, 6)
    assert set(users) == {1, 2, 3, 4, 5, 6}
    assert set(items) == {10, 11, 12, 13, 14, 15}
    assert set(np.unique(matrix.data)) == {1.0}
    assert matrix[:, list(items).index(15)].nnz == 0


def test_puresvd_excludes_seen_items() -> None:
    decomp = fit_puresvd(_train(), max_rank=2, seed=7)
    model = PureSVDRecommender(decomp, rank=2, popularity_blend=0.1)
    recs = model.recommend(1, 10)
    assert set(recs).isdisjoint({10, 11, 15})


def test_puresvd_is_deterministic_for_same_fitted_artifact() -> None:
    decomp = fit_puresvd(_train(), max_rank=2, seed=7)
    model = PureSVDRecommender(decomp, rank=2, popularity_blend=0.2)
    assert model.recommend(3, 3) == model.recommend(3, 3)


def test_puresvd_unknown_user_falls_back_without_seen_items() -> None:
    decomp = fit_puresvd(_train(), max_rank=2, seed=7)
    model = PureSVDRecommender(decomp, rank=2, popularity_blend=0.0)
    recs = model.recommend(999, 3)
    assert len(recs) == 3
    assert len(set(recs)) == 3


def test_puresvd_artifact_can_be_saved(tmp_path: Path) -> None:
    decomp = fit_puresvd(_train(), max_rank=2, seed=7)
    model = PureSVDRecommender(decomp, rank=1, popularity_blend=0.1)
    path = tmp_path / "model.npz"
    model.save(path)
    saved = np.load(path)
    assert int(saved["rank"][0]) == 1
    assert saved["item_components"].shape[1] == 1


def test_invalid_rank_is_rejected() -> None:
    decomp = fit_puresvd(_train(), max_rank=2, seed=7)
    with pytest.raises(ValueError):
        PureSVDRecommender(decomp, rank=3)
