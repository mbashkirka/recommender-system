from __future__ import annotations

import pandas as pd

from recsys_lab.evaluation.split import assign_periods, chronological_cutoffs


def _ratings(n: int = 10) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": [1] * n,
            "item_id": range(1, n + 1),
            "rating": [5] * n,
            "timestamp": pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC"),
            "is_positive": [True] * n,
        }
    )


def test_chronological_cutoffs_respect_order() -> None:
    ratings = _ratings(10)
    train_end, validation_end = chronological_cutoffs(ratings, 0.8, 0.1)
    periods = assign_periods(ratings, train_end, validation_end)
    assert periods.value_counts().to_dict() == {"train": 8, "validation": 1, "test": 1}


def test_boundary_timestamp_is_not_split_across_periods() -> None:
    ratings = _ratings(10)
    ratings.loc[7, "timestamp"] = ratings.loc[6, "timestamp"]
    train_end, validation_end = chronological_cutoffs(ratings, 0.8, 0.1)
    periods = assign_periods(ratings, train_end, validation_end)
    grouped = ratings.assign(period=periods).groupby("timestamp")["period"].nunique()
    assert int(grouped.max()) == 1


def test_periods_are_globally_ordered() -> None:
    ratings = _ratings(20)
    train_end, validation_end = chronological_cutoffs(ratings, 0.8, 0.1)
    periods = assign_periods(ratings, train_end, validation_end)
    train = ratings.loc[periods.eq("train")]
    validation = ratings.loc[periods.eq("validation")]
    test = ratings.loc[periods.eq("test")]
    assert train["timestamp"].max() < validation["timestamp"].min()
    assert validation["timestamp"].max() < test["timestamp"].min()
