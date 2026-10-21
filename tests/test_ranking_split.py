import pandas as pd
import pytest

from recsys_lab.evaluation.ranking import split_ranker_validation


def _validation() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": list(range(1, 13)),
            "item_id": list(range(101, 113)),
            "rating": [5] * 12,
            "is_positive": [True] * 12,
            "timestamp": pd.date_range("2026-01-01", periods=12, freq="h", tz="UTC"),
        }
    )


def test_ranker_windows_are_chronological_and_exhaustive() -> None:
    a, b, c, manifest = split_ranker_validation(_validation(), 0.5, 0.25)
    assert len(a) + len(b) + len(c) == 12
    assert a["timestamp"].max() <= b["timestamp"].min() <= c["timestamp"].min()
    assert manifest["chronological_order_check"] is True


def test_invalid_ranker_window_fraction_rejected() -> None:
    with pytest.raises(ValueError):
        split_ranker_validation(_validation(), 0.8, 0.2)
