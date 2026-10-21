from __future__ import annotations

import pandas as pd
import pytest

from recsys_lab.evaluation.reranking import _split_rerank_window


def _frame(n: int = 10) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": [1 + i % 3 for i in range(n)],
            "item_id": list(range(100, 100 + n)),
            "rating": [5] * n,
            "is_positive": [True] * n,
            "timestamp": pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC"),
        }
    )


def test_m6_window_is_chronological_and_disjoint() -> None:
    tune, evaluation, manifest = _split_rerank_window(_frame(), 0.5)
    assert len(tune) + len(evaluation) == 10
    assert tune["timestamp"].max() <= evaluation["timestamp"].min()
    assert manifest["chronological_order_check"] is True


def test_m6_window_rejects_invalid_fraction() -> None:
    with pytest.raises(ValueError):
        _split_rerank_window(_frame(), 0.0)
