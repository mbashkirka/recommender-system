from __future__ import annotations

import pandas as pd

from recsys_lab.evaluation.offline import build_ground_truth


def test_ground_truth_keeps_only_warm_positive_events() -> None:
    train = pd.DataFrame(
        {
            "user_id": [1, 2],
            "item_id": [10, 20],
            "rating": [5, 4],
            "is_positive": [True, True],
        }
    )
    validation = pd.DataFrame(
        {
            "user_id": [1, 1, 3, 2],
            "item_id": [20, 99, 10, 10],
            "rating": [5, 5, 5, 2],
            "is_positive": [True, True, True, False],
        }
    )
    truth, diagnostics = build_ground_truth(train, validation)
    assert truth == {1: {20}}
    assert diagnostics["positive_events"] == 3
    assert diagnostics["warm_positive_events"] == 1
    assert diagnostics["evaluation_users_with_warm_positive"] == 1
