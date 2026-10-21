from __future__ import annotations

import pytest

from recsys_lab.config import SplitConfig


def test_split_config_accepts_valid_fractions() -> None:
    cfg = SplitConfig(0.8, 0.1, 0.1, "warm_user_warm_item")
    cfg.validate()


def test_split_config_rejects_invalid_sum() -> None:
    cfg = SplitConfig(0.8, 0.1, 0.2, "warm_user_warm_item")
    with pytest.raises(ValueError):
        cfg.validate()
