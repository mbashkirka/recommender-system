from dataclasses import replace

import pytest

from recsys_lab.config import RankingConfig, load_settings


def test_project_ranking_config_has_no_regression_anchor() -> None:
    cfg = load_settings().ranking
    assert cfg is not None
    assert cfg.candidate_k == 100
    assert cfg.primary_k == 10
    assert 0.0 in cfg.ltr_blends


def test_ranking_config_requires_holdout_window() -> None:
    cfg = RankingConfig(
        candidate_k=100,
        rank_train_fraction=0.5,
        rank_tune_fraction=0.25,
        negatives_per_positive=20,
        c_values=(0.1, 1.0),
        ltr_blends=(0.0, 0.5),
        primary_k=10,
    )
    cfg.validate()
    with pytest.raises(ValueError):
        replace(cfg, rank_train_fraction=0.8, rank_tune_fraction=0.2).validate()


def test_ranking_config_requires_zero_blend_anchor() -> None:
    cfg = RankingConfig(100, 0.5, 0.25, 20, (1.0,), (0.5, 1.0), 10)
    with pytest.raises(ValueError):
        cfg.validate()
