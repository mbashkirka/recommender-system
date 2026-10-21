from __future__ import annotations

from dataclasses import replace

import pytest

from recsys_lab.config import RerankingConfig, load_settings


def _cfg() -> RerankingConfig:
    return RerankingConfig(
        diversity_weights=(0.0, 0.2), novelty_weights=(0.0, 0.1),
        ndcg_guardrail_ratio=0.97, recall_guardrail_ratio=0.95, tune_fraction=0.5,
        pool_k=50, primary_k=10, max_per_genre=4,
    )


def test_project_reranking_config_has_anchor_and_guardrails() -> None:
    cfg = load_settings().reranking
    assert cfg is not None
    assert 0.0 in cfg.diversity_weights and 0.0 in cfg.novelty_weights
    assert cfg.ndcg_guardrail_ratio == pytest.approx(0.97)
    assert cfg.recall_guardrail_ratio == pytest.approx(0.95)


def test_reranking_config_requires_zero_zero_anchor() -> None:
    cfg = _cfg()
    with pytest.raises(ValueError):
        replace(cfg, diversity_weights=(0.1, 0.2)).validate()


def test_reranking_config_rejects_invalid_window_and_k() -> None:
    cfg = _cfg()
    with pytest.raises(ValueError):
        replace(cfg, tune_fraction=1.0).validate()
    with pytest.raises(ValueError):
        replace(cfg, primary_k=60).validate()


def test_reranking_config_accepts_business_genre_cap_disabled() -> None:
    replace(_cfg(), max_per_genre=0).validate()
