import pytest

from recsys_lab.config import HybridConfig, load_settings


def test_project_hybrid_config_loads_with_zero_anchor() -> None:
    settings = load_settings()
    assert settings.hybrid is not None
    assert settings.hybrid.metadata_blends[0] == 0.0
    assert settings.hybrid.cold_start_k == 100


def test_hybrid_config_requires_zero_no_regression_anchor() -> None:
    cfg = HybridConfig(
        metadata_blends=(0.1, 0.2),
        ndcg_guardrail_ratio=0.97,
        sparse_positive_threshold=5,
        sparse_metadata_multiplier=1.5,
        cold_item_penalty=0.25,
        cold_start_k=100,
        cold_item_quota=20,
    )
    with pytest.raises(ValueError):
        cfg.validate()
