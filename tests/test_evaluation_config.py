import pytest

from recsys_lab.config import EvaluationConfig


def test_evaluation_config_accepts_primary_k_in_ks() -> None:
    cfg = EvaluationConfig((5, 10, 20), 10)
    cfg.validate()


def test_evaluation_config_rejects_missing_primary_k() -> None:
    cfg = EvaluationConfig((5, 20), 10)
    with pytest.raises(ValueError):
        cfg.validate()
