from __future__ import annotations

from dataclasses import replace

import pytest

from recsys_lab.config import ABTestingConfig, load_settings


def _cfg() -> ABTestingConfig:
    return ABTestingConfig(
        demo_experiment_name="demo",
        demo_units=1000,
        demo_exposures_per_unit=2,
        demo_treatment_relative_click_lift=0.1,
        alpha=0.05,
        power=0.8,
        target_relative_mde=0.1,
        min_practical_relative_lift=0.03,
        srm_alpha=0.001,
        max_latency_increase_ms=5.0,
        max_error_rate_increase=0.005,
    )


def test_project_ab_config_is_loaded() -> None:
    cfg = load_settings().ab_testing
    assert cfg is not None
    assert cfg.demo_units == 30000
    assert cfg.alpha == pytest.approx(0.05)
    assert cfg.target_relative_mde == pytest.approx(0.10)


def test_ab_config_rejects_small_demo() -> None:
    with pytest.raises(ValueError):
        replace(_cfg(), demo_units=99).validate()


def test_ab_config_rejects_bad_probabilities() -> None:
    with pytest.raises(ValueError):
        replace(_cfg(), alpha=1.0).validate()
    with pytest.raises(ValueError):
        replace(_cfg(), power=0.0).validate()
    with pytest.raises(ValueError):
        replace(_cfg(), srm_alpha=0.0).validate()


def test_ab_config_rejects_negative_guardrails() -> None:
    with pytest.raises(ValueError):
        replace(_cfg(), max_latency_increase_ms=-1).validate()
    with pytest.raises(ValueError):
        replace(_cfg(), max_error_rate_increase=-0.1).validate()
