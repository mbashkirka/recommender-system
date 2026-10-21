from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from recsys_lab.experiments.statistics import (
    cuped_adjust,
    detectable_relative_lift,
    required_sample_size_per_arm_proportion,
    sample_ratio_mismatch,
    two_sample_mean_difference,
)


def test_mean_difference_detects_positive_lift() -> None:
    control = np.array([0, 0, 0, 1, 0, 1, 0, 0] * 100, dtype=float)
    treatment = np.array([0, 1, 0, 1, 1, 1, 0, 0] * 100, dtype=float)
    result = two_sample_mean_difference(control, treatment)
    assert result.absolute_lift > 0
    assert result.treatment_mean > result.control_mean


def test_srm_balanced_assignment_passes_high_p_value() -> None:
    assignments = pd.Series(["control"] * 500 + ["treatment"] * 500)
    result = sample_ratio_mismatch(assignments, 0.5)
    assert result["p_value"] == pytest.approx(1.0)


def test_srm_detects_extreme_mismatch() -> None:
    assignments = pd.Series(["control"] * 900 + ["treatment"] * 100)
    result = sample_ratio_mismatch(assignments, 0.5)
    assert result["p_value"] < 1e-10


def test_power_sample_size_decreases_for_larger_effect() -> None:
    small = required_sample_size_per_arm_proportion(0.15, 0.05)
    large = required_sample_size_per_arm_proportion(0.15, 0.15)
    assert small > large


def test_detectable_lift_decreases_with_more_users() -> None:
    small = detectable_relative_lift(0.15, 2000)
    large = detectable_relative_lift(0.15, 20000)
    assert large < small


def test_cuped_reduces_variance_when_covariate_is_predictive() -> None:
    rng = np.random.default_rng(7)
    x = rng.normal(size=5000)
    y = 2.0 * x + rng.normal(scale=0.5, size=5000)
    adjusted, info = cuped_adjust(y, x)
    assert adjusted.var(ddof=1) < y.var(ddof=1)
    assert info["variance_reduction_pct"] > 50.0
