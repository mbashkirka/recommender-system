from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm


@dataclass(frozen=True)
class DifferenceResult:
    control_mean: float
    treatment_mean: float
    absolute_lift: float
    relative_lift: float | None
    ci_low: float
    ci_high: float
    z_stat: float
    p_value: float
    control_n: int
    treatment_n: int


def two_sample_mean_difference(
    control: np.ndarray,
    treatment: np.ndarray,
    alpha: float = 0.05,
) -> DifferenceResult:
    c = np.asarray(control, dtype=float)
    t = np.asarray(treatment, dtype=float)
    if len(c) < 2 or len(t) < 2:
        raise ValueError("Both experiment arms require at least two observations")
    cm = float(c.mean())
    tm = float(t.mean())
    delta = tm - cm
    cvar = float(c.var(ddof=1))
    tvar = float(t.var(ddof=1))
    se = math.sqrt(cvar / len(c) + tvar / len(t))
    if se == 0.0:
        z = 0.0 if delta == 0.0 else math.copysign(math.inf, delta)
        p = 1.0 if delta == 0.0 else 0.0
    else:
        z = delta / se
        p = float(2.0 * norm.sf(abs(z)))
    zcrit = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    rel = None if cm == 0.0 else delta / cm
    return DifferenceResult(
        control_mean=cm,
        treatment_mean=tm,
        absolute_lift=delta,
        relative_lift=rel,
        ci_low=delta - zcrit * se,
        ci_high=delta + zcrit * se,
        z_stat=float(z),
        p_value=p,
        control_n=len(c),
        treatment_n=len(t),
    )


def sample_ratio_mismatch(
    assignments: pd.Series,
    control_weight: float,
) -> dict[str, float | int | bool]:
    if not 0.0 < control_weight < 1.0:
        raise ValueError("control_weight must be in (0, 1)")
    counts = assignments.value_counts()
    c = int(counts.get("control", 0))
    t = int(counts.get("treatment", 0))
    total = c + t
    if total == 0:
        raise ValueError("No assignments supplied")
    expected = np.array([total * control_weight, total * (1.0 - control_weight)], dtype=float)
    observed = np.array([c, t], dtype=float)
    stat = float(np.sum((observed - expected) ** 2 / expected))
    p_value = float(chi2.sf(stat, df=1))
    return {
        "control_units": c,
        "treatment_units": t,
        "control_share": c / total,
        "chi2": stat,
        "p_value": p_value,
    }


def required_sample_size_per_arm_proportion(
    baseline_rate: float,
    relative_mde: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> int:
    if not 0.0 < baseline_rate < 1.0:
        raise ValueError("baseline_rate must be in (0, 1)")
    if relative_mde <= 0.0:
        raise ValueError("relative_mde must be > 0")
    p0 = baseline_rate
    p1 = min(p0 * (1.0 + relative_mde), 1.0 - 1e-12)
    delta = p1 - p0
    if delta <= 0.0:
        raise ValueError("relative_mde produces no valid difference")
    z_alpha = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    z_power = NormalDist().inv_cdf(power)
    p_bar = (p0 + p1) / 2.0
    numerator = (
        z_alpha * math.sqrt(2.0 * p_bar * (1.0 - p_bar))
        + z_power * math.sqrt(p0 * (1.0 - p0) + p1 * (1.0 - p1))
    ) ** 2
    return math.ceil(numerator / (delta**2))


def detectable_relative_lift(
    baseline_rate: float,
    n_per_arm: int,
    alpha: float = 0.05,
    power: float = 0.80,
) -> float:
    if n_per_arm < 2:
        raise ValueError("n_per_arm must be >= 2")
    lo, hi = 1e-6, min(5.0, (1.0 - baseline_rate) / baseline_rate - 1e-9)
    if hi <= lo:
        return hi
    for _ in range(80):
        mid = (lo + hi) / 2.0
        required = required_sample_size_per_arm_proportion(
            baseline_rate=baseline_rate,
            relative_mde=mid,
            alpha=alpha,
            power=power,
        )
        if required > n_per_arm:
            lo = mid
        else:
            hi = mid
    return hi


def cuped_adjust(
    metric: np.ndarray,
    pre_period: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    y = np.asarray(metric, dtype=float)
    x = np.asarray(pre_period, dtype=float)
    if len(y) != len(x):
        raise ValueError("metric and pre_period must have the same length")
    if len(y) < 2:
        raise ValueError("CUPED needs at least two observations")
    var_x = float(np.var(x, ddof=1))
    if var_x <= 0.0:
        return y.copy(), {"theta": 0.0, "variance_reduction_pct": 0.0}
    covariance = float(np.cov(y, x, ddof=1)[0, 1])
    theta = covariance / var_x
    adjusted = y - theta * (x - x.mean())
    raw_var = float(np.var(y, ddof=1))
    adj_var = float(np.var(adjusted, ddof=1))
    reduction = 0.0 if raw_var <= 0 else 100.0 * (1.0 - adj_var / raw_var)
    return adjusted, {"theta": theta, "variance_reduction_pct": reduction}
