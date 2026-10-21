from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd

from recsys_lab.experiments.assignment import assign_two_arm


def simulate_ab_traffic(
    *,
    experiment: str,
    salt: str,
    control_weight: float,
    seed: int,
    n_units: int,
    exposures_per_unit: int,
    treatment_relative_click_lift: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate deterministic demo traffic for exercising the M7 statistics pipeline.

    This is intentionally synthetic: MovieLens contains ratings, not randomized product exposure
    logs. The synthetic treatment effect must never be interpreted as measured recommender lift.
    """
    if n_units <= 0 or exposures_per_unit <= 0:
        raise ValueError("n_units and exposures_per_unit must be positive")
    rng = np.random.default_rng(seed)
    start = datetime(2026, 8, 1, 12, 0, tzinfo=UTC)

    exposure_rows: list[dict[str, object]] = []
    outcome_rows: list[dict[str, object]] = []
    for idx in range(n_units):
        unit_id = f"demo-user-{idx:06d}"
        assignment = assign_two_arm(
            experiment=experiment,
            unit_id=unit_id,
            salt=salt,
            control_weight=control_weight,
        )
        pre_activity = int(rng.poisson(2.5))
        base_click_p = min(0.18, 0.055 + 0.012 * min(pre_activity, 8))
        click_p = base_click_p * (
            1.0 + treatment_relative_click_lift if assignment.variant == "treatment" else 1.0
        )
        click_p = min(click_p, 0.95)
        model_version = "control_policy_v1" if assignment.variant == "control" else "challenger_v1"

        for impression_idx in range(exposures_per_unit):
            impression_id = f"imp-{idx:06d}-{impression_idx:02d}"
            exposure_ts = start + timedelta(seconds=idx * exposures_per_unit + impression_idx)
            latency_mean = 45.0 if assignment.variant == "control" else 46.0
            latency_ms = max(1.0, float(rng.normal(latency_mean, 7.0)))
            error_p = 0.004 if assignment.variant == "control" else 0.0042
            is_error = bool(rng.random() < error_p)
            clicked = False if is_error else bool(rng.random() < click_p)
            converted = bool(clicked and rng.random() < 0.18)
            value = float(rng.gamma(shape=2.0, scale=3.0)) if converted else 0.0

            exposure_rows.append(
                {
                    "event_id": f"exp-{impression_id}",
                    "impression_id": impression_id,
                    "experiment": experiment,
                    "unit_id": unit_id,
                    "variant": assignment.variant,
                    "timestamp": exposure_ts.isoformat(),
                    "model_version": model_version,
                    "latency_ms": latency_ms,
                    "is_error": int(is_error),
                    "pre_period_activity": pre_activity,
                }
            )
            outcome_rows.append(
                {
                    "event_id": f"out-{impression_id}",
                    "impression_id": impression_id,
                    "experiment": experiment,
                    "unit_id": unit_id,
                    "variant": assignment.variant,
                    "timestamp": (exposure_ts + timedelta(seconds=2)).isoformat(),
                    "clicked": int(clicked),
                    "converted": int(converted),
                    "value": value,
                }
            )

    return pd.DataFrame(exposure_rows), pd.DataFrame(outcome_rows)
