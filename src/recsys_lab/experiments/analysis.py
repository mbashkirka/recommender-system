from __future__ import annotations

from dataclasses import asdict

import pandas as pd

from recsys_lab.experiments.statistics import (
    cuped_adjust,
    detectable_relative_lift,
    required_sample_size_per_arm_proportion,
    sample_ratio_mismatch,
    two_sample_mean_difference,
)


def build_unit_table(exposures: pd.DataFrame, outcomes: pd.DataFrame) -> pd.DataFrame:
    exp = exposures.copy()
    out = outcomes.copy()
    joined = exp.merge(
        out[["impression_id", "clicked", "converted", "value"]],
        on="impression_id",
        how="left",
        validate="one_to_one",
    )
    return (
        joined.groupby(["unit_id", "variant"], as_index=False)
        .agg(
            exposures=("impression_id", "count"),
            engaged=("clicked", "max"),
            clicks=("clicked", "sum"),
            converted=("converted", "max"),
            conversions=("converted", "sum"),
            value=("value", "sum"),
            mean_latency_ms=("latency_ms", "mean"),
            error_rate=("is_error", "mean"),
            pre_period_activity=("pre_period_activity", "first"),
        )
        .sort_values("unit_id")
        .reset_index(drop=True)
    )


def _as_dict(result) -> dict[str, object]:
    return asdict(result)


def analyze_ab_test(
    exposures: pd.DataFrame,
    outcomes: pd.DataFrame,
    *,
    control_weight: float,
    alpha: float,
    power: float,
    target_relative_mde: float,
    min_practical_relative_lift: float,
    max_latency_increase_ms: float,
    max_error_rate_increase: float,
    srm_alpha: float,
) -> dict[str, object]:
    unit = build_unit_table(exposures, outcomes)
    control = unit[unit["variant"] == "control"].copy()
    treatment = unit[unit["variant"] == "treatment"].copy()

    srm = sample_ratio_mismatch(unit["variant"], control_weight=control_weight)
    srm["alpha"] = srm_alpha
    srm["passed"] = bool(srm["p_value"] >= srm_alpha)

    primary = two_sample_mean_difference(
        control["engaged"].to_numpy(), treatment["engaged"].to_numpy(), alpha=alpha
    )
    conversion = two_sample_mean_difference(
        control["converted"].to_numpy(), treatment["converted"].to_numpy(), alpha=alpha
    )
    latency = two_sample_mean_difference(
        control["mean_latency_ms"].to_numpy(),
        treatment["mean_latency_ms"].to_numpy(),
        alpha=alpha,
    )
    errors = two_sample_mean_difference(
        control["error_rate"].to_numpy(), treatment["error_rate"].to_numpy(), alpha=alpha
    )

    raw_clicks = unit["clicks"].to_numpy(dtype=float)
    pre = unit["pre_period_activity"].to_numpy(dtype=float)
    adjusted, cuped_info = cuped_adjust(raw_clicks, pre)
    unit = unit.assign(cuped_clicks=adjusted)
    cuped = two_sample_mean_difference(
        unit.loc[unit["variant"] == "control", "cuped_clicks"].to_numpy(),
        unit.loc[unit["variant"] == "treatment", "cuped_clicks"].to_numpy(),
        alpha=alpha,
    )

    required = required_sample_size_per_arm_proportion(
        baseline_rate=primary.control_mean,
        relative_mde=target_relative_mde,
        alpha=alpha,
        power=power,
    )
    achieved_per_arm = min(primary.control_n, primary.treatment_n)
    achieved_mde = detectable_relative_lift(
        baseline_rate=primary.control_mean,
        n_per_arm=achieved_per_arm,
        alpha=alpha,
        power=power,
    )

    latency_ok = latency.absolute_lift <= max_latency_increase_ms
    errors_ok = errors.absolute_lift <= max_error_rate_increase
    guardrails_pass = bool(latency_ok and errors_ok)
    practical = bool(
        primary.relative_lift is not None
        and primary.relative_lift >= min_practical_relative_lift
    )
    significant_positive = bool(primary.p_value < alpha and primary.ci_low > 0.0)
    powered_for_target = achieved_per_arm >= required

    blockers: list[str] = []
    if not srm["passed"]:
        blockers.append("sample_ratio_mismatch")
    if not latency_ok:
        blockers.append("latency_guardrail")
    if not errors_ok:
        blockers.append("error_rate_guardrail")

    if blockers:
        decision = "no_ship"
    elif significant_positive and practical and powered_for_target:
        decision = "ship"
    elif primary.ci_high < 0.0:
        decision = "no_ship"
    else:
        decision = "inconclusive"

    return {
        "unit_of_randomization": "user",
        "inference_level": "user",
        "primary_metric": "engaged_user_rate",
        "descriptive_ctr": float(outcomes["clicked"].mean()),
        "srm": srm,
        "primary": _as_dict(primary),
        "secondary_conversion": _as_dict(conversion),
        "cuped_clicks_per_user": {
            **_as_dict(cuped),
            **cuped_info,
        },
        "power": {
            "alpha": alpha,
            "target_power": power,
            "target_relative_mde": target_relative_mde,
            "required_sample_size_per_arm": required,
            "achieved_sample_size_per_arm": achieved_per_arm,
            "achieved_relative_mde": achieved_mde,
            "powered_for_target_mde": bool(powered_for_target),
        },
        "guardrails": {
            "latency_ms": {
                **_as_dict(latency),
                "max_allowed_increase_ms": max_latency_increase_ms,
                "passed": bool(latency_ok),
            },
            "error_rate": {
                **_as_dict(errors),
                "max_allowed_absolute_increase": max_error_rate_increase,
                "passed": bool(errors_ok),
            },
            "passed": guardrails_pass,
        },
        "decision": {
            "status": decision,
            "significant_positive_primary": significant_positive,
            "practical_lift_passed": practical,
            "powered_for_target_mde": bool(powered_for_target),
            "guardrails_passed": guardrails_pass,
            "srm_passed": bool(srm["passed"]),
            "blockers": blockers,
        },
        "unit_table": unit,
    }
