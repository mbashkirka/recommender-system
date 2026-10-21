from __future__ import annotations

import json

import matplotlib.pyplot as plt
import pandas as pd

from recsys_lab.config import Settings
from recsys_lab.experiments.analysis import analyze_ab_test
from recsys_lab.experiments.events import (
    validate_event_tables,
    validate_hash_assignments,
    write_event_tables,
)
from recsys_lab.experiments.simulation import simulate_ab_traffic
from recsys_lab.experiments.statistics import required_sample_size_per_arm_proportion


def _load_m6_serving_state(settings: Settings) -> dict[str, object]:
    path = settings.paths.artifacts / "m6" / "m6_manifest.json"
    if not path.exists():
        return {
            "m6_manifest_found": False,
            "control_policy": "latest_deployed_recommender",
            "offline_challenger_eligible": None,
            "note": "M6 artifact was not present; M7 demo statistics can still run.",
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    gate = payload.get("deployment_gate", {})
    deployed = payload.get("deployed", {})
    return {
        "m6_manifest_found": True,
        "control_policy": "m6_deployed",
        "control_parameters": deployed,
        "offline_challenger_eligible": bool(gate.get("passed", False)),
        "offline_challenger_parameters": payload.get("development_selected", {}),
        "offline_gate_action": gate.get("action"),
    }


def run_m7_ab_testing(settings: Settings) -> dict[str, object]:
    if settings.ab_testing is None:
        raise ValueError("[ab_testing] config is required for M7")
    cfg = settings.ab_testing
    report_dir = settings.paths.reports / "m7"
    artifact_dir = settings.paths.artifacts / "m7"
    report_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    serving_state = _load_m6_serving_state(settings)
    contract = {
        "experiment": cfg.demo_experiment_name,
        "assignment_unit": "user",
        "assignment_method": "sha256 sticky bucketing",
        "control_weight": settings.experiment.control_weight,
        "control_serving_state": serving_state,
        "demo_treatment": "synthetic_challenger_v1",
        "demo_only": True,
        "warning": (
            "Synthetic M7 lift validates experiment infrastructure only; it is not evidence that "
            "the recommender treatment improves real users."
        ),
    }
    (artifact_dir / "experiment_contract.json").write_text(
        json.dumps(contract, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    exposures, outcomes = simulate_ab_traffic(
        experiment=cfg.demo_experiment_name,
        salt=settings.experiment.default_salt,
        control_weight=settings.experiment.control_weight,
        seed=settings.seed,
        n_units=cfg.demo_units,
        exposures_per_unit=cfg.demo_exposures_per_unit,
        treatment_relative_click_lift=cfg.demo_treatment_relative_click_lift,
    )
    write_event_tables(exposures, outcomes, report_dir)
    validation = validate_event_tables(exposures, outcomes)
    hash_assignment_ok = validate_hash_assignments(
        exposures,
        salt=settings.experiment.default_salt,
        control_weight=settings.experiment.control_weight,
    )
    if not validation.passed or not hash_assignment_ok:
        raise RuntimeError(
            f"M7 event contract failed: checks={validation.checks}, "
            f"hash_assignment_ok={hash_assignment_ok}"
        )

    analysis = analyze_ab_test(
        exposures,
        outcomes,
        control_weight=settings.experiment.control_weight,
        alpha=cfg.alpha,
        power=cfg.power,
        target_relative_mde=cfg.target_relative_mde,
        min_practical_relative_lift=cfg.min_practical_relative_lift,
        max_latency_increase_ms=cfg.max_latency_increase_ms,
        max_error_rate_increase=cfg.max_error_rate_increase,
        srm_alpha=cfg.srm_alpha,
    )
    unit_table: pd.DataFrame = analysis.pop("unit_table")
    unit_table.to_csv(report_dir / "unit_level_metrics.csv.gz", index=False, compression="gzip")

    compact = {
        "experiment": cfg.demo_experiment_name,
        "demo_only": True,
        "test_split_accessed": False,
        "event_validation": {
            "checks": validation.checks,
            "exposures": validation.exposures,
            "outcomes": validation.outcomes,
            "unique_units": validation.unique_units,
            "hash_assignment_matches_log": hash_assignment_ok,
        },
        **analysis,
    }
    (report_dir / "ab_analysis.json").write_text(
        json.dumps(compact, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    rows = []
    for label, result in [
        ("primary_engaged_user_rate", compact["primary"]),
        ("secondary_conversion", compact["secondary_conversion"]),
        ("cuped_clicks_per_user", compact["cuped_clicks_per_user"]),
    ]:
        rows.append(
            {
                "metric": label,
                "control_mean": result["control_mean"],
                "treatment_mean": result["treatment_mean"],
                "absolute_lift": result["absolute_lift"],
                "relative_lift": result["relative_lift"],
                "ci_low": result["ci_low"],
                "ci_high": result["ci_high"],
                "p_value": result["p_value"],
            }
        )
    pd.DataFrame(rows).to_csv(report_dir / "metric_summary.csv", index=False)

    baseline = float(compact["primary"]["control_mean"])
    power_rows = []
    for rel_mde in (0.05, 0.075, 0.10, 0.15, 0.20):
        required = required_sample_size_per_arm_proportion(
            baseline_rate=baseline, relative_mde=rel_mde, alpha=cfg.alpha, power=cfg.power
        )
        power_rows.append({"relative_mde": rel_mde, "required_users_per_arm": required})
    power_df = pd.DataFrame(power_rows)
    power_df.to_csv(report_dir / "power_curve.csv", index=False)
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.plot(power_df["relative_mde"] * 100.0, power_df["required_users_per_arm"], marker="o")
    ax.set_xlabel("Relative MDE (%)")
    ax.set_ylabel("Required users per arm")
    ax.set_title("M7 power planning")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(report_dir / "power_curve.png", dpi=150)
    plt.close(fig)

    manifest = {
        "milestone": "M7",
        "test_split_accessed": False,
        "experiment": cfg.demo_experiment_name,
        "demo_only": True,
        "assignment": {
            "unit": "user",
            "sticky": True,
            "control_weight": settings.experiment.control_weight,
        },
        "serving_state": serving_state,
        "event_validation_passed": validation.passed,
        "analysis": compact,
    }
    (artifact_dir / "m7_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest
