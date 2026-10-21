from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from recsys_lab.experiments.assignment import assign_two_arm

EXPOSURE_COLUMNS = [
    "event_id",
    "impression_id",
    "experiment",
    "unit_id",
    "variant",
    "timestamp",
    "model_version",
    "latency_ms",
    "is_error",
    "pre_period_activity",
]

OUTCOME_COLUMNS = [
    "event_id",
    "impression_id",
    "experiment",
    "unit_id",
    "variant",
    "timestamp",
    "clicked",
    "converted",
    "value",
]


@dataclass(frozen=True)
class EventValidationResult:
    checks: dict[str, bool]
    exposures: int
    outcomes: int
    unique_units: int

    @property
    def passed(self) -> bool:
        return all(self.checks.values())


def _require_columns(df: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = sorted(set(columns) - set(df.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def validate_event_tables(exposures: pd.DataFrame, outcomes: pd.DataFrame) -> EventValidationResult:
    """Validate the M7 assignment/exposure/outcome contract.

    The assignment unit is the user. Multiple impressions are allowed, but every impression for a
    user must carry the same variant. Outcomes must join to exactly one exposure and cannot occur
    before it.
    """
    _require_columns(exposures, EXPOSURE_COLUMNS, "exposures")
    _require_columns(outcomes, OUTCOME_COLUMNS, "outcomes")

    exp = exposures.copy()
    out = outcomes.copy()
    exp["timestamp"] = pd.to_datetime(exp["timestamp"], utc=True)
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)

    exp_key = exp[["impression_id", "experiment", "unit_id", "variant", "timestamp"]].rename(
        columns={"timestamp": "exposure_timestamp"}
    )
    joined = out.merge(
        exp_key,
        on=["impression_id", "experiment", "unit_id", "variant"],
        how="left",
        validate="many_to_one",
    )

    variant_counts = exp.groupby(["experiment", "unit_id"], sort=False)["variant"].nunique()
    model_counts = exp.groupby(["experiment", "variant"], sort=False)["model_version"].nunique()

    checks = {
        "exposure_event_ids_unique": not exp["event_id"].duplicated().any(),
        "outcome_event_ids_unique": not out["event_id"].duplicated().any(),
        "outcome_impression_ids_unique": not out["impression_id"].duplicated().any(),
        "impression_ids_unique_in_exposures": not exp["impression_id"].duplicated().any(),
        "sticky_assignment_per_unit": bool((variant_counts == 1).all()),
        "single_model_version_per_variant": bool((model_counts == 1).all()),
        "outcomes_link_to_exposure": bool(joined["exposure_timestamp"].notna().all()),
        "outcomes_not_before_exposure": bool(
            (joined["timestamp"] >= joined["exposure_timestamp"]).all()
        ),
        "binary_clicked": bool(out["clicked"].isin([0, 1, False, True]).all()),
        "binary_converted": bool(out["converted"].isin([0, 1, False, True]).all()),
        "non_negative_latency": bool((exp["latency_ms"] >= 0).all()),
        "non_negative_value": bool((out["value"] >= 0).all()),
    }
    return EventValidationResult(
        checks=checks,
        exposures=len(exp),
        outcomes=len(out),
        unique_units=exp["unit_id"].nunique(),
    )


def write_event_tables(
    exposures: pd.DataFrame,
    outcomes: pd.DataFrame,
    directory: Path,
) -> tuple[Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    exposure_path = directory / "exposures.csv.gz"
    outcome_path = directory / "outcomes.csv.gz"
    exposures.to_csv(exposure_path, index=False, compression="gzip")
    outcomes.to_csv(outcome_path, index=False, compression="gzip")
    return exposure_path, outcome_path


def validate_hash_assignments(
    exposures: pd.DataFrame,
    *,
    salt: str,
    control_weight: float,
) -> bool:
    """Verify logged variants against the deterministic assignment function."""
    _require_columns(exposures, ["experiment", "unit_id", "variant"], "exposures")
    units = exposures[["experiment", "unit_id", "variant"]].drop_duplicates()
    for row in units.itertuples(index=False):
        expected = assign_two_arm(
            experiment=row.experiment,
            unit_id=row.unit_id,
            salt=salt,
            control_weight=control_weight,
        ).variant
        if expected != row.variant:
            return False
    return True
