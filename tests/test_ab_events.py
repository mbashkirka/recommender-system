from __future__ import annotations

import pandas as pd
import pytest

from recsys_lab.experiments.events import validate_event_tables


def _tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    exposures = pd.DataFrame(
        [
            {"event_id":"e1","impression_id":"i1","experiment":"x","unit_id":"u1","variant":"control","timestamp":"2026-08-01T00:00:00Z","model_version":"a","latency_ms":10.0,"is_error":0,"pre_period_activity":2},
            {"event_id":"e2","impression_id":"i2","experiment":"x","unit_id":"u1","variant":"control","timestamp":"2026-08-01T00:01:00Z","model_version":"a","latency_ms":11.0,"is_error":0,"pre_period_activity":2},
            {"event_id":"e3","impression_id":"i3","experiment":"x","unit_id":"u2","variant":"treatment","timestamp":"2026-08-01T00:00:30Z","model_version":"b","latency_ms":12.0,"is_error":0,"pre_period_activity":1},
        ]
    )
    outcomes = pd.DataFrame(
        [
            {"event_id":"o1","impression_id":"i1","experiment":"x","unit_id":"u1","variant":"control","timestamp":"2026-08-01T00:00:05Z","clicked":1,"converted":0,"value":0.0},
            {"event_id":"o2","impression_id":"i2","experiment":"x","unit_id":"u1","variant":"control","timestamp":"2026-08-01T00:01:05Z","clicked":0,"converted":0,"value":0.0},
            {"event_id":"o3","impression_id":"i3","experiment":"x","unit_id":"u2","variant":"treatment","timestamp":"2026-08-01T00:00:35Z","clicked":1,"converted":1,"value":3.0},
        ]
    )
    return exposures, outcomes


def test_valid_event_contract_passes() -> None:
    result = validate_event_tables(*_tables())
    assert result.passed
    assert result.unique_units == 2


def test_assignment_must_be_sticky() -> None:
    exposures, outcomes = _tables()
    exposures.loc[1, "variant"] = "treatment"
    outcomes.loc[1, "variant"] = "treatment"
    result = validate_event_tables(exposures, outcomes)
    assert not result.checks["sticky_assignment_per_unit"]


def test_outcome_cannot_precede_exposure() -> None:
    exposures, outcomes = _tables()
    outcomes.loc[0, "timestamp"] = "2026-07-31T23:59:00Z"
    result = validate_event_tables(exposures, outcomes)
    assert not result.checks["outcomes_not_before_exposure"]


def test_missing_event_column_rejected() -> None:
    exposures, outcomes = _tables()
    with pytest.raises(ValueError):
        validate_event_tables(exposures.drop(columns=["model_version"]), outcomes)


def test_hash_assignment_validation_detects_tampering() -> None:
    from recsys_lab.experiments.events import validate_hash_assignments
    from recsys_lab.experiments.simulation import simulate_ab_traffic

    exposures, _ = simulate_ab_traffic(
        experiment="hash-test",
        salt="salt",
        control_weight=0.5,
        seed=1,
        n_units=200,
        exposures_per_unit=2,
        treatment_relative_click_lift=0.1,
    )
    assert validate_hash_assignments(exposures, salt="salt", control_weight=0.5)
    exposures.loc[0, "variant"] = (
        "treatment" if exposures.loc[0, "variant"] == "control" else "control"
    )
    assert not validate_hash_assignments(exposures, salt="salt", control_weight=0.5)
