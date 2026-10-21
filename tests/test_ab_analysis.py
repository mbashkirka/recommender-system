from __future__ import annotations

from recsys_lab.experiments.analysis import analyze_ab_test, build_unit_table
from recsys_lab.experiments.events import validate_event_tables
from recsys_lab.experiments.simulation import simulate_ab_traffic


def _demo(lift: float = 0.20):
    return simulate_ab_traffic(
        experiment="test_demo",
        salt="salt",
        control_weight=0.5,
        seed=123,
        n_units=8000,
        exposures_per_unit=2,
        treatment_relative_click_lift=lift,
    )


def test_simulation_preserves_sticky_assignment_and_contract() -> None:
    exposures, outcomes = _demo()
    result = validate_event_tables(exposures, outcomes)
    assert result.passed
    assert exposures.groupby("unit_id")["variant"].nunique().max() == 1


def test_unit_table_aggregates_at_randomization_level() -> None:
    exposures, outcomes = _demo()
    unit = build_unit_table(exposures, outcomes)
    assert len(unit) == exposures["unit_id"].nunique()
    assert set(unit["variant"]) == {"control", "treatment"}


def test_analysis_returns_primary_power_guardrails_and_decision() -> None:
    exposures, outcomes = _demo()
    result = analyze_ab_test(
        exposures,
        outcomes,
        control_weight=0.5,
        alpha=0.05,
        power=0.80,
        target_relative_mde=0.20,
        min_practical_relative_lift=0.01,
        max_latency_increase_ms=5.0,
        max_error_rate_increase=0.01,
        srm_alpha=0.001,
    )
    assert result["primary"]["treatment_mean"] > result["primary"]["control_mean"]
    assert "powered_for_target_mde" in result["power"]
    assert result["guardrails"]["passed"]
    assert result["srm"]["passed"]
    assert result["decision"]["status"] in {"ship", "inconclusive"}


def test_analysis_can_block_on_latency_guardrail() -> None:
    exposures, outcomes = _demo()
    exposures.loc[exposures["variant"] == "treatment", "latency_ms"] += 20.0
    result = analyze_ab_test(
        exposures,
        outcomes,
        control_weight=0.5,
        alpha=0.05,
        power=0.80,
        target_relative_mde=0.20,
        min_practical_relative_lift=0.0,
        max_latency_increase_ms=5.0,
        max_error_rate_increase=0.01,
        srm_alpha=0.001,
    )
    assert result["decision"]["status"] == "no_ship"
    assert "latency_guardrail" in result["decision"]["blockers"]
