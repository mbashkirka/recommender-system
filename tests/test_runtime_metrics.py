from __future__ import annotations

from recsys_lab.serving.metrics import RuntimeMetrics


def test_runtime_metrics_collect_recommendation_latency_and_routes() -> None:
    metrics = RuntimeMetrics(latency_window=10)
    metrics.observe_recommend(1.0, "warm_user_hybrid", "control", "control")
    metrics.observe_recommend(3.0, "cold_user_popularity", "treatment", "control")
    snap = metrics.snapshot()
    assert snap["recommend_requests_total"] == 2
    assert snap["recommend_errors_total"] == 0
    assert snap["latency_ms"]["mean"] == 2.0
    assert snap["routes"]["warm_user_hybrid"] == 1
    assert snap["variant_pairs"]["treatment->control"] == 1


def test_runtime_metrics_tracks_errors_and_outcomes() -> None:
    metrics = RuntimeMetrics(latency_window=3)
    metrics.observe_recommend(2.0, "unknown", "unknown", "unknown", error=True)
    metrics.observe_outcome("accepted")
    metrics.observe_outcome("rejected", "unknown_impression")
    snap = metrics.snapshot()
    assert snap["recommend_error_rate"] == 1.0
    assert snap["outcomes"]["accepted:none"] == 1
    assert snap["outcomes"]["rejected:unknown_impression"] == 1


def test_prometheus_surface_contains_core_metrics() -> None:
    metrics = RuntimeMetrics(latency_window=5)
    metrics.observe_recommend(5.0, "warm_user_hybrid", "control", "control")
    text = metrics.prometheus_text(integrity_ok=True, challenger_eligible=False)
    assert "recsys_artifact_integrity_ok 1" in text
    assert "recsys_challenger_offline_eligible 0" in text
    assert "recsys_recommend_requests_total 1" in text
    assert 'route="warm_user_hybrid"' in text
