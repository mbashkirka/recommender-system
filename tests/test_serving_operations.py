from __future__ import annotations

from pathlib import Path

from test_serving_engine import make_bundle

from recsys_lab.serving.engine import RecommendationEngine
from recsys_lab.serving.operations import benchmark_engine


def test_benchmark_engine_mixes_warm_and_cold_requests(tmp_path: Path) -> None:
    engine = RecommendationEngine(make_bundle(tmp_path / "bundle.npz"))
    result = benchmark_engine(engine, requests=20, cold_share=0.5, k=2, seed=42)
    assert result["requests"] == 20
    assert result["error_rate"] == 0.0
    assert result["empty_response_rate"] == 0.0
    assert result["throughput_rps"] > 0.0
    assert result["latency_ms"]["p99"] >= result["latency_ms"]["p50"]
    assert any(row["cold_user"] for row in result["records"])
    assert any(not row["cold_user"] for row in result["records"])


def test_benchmark_engine_validates_arguments(tmp_path: Path) -> None:
    engine = RecommendationEngine(make_bundle(tmp_path / "bundle.npz"))
    try:
        benchmark_engine(engine, requests=0, cold_share=0.2, k=2, seed=42)
    except ValueError as exc:
        assert "requests" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")
