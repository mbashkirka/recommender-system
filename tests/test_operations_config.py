from __future__ import annotations

import pytest

from recsys_lab.config import OperationsConfig, load_settings


def valid_config(**overrides) -> OperationsConfig:
    values = {
        "benchmark_requests": 100,
        "benchmark_cold_share": 0.2,
        "benchmark_top_k": 10,
        "max_p95_latency_ms": 50.0,
        "max_p99_latency_ms": 100.0,
        "max_error_rate": 0.0,
        "min_throughput_rps": 20.0,
        "metrics_window": 5000,
    }
    values.update(overrides)
    return OperationsConfig(**values)


def test_project_operations_config_is_valid() -> None:
    settings = load_settings()
    assert settings.operations is not None
    settings.operations.validate()


def test_operations_rejects_invalid_cold_share() -> None:
    with pytest.raises(ValueError):
        valid_config(benchmark_cold_share=1.1).validate()


def test_operations_rejects_p99_below_p95() -> None:
    with pytest.raises(ValueError):
        valid_config(max_p95_latency_ms=100.0, max_p99_latency_ms=50.0).validate()


def test_operations_rejects_empty_metrics_window() -> None:
    with pytest.raises(ValueError):
        valid_config(metrics_window=0).validate()
