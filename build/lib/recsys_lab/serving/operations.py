from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from time import perf_counter
from typing import Any

from recsys_lab.config import Settings
from recsys_lab.serving.engine import RecommendationEngine
from recsys_lab.serving.integrity import verify_serving_artifact


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    pos = (len(values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    frac = pos - lo
    return float(values[lo] * (1 - frac) + values[hi] * frac)


def benchmark_engine(
    engine: RecommendationEngine,
    requests: int,
    cold_share: float,
    k: int,
    seed: int,
) -> dict[str, Any]:
    if requests <= 0:
        raise ValueError("requests must be > 0")
    if not 0.0 <= cold_share <= 1.0:
        raise ValueError("cold_share must be in [0, 1]")
    if k <= 0:
        raise ValueError("k must be > 0")

    rng = random.Random(seed)
    warm_users = [int(x) for x in engine.user_ids]
    if not warm_users:
        raise ValueError("serving bundle has no warm users")
    max_user = max(warm_users)
    records: list[dict[str, Any]] = []
    wall_start = perf_counter()
    for idx in range(requests):
        cold = rng.random() < cold_share
        user_id = max_user + 1 + idx if cold else warm_users[rng.randrange(len(warm_users))]
        started = perf_counter()
        error = None
        rec_count = 0
        route = "unknown"
        try:
            recs, meta = engine.recommend(user_id=user_id, k=k, policy="control")
            rec_count = len(recs)
            route = str(meta.get("route", "unknown"))
        except Exception as exc:  # noqa: BLE001 - benchmark intentionally records all failures
            error = type(exc).__name__
        latency_ms = (perf_counter() - started) * 1000.0
        records.append(
            {
                "request": idx + 1,
                "user_id": user_id,
                "cold_user": cold,
                "route": route,
                "latency_ms": latency_ms,
                "recommendations": rec_count,
                "error": error or "",
            }
        )
    wall_seconds = max(perf_counter() - wall_start, 1e-12)
    latencies = [float(row["latency_ms"]) for row in records]
    errors = sum(bool(row["error"]) for row in records)
    empty = sum(int(row["recommendations"]) == 0 for row in records)
    return {
        "requests": requests,
        "cold_share": cold_share,
        "top_k": k,
        "wall_seconds": wall_seconds,
        "throughput_rps": requests / wall_seconds,
        "error_rate": errors / requests,
        "empty_response_rate": empty / requests,
        "latency_ms": {
            "mean": sum(latencies) / len(latencies),
            "p50": _quantile(latencies, 0.50),
            "p95": _quantile(latencies, 0.95),
            "p99": _quantile(latencies, 0.99),
            "max": max(latencies),
        },
        "records": records,
    }


def _is_writable(directory: Path) -> bool:
    directory.mkdir(parents=True, exist_ok=True)
    probe = directory / ".m9_write_probe"
    try:
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return True
    except OSError:
        return False


def run_m9_preflight(settings: Settings) -> dict[str, Any]:
    if settings.operations is None:
        raise ValueError("[operations] configuration is required for M9")
    ops = settings.operations
    bundle = settings.paths.artifacts / "m8" / "serving_bundle.npz"
    m8_manifest = settings.paths.artifacts / "m8" / "m8_manifest.json"
    if not bundle.exists() or not m8_manifest.exists():
        raise FileNotFoundError("M8 serving artifact is missing. Run `python -m recsys_lab m8` first.")

    integrity = verify_serving_artifact(bundle, m8_manifest)
    engine = RecommendationEngine(bundle)
    bench = benchmark_engine(
        engine,
        requests=ops.benchmark_requests,
        cold_share=ops.benchmark_cold_share,
        k=min(ops.benchmark_top_k, engine.primary_k),
        seed=settings.seed,
    )

    report_dir = settings.paths.reports / "m9"
    artifact_dir = settings.paths.artifacts / "m9"
    report_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    with (report_dir / "benchmark_requests.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(bench["records"][0].keys()))
        writer.writeheader()
        writer.writerows(bench["records"])

    checks = {
        "artifact_integrity": bool(integrity["passed"]),
        "p95_latency": bench["latency_ms"]["p95"] <= ops.max_p95_latency_ms,
        "p99_latency": bench["latency_ms"]["p99"] <= ops.max_p99_latency_ms,
        "error_rate": bench["error_rate"] <= ops.max_error_rate,
        "nonempty_responses": bench["empty_response_rate"] == 0.0,
        "throughput": bench["throughput_rps"] >= ops.min_throughput_rps,
        "online_event_store_writable": _is_writable(settings.paths.root / "data" / "online"),
        "dockerfile_present": (settings.paths.root / "Dockerfile").exists(),
        "docker_config_contract": (
            settings.paths.root / "Dockerfile"
        ).exists()
        and "RECSYS_CONFIG=/app/config/project.toml"
        in (settings.paths.root / "Dockerfile").read_text(encoding="utf-8"),
        "ci_workflow_present": (settings.paths.root / ".github" / "workflows" / "ci.yml").exists(),
    }
    passed = all(checks.values())
    summary = {key: value for key, value in bench.items() if key != "records"}
    report = {
        "milestone": "M9",
        "status": "PASS" if passed else "FAIL",
        "model_version": "recsys-m9",
        "artifact_fingerprint": engine.fingerprint,
        "validation_split_accessed": False,
        "test_split_accessed": False,
        "integrity": integrity,
        "benchmark": summary,
        "thresholds": {
            "max_p95_latency_ms": ops.max_p95_latency_ms,
            "max_p99_latency_ms": ops.max_p99_latency_ms,
            "max_error_rate": ops.max_error_rate,
            "min_throughput_rps": ops.min_throughput_rps,
        },
        "checks": checks,
    }
    (report_dir / "operational_preflight.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    manifest = {
        "milestone": "M9",
        "status": report["status"],
        "serving_artifact_sha256": engine.fingerprint,
        "source_splits": ["train"],
        "validation_split_accessed": False,
        "test_split_accessed": False,
        "checks": checks,
        "benchmark_summary": summary,
    }
    (artifact_dir / "m9_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if not passed:
        failed = [name for name, ok in checks.items() if not ok]
        raise RuntimeError(f"M9 operational preflight failed checks: {failed}")
    return report
