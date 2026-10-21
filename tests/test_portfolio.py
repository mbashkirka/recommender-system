from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from recsys_lab import portfolio


def _settings(root: Path):
    for path in [
        root / "docs",
        root / "artifacts" / "m10",
        root / "reports" / "m10",
        root / ".github" / "workflows",
    ]:
        path.mkdir(parents=True, exist_ok=True)
    for name in ["README.md", "Dockerfile"]:
        (root / name).write_text("ok\n", encoding="utf-8")
    for name in ["ARCHITECTURE.md", "RUNBOOK.md", "INTERVIEW_GUIDE.md"]:
        (root / "docs" / name).write_text("ok\n", encoding="utf-8")
    (root / ".github" / "workflows" / "ci.yml").write_text("name: ci\n", encoding="utf-8")
    return SimpleNamespace(
        paths=SimpleNamespace(
            root=root,
            artifacts=root / "artifacts",
            reports=root / "reports",
        ),
        split=SimpleNamespace(primary_eval_segment="warm_user_warm_item"),
    )


def _final_test() -> dict:
    integrity = {"passed": True}
    base = {
        "users": 2,
        "recall": 0.1,
        "precision": 0.1,
        "hit_rate": 0.2,
        "ndcg": 0.1,
        "map": 0.05,
        "mrr": 0.1,
        "candidate_recall@100": 0.3,
        "catalog_coverage@10": 0.02,
    }
    deployed = dict(base)
    deployed.update({"recall": 0.2, "ndcg": 0.15, "candidate_recall@100": 0.4})
    return {
        "protocol": "single_final_evaluation_of_frozen_serving_policy",
        "primary_segment": "warm_user_warm_item",
        "artifact_fingerprint": "abc",
        "artifact_integrity": integrity,
        "cohort": {"warm_positive_event_rate": 0.8},
        "no_refit_on_test": True,
        "no_hyperparameter_selection_on_test": True,
        "popularity_baseline": base,
        "deployed_policy": deployed,
        "lift_vs_popularity_pct": {
            "ndcg@10": 50.0,
            "recall@10": 100.0,
            "candidate_recall@100": 33.3,
        },
    }


def test_mean_metrics() -> None:
    rows = [
        {"recall": 0.2, "precision": 0.1, "hit_rate": 1.0, "ndcg": 0.3, "map": 0.2, "mrr": 0.5},
        {"recall": 0.4, "precision": 0.3, "hit_rate": 0.0, "ndcg": 0.1, "map": 0.0, "mrr": 0.0},
    ]
    result = portfolio._mean_metrics(rows)
    assert result["users"] == 2
    assert abs(float(result["recall"]) - 0.3) < 1e-12
    assert abs(float(result["ndcg"]) - 0.2) < 1e-12


def test_safe_json_rejects_invalid(tmp_path: Path) -> None:
    path = tmp_path / "x.json"
    path.write_text("not-json", encoding="utf-8")
    assert portfolio._safe_json(path) is None


def test_write_final_results(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    portfolio._write_final_results(settings, _final_test())
    text = (tmp_path / "docs" / "FINAL_RESULTS.md").read_text(encoding="utf-8")
    assert "Frozen Production Policy" in text
    assert "50.00%" in text
    assert "no model or hyperparameter" in text


def test_m10_locks_first_final_test_and_reuses_cache(tmp_path: Path, monkeypatch) -> None:
    settings = _settings(tmp_path)
    calls = {"count": 0}

    def fake_eval(_settings):
        calls["count"] += 1
        return _final_test()

    monkeypatch.setattr(portfolio, "_evaluate_frozen_test", fake_eval)
    first = portfolio.run_m10_portfolio(settings)
    second = portfolio.run_m10_portfolio(settings)
    assert calls["count"] == 1
    assert first["test_split_accessed_this_run"] is True
    assert second["test_split_accessed_this_run"] is False
    assert (tmp_path / "artifacts" / "m10" / "final_test_locked.json").exists()


def test_m10_manifest_is_json(tmp_path: Path, monkeypatch) -> None:
    settings = _settings(tmp_path)
    monkeypatch.setattr(portfolio, "_evaluate_frozen_test", lambda _: _final_test())
    portfolio.run_m10_portfolio(settings)
    payload = json.loads(
        (tmp_path / "artifacts" / "m10" / "m10_manifest.json").read_text(encoding="utf-8")
    )
    assert payload["release"] == "1.0.0"
    assert payload["status"] == "PASS"
