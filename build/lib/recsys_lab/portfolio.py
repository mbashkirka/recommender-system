from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from recsys_lab.config import Settings
from recsys_lab.evaluation.metrics import ranking_metrics
from recsys_lab.evaluation.offline import build_ground_truth
from recsys_lab.evaluation.split import load_split
from recsys_lab.models.baselines import PopularityRecommender, build_history
from recsys_lab.serving.engine import RecommendationEngine
from recsys_lab.serving.integrity import verify_serving_artifact


def _mean_metrics(rows: list[dict[str, float | int]]) -> dict[str, float | int]:
    if not rows:
        return {
            "users": 0,
            "recall": 0.0,
            "precision": 0.0,
            "hit_rate": 0.0,
            "ndcg": 0.0,
            "map": 0.0,
            "mrr": 0.0,
        }
    keys = ("recall", "precision", "hit_rate", "ndcg", "map", "mrr")
    return {
        "users": len(rows),
        **{key: float(sum(float(row[key]) for row in rows) / len(rows)) for key in keys},
    }


def _safe_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _evaluate_frozen_test(settings: Settings) -> dict[str, Any]:
    bundle = settings.paths.artifacts / "m8" / "serving_bundle.npz"
    m8_manifest = settings.paths.artifacts / "m8" / "m8_manifest.json"
    if not bundle.exists() or not m8_manifest.exists():
        raise FileNotFoundError("M8 serving artifact is missing. Run `python -m recsys_lab m8` first.")

    integrity = verify_serving_artifact(bundle, m8_manifest)
    if not integrity["passed"]:
        raise RuntimeError("Frozen serving artifact failed integrity verification before final test.")

    train = load_split(settings, "train")
    test = load_split(settings, "test")
    ground_truth, cohort = build_ground_truth(train, test)
    if not ground_truth:
        raise RuntimeError("Final test cohort is empty after warm-user/warm-item filtering.")

    engine = RecommendationEngine(bundle)
    history = build_history(train)
    popularity = PopularityRecommender(history)
    primary_k = settings.evaluation.primary_k
    candidate_k = 100

    deployed_rows: list[dict[str, float | int]] = []
    popularity_rows: list[dict[str, float | int]] = []
    deployed_candidate_recalls: list[float] = []
    popularity_candidate_recalls: list[float] = []
    deployed_top10: list[list[int]] = []
    popularity_top10: list[list[int]] = []
    route_counts: dict[str, int] = {}
    per_user_rows: list[dict[str, Any]] = []

    for user_id, relevant in ground_truth.items():
        deployed_100, meta = engine.recommend(
            user_id=user_id,
            k=candidate_k,
            policy="control",
            candidate_k=candidate_k,
        )
        deployed_ids = [rec.item_id for rec in deployed_100]
        popularity_ids = popularity.recommend(user_id, candidate_k)

        deployed_k = deployed_ids[:primary_k]
        popularity_k = popularity_ids[:primary_k]
        deployed_metrics = ranking_metrics(deployed_k, relevant, primary_k).as_dict()
        popularity_metrics = ranking_metrics(popularity_k, relevant, primary_k).as_dict()
        deployed_candidate = ranking_metrics(deployed_ids, relevant, candidate_k)
        popularity_candidate = ranking_metrics(popularity_ids, relevant, candidate_k)

        deployed_rows.append(deployed_metrics)
        popularity_rows.append(popularity_metrics)
        deployed_candidate_recalls.append(deployed_candidate.recall)
        popularity_candidate_recalls.append(popularity_candidate.recall)
        deployed_top10.append(deployed_k)
        popularity_top10.append(popularity_k)
        route = str(meta.get("route", "unknown"))
        route_counts[route] = route_counts.get(route, 0) + 1
        per_user_rows.append(
            {
                "user_id": int(user_id),
                "relevant_items": len(relevant),
                "route": route,
                "deployed_recall@10": deployed_metrics["recall"],
                "deployed_ndcg@10": deployed_metrics["ndcg"],
                "deployed_recall@100": deployed_candidate.recall,
                "popularity_recall@10": popularity_metrics["recall"],
                "popularity_ndcg@10": popularity_metrics["ndcg"],
                "popularity_recall@100": popularity_candidate.recall,
            }
        )

    deployed = _mean_metrics(deployed_rows)
    pop = _mean_metrics(popularity_rows)
    deployed["candidate_recall@100"] = float(
        sum(deployed_candidate_recalls) / len(deployed_candidate_recalls)
    )
    pop["candidate_recall@100"] = float(
        sum(popularity_candidate_recalls) / len(popularity_candidate_recalls)
    )

    train_catalog = {int(x) for x in history.catalog}
    deployed_unique = {item for row in deployed_top10 for item in row if item in train_catalog}
    popularity_unique = {item for row in popularity_top10 for item in row if item in train_catalog}
    deployed["catalog_coverage@10"] = len(deployed_unique) / len(train_catalog)
    pop["catalog_coverage@10"] = len(popularity_unique) / len(train_catalog)

    ndcg_lift = (
        (float(deployed["ndcg"]) / float(pop["ndcg"]) - 1.0) * 100.0
        if float(pop["ndcg"]) > 0
        else 0.0
    )
    recall_lift = (
        (float(deployed["recall"]) / float(pop["recall"]) - 1.0) * 100.0
        if float(pop["recall"]) > 0
        else 0.0
    )
    candidate_lift = (
        (float(deployed["candidate_recall@100"]) / float(pop["candidate_recall@100"]) - 1.0)
        * 100.0
        if float(pop["candidate_recall@100"]) > 0
        else 0.0
    )

    report_dir = settings.paths.reports / "m10"
    report_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(per_user_rows).to_csv(
        report_dir / "final_test_per_user.csv.gz", index=False, compression="gzip"
    )

    return {
        "protocol": "single_final_evaluation_of_frozen_serving_policy",
        "fit_source": "train_only_frozen_m8_serving_bundle",
        "selection_source": "validation_only_m2_to_m6",
        "evaluation_split": "test",
        "primary_segment": settings.split.primary_eval_segment,
        "test_split_accessed": True,
        "no_refit_on_test": True,
        "no_hyperparameter_selection_on_test": True,
        "artifact_fingerprint": engine.fingerprint,
        "artifact_integrity": integrity,
        "cohort": cohort,
        "routing": route_counts,
        "primary_k": primary_k,
        "candidate_k": candidate_k,
        "popularity_baseline": pop,
        "deployed_policy": deployed,
        "lift_vs_popularity_pct": {
            "ndcg@10": ndcg_lift,
            "recall@10": recall_lift,
            "candidate_recall@100": candidate_lift,
        },
    }


def _historical_summary(settings: Settings) -> dict[str, Any]:
    sources = {
        "m3": settings.paths.artifacts / "m3" / "m3_manifest.json",
        "m4": settings.paths.artifacts / "m4" / "m4_manifest.json",
        "m5": settings.paths.artifacts / "m5" / "m5_manifest.json",
        "m6": settings.paths.artifacts / "m6" / "m6_manifest.json",
        "m7": settings.paths.artifacts / "m7" / "m7_manifest.json",
        "m8": settings.paths.artifacts / "m8" / "m8_manifest.json",
        "m9": settings.paths.artifacts / "m9" / "m9_manifest.json",
    }
    return {name: payload for name, path in sources.items() if (payload := _safe_json(path))}


def _fmt(value: float) -> str:
    return f"{value:.4f}"


def _write_final_results(settings: Settings, report: dict[str, Any]) -> None:
    deployed = report["deployed_policy"]
    popularity = report["popularity_baseline"]
    lift = report["lift_vs_popularity_pct"]
    cohort = report["cohort"]
    fingerprint = report["artifact_fingerprint"]
    text = "# Final Results — Frozen Production Policy\n\n"
    text += "This file is generated by `python -m recsys_lab m10`. The test split is used only for this final, frozen-policy evaluation; no model or hyperparameter is selected from test results.\n\n"
    text += "## Final test protocol\n\n"
    text += f"- Serving artifact SHA-256: `{fingerprint}`\n"
    text += "- Fit source: train-only frozen M8 bundle\n"
    text += "- Model selection/tuning source: validation only\n"
    text += "- Final evaluation split: test\n"
    text += f"- Primary cohort: `{settings.split.primary_eval_segment}`\n"
    text += f"- Evaluation users: {int(deployed['users'])}\n"
    text += f"- Warm positive event rate in test: {float(cohort['warm_positive_event_rate']):.2%}\n\n"
    text += "## Test metrics\n\n"
    text += "| Metric | Popularity baseline | Frozen deployed policy | Lift |\n"
    text += "|---|---:|---:|---:|\n"
    text += f"| NDCG@10 | {_fmt(float(popularity['ndcg']))} | {_fmt(float(deployed['ndcg']))} | {float(lift['ndcg@10']):+.2f}% |\n"
    text += f"| Recall@10 | {_fmt(float(popularity['recall']))} | {_fmt(float(deployed['recall']))} | {float(lift['recall@10']):+.2f}% |\n"
    text += f"| Precision@10 | {_fmt(float(popularity['precision']))} | {_fmt(float(deployed['precision']))} | — |\n"
    text += f"| HitRate@10 | {_fmt(float(popularity['hit_rate']))} | {_fmt(float(deployed['hit_rate']))} | — |\n"
    text += f"| MAP@10 | {_fmt(float(popularity['map']))} | {_fmt(float(deployed['map']))} | — |\n"
    text += f"| MRR@10 | {_fmt(float(popularity['mrr']))} | {_fmt(float(deployed['mrr']))} | — |\n"
    text += f"| Candidate Recall@100 | {_fmt(float(popularity['candidate_recall@100']))} | {_fmt(float(deployed['candidate_recall@100']))} | {float(lift['candidate_recall@100']):+.2f}% |\n"
    text += f"| Catalog coverage@10 | {_fmt(float(popularity['catalog_coverage@10']))} | {_fmt(float(deployed['catalog_coverage@10']))} | — |\n\n"
    text += "## Interpretation\n\n"
    text += "The deployed policy is the exact policy admitted by the offline deployment gates. M5 selected an LTR blend of zero, and M6 rolled the diversity/novelty challenger back to the exact anchor after holdout guardrails failed. Those negative results are preserved rather than hidden: the portfolio demonstrates model selection and rollback discipline, not mandatory complexity.\n\n"
    text += "The A/B framework in M7 is a deterministic synthetic experiment harness because MovieLens does not contain randomized recommendation impressions/clicks. Its uplift must not be presented as a real online effect of this recommender.\n"
    (settings.paths.root / "docs" / "FINAL_RESULTS.md").write_text(text, encoding="utf-8")


def run_m10_portfolio(settings: Settings) -> dict[str, Any]:
    artifact_dir = settings.paths.artifacts / "m10"
    report_dir = settings.paths.reports / "m10"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = artifact_dir / "m10_manifest.json"
    locked_test_path = artifact_dir / "final_test_locked.json"

    cached = _safe_json(manifest_path)
    locked = _safe_json(locked_test_path)
    if cached and isinstance(cached.get("final_test"), dict):
        final_test = cached["final_test"]
        test_accessed_this_run = False
        evaluation_source = "cached_final_test_evaluation"
    elif locked:
        final_test = locked
        test_accessed_this_run = False
        evaluation_source = "locked_final_test_evaluation"
    else:
        final_test = _evaluate_frozen_test(settings)
        locked_test_path.write_text(
            json.dumps(final_test, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        test_accessed_this_run = True
        evaluation_source = "fresh_single_final_test_evaluation"

    historical = _historical_summary(settings)
    checks = {
        "readme_present": (settings.paths.root / "README.md").exists(),
        "architecture_doc_present": (settings.paths.root / "docs" / "ARCHITECTURE.md").exists(),
        "runbook_present": (settings.paths.root / "docs" / "RUNBOOK.md").exists(),
        "interview_guide_present": (settings.paths.root / "docs" / "INTERVIEW_GUIDE.md").exists(),
        "dockerfile_present": (settings.paths.root / "Dockerfile").exists(),
        "ci_present": (settings.paths.root / ".github" / "workflows" / "ci.yml").exists(),
        "serving_artifact_integrity": bool(final_test["artifact_integrity"]["passed"]),
        "final_test_no_refit": bool(final_test["no_refit_on_test"]),
        "final_test_no_tuning": bool(final_test["no_hyperparameter_selection_on_test"]),
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    manifest = {
        "milestone": "M10",
        "release": "1.0.0",
        "status": status,
        "model_version": "recsys-m9",
        "artifact_fingerprint": final_test["artifact_fingerprint"],
        "test_split_ever_accessed": True,
        "test_split_accessed_this_run": test_accessed_this_run,
        "evaluation_source": evaluation_source,
        "final_test": final_test,
        "historical_manifests_available": sorted(historical),
        "portfolio_checks": checks,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (report_dir / "portfolio_summary.json").write_text(
        json.dumps({"final_test": final_test, "historical": historical}, indent=2, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {"model": "popularity_test", **final_test["popularity_baseline"]},
            {"model": "deployed_policy_test", **final_test["deployed_policy"]},
        ]
    ).to_csv(report_dir / "final_test_metrics.csv", index=False)
    _write_final_results(settings, final_test)

    if status != "PASS":
        failed = [name for name, ok in checks.items() if not ok]
        raise RuntimeError(f"M10 portfolio finalization failed checks: {failed}")
    return manifest
