from __future__ import annotations

import json
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from recsys_lab.config import Settings
from recsys_lab.evaluation.collaborative import (
    _candidate_recall,
    _mean_metric_at_k,
    run_m3_collaborative,
)
from recsys_lab.evaluation.offline import build_ground_truth, evaluate_one_model
from recsys_lab.evaluation.split import load_split
from recsys_lab.models.baselines import build_history
from recsys_lab.models.collaborative import fit_puresvd
from recsys_lab.models.hybrid import HybridSVDMetadataRecommender


def _load_movies(settings: Settings) -> pd.DataFrame:
    path = settings.paths.processed / "movies.csv.gz"
    if not path.exists():
        raise FileNotFoundError(f"Processed movies missing at {path}. Run `prepare` first.")
    return pd.read_csv(path)


def _segment_ground_truth(
    train: pd.DataFrame,
    evaluation: pd.DataFrame,
    *,
    warm_user: bool | None,
    warm_item: bool | None,
) -> dict[int, set[int]]:
    train_users = set(train["user_id"].astype(int).unique())
    train_items = set(train["item_id"].astype(int).unique())
    positive = evaluation.loc[evaluation["is_positive"].astype(bool), ["user_id", "item_id"]].copy()
    positive["user_id"] = positive["user_id"].astype(int)
    positive["item_id"] = positive["item_id"].astype(int)
    mask = pd.Series(True, index=positive.index)
    if warm_user is not None:
        mask &= positive["user_id"].isin(train_users).eq(warm_user)
    if warm_item is not None:
        mask &= positive["item_id"].isin(train_items).eq(warm_item)
    selected = positive.loc[mask]
    return {
        int(user_id): set(group["item_id"].astype(int))
        for user_id, group in selected.groupby("user_id", sort=True)
    }


def _safe_candidate_recall(model: object, gt: dict[int, set[int]], k: int) -> float:
    if not gt:
        return 0.0
    return _candidate_recall(model, gt, (k,))[k]


def _read_or_build_m3(settings: Settings) -> dict[str, Any]:
    manifest_path = settings.paths.artifacts / "m3" / "m3_manifest.json"
    if manifest_path.exists():
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    return run_m3_collaborative(settings)


def run_m4_hybrid(settings: Settings) -> dict[str, Any]:
    """Strengthen M3 retrieval and validate explicit cold-start routing on validation only."""
    if settings.collaborative is None or settings.hybrid is None:
        raise ValueError("M4 requires [collaborative] and [hybrid] configuration")

    train = load_split(settings, "train")
    validation = load_split(settings, "validation")
    movies = _load_movies(settings)
    warm_gt, warm_diag = build_ground_truth(train, validation)
    if not warm_gt:
        raise ValueError("M4 found no warm positive validation targets")

    m3 = _read_or_build_m3(settings)
    selected_m3 = m3["selected"]
    rank = int(selected_m3["rank"])
    popularity_blend = float(selected_m3["popularity_blend"])

    # Fit only the rank already selected by M3. No test data and no extra rank re-tuning.
    decomposition = fit_puresvd(train, max_rank=max(settings.collaborative.ranks), seed=settings.seed)
    history = build_history(train)
    cfg = settings.hybrid
    report_dir = settings.paths.reports / "m4"
    artifact_dir = settings.paths.artifacts / "m4"
    report_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    m3_ndcg = float(m3["m3_selected"][f"ndcg@{settings.collaborative.tuning_k}"])
    m3_candidate_key = f"candidate_recall@{max(settings.collaborative.candidate_ks)}"
    m3_candidate = float(m3["m3_selected"][m3_candidate_key])

    tuning_rows: list[dict[str, Any]] = []
    models: dict[float, HybridSVDMetadataRecommender] = {}
    for metadata_blend in cfg.metadata_blends:
        model = HybridSVDMetadataRecommender(
            decomposition=decomposition,
            movies=movies,
            rank=rank,
            popularity_blend=popularity_blend,
            metadata_blend=metadata_blend,
            sparse_positive_threshold=cfg.sparse_positive_threshold,
            sparse_metadata_multiplier=cfg.sparse_metadata_multiplier,
            cold_item_penalty=cfg.cold_item_penalty,
        )
        models[metadata_blend] = model
        primary = _mean_metric_at_k(model, warm_gt, settings.collaborative.tuning_k)
        candidate = _candidate_recall(model, warm_gt, settings.collaborative.candidate_ks)
        tuning_rows.append(
            {
                "model": model.name,
                "metadata_blend": float(metadata_blend),
                f"ndcg@{settings.collaborative.tuning_k}": primary["ndcg"],
                f"recall@{settings.collaborative.tuning_k}": primary["recall"],
                **{f"candidate_recall@{k}": v for k, v in candidate.items()},
            }
        )

    tuning = pd.DataFrame(tuning_rows)
    ndcg_key = f"ndcg@{settings.collaborative.tuning_k}"
    candidate_key = f"candidate_recall@{max(settings.collaborative.candidate_ks)}"
    tuning["passes_ndcg_guardrail"] = tuning[ndcg_key].ge(m3_ndcg * cfg.ndcg_guardrail_ratio)
    eligible = tuning.loc[tuning["passes_ndcg_guardrail"]].copy()
    if eligible.empty:
        eligible = tuning.loc[tuning["metadata_blend"].eq(0.0)].copy()
        if eligible.empty:
            raise RuntimeError("M4 no-regression anchor is missing from the tuning table")
    eligible = eligible.sort_values([candidate_key, ndcg_key], ascending=[False, False])
    best = eligible.iloc[0]
    best_blend = float(best["metadata_blend"])
    best_model = models[best_blend]
    tuning.sort_values([candidate_key, ndcg_key], ascending=[False, False]).to_csv(
        report_dir / "hybrid_tuning.csv", index=False
    )

    # Comparable warm ranking reports.
    sample_rows: list[dict[str, Any]] = []
    selected_rows = evaluate_one_model(
        model=best_model,
        model_name="hybrid_selected",
        ground_truth=warm_gt,
        history=history,
        ks=tuple(settings.evaluation.ks if settings.evaluation else (5, 10, 20)),
        report_dir=report_dir,
        sample_rows=sample_rows,
    )
    pd.DataFrame(selected_rows).to_csv(report_dir / "selected_validation_metrics.csv", index=False)
    pd.DataFrame(sample_rows).to_csv(report_dir / "sample_recommendations.csv", index=False)

    # Explicit validation cohorts that were intentionally excluded from M2/M3 primary metrics.
    cold_item_gt = _segment_ground_truth(
        train, validation, warm_user=True, warm_item=False
    )
    cold_user_gt = _segment_ground_truth(
        train, validation, warm_user=False, warm_item=True
    )
    cold_both_gt = _segment_ground_truth(
        train, validation, warm_user=False, warm_item=False
    )
    cold_k = cfg.cold_start_k
    cold_item_values = []
    for user_id, relevant in cold_item_gt.items():
        recs = best_model.recommend_with_cold_item_quota(
            user_id, cold_k, cfg.cold_item_quota
        )
        cold_item_values.append(len(set(recs) & relevant) / len(relevant) if relevant else 0.0)
    cold_item_recall = (
        float(sum(cold_item_values) / len(cold_item_values)) if cold_item_values else 0.0
    )
    cold_metrics = {
        "warm_user_cold_item": {
            "users": len(cold_item_gt),
            f"recall@{cold_k}": cold_item_recall,
            "cold_item_quota": cfg.cold_item_quota,
            "route": "metadata-ranked cold-item injection",
        },
        "cold_user_warm_item": {
            "users": len(cold_user_gt),
            f"recall@{cold_k}": _safe_candidate_recall(best_model, cold_user_gt, cold_k),
            "route": "global train popularity",
        },
        "cold_user_cold_item": {
            "users": len(cold_both_gt),
            f"recall@{cold_k}": _safe_candidate_recall(best_model, cold_both_gt, cold_k),
            "route": "global train popularity + metadata catalog",
        },
    }
    (report_dir / "cold_start_metrics.json").write_text(
        json.dumps(cold_metrics, indent=2) + "\n", encoding="utf-8"
    )

    best_model.save(artifact_dir / "hybrid_selected.npz")
    model_metadata = {
        "model_type": "HybridSVDMetadataRecommender",
        "fit_split": "train",
        "tuning_split": "validation",
        "test_split_accessed": False,
        "rank": rank,
        "popularity_blend": popularity_blend,
        "metadata_blend": best_blend,
        "sparse_positive_threshold": cfg.sparse_positive_threshold,
        "sparse_metadata_multiplier": cfg.sparse_metadata_multiplier,
        "cold_item_penalty": cfg.cold_item_penalty,
        "cold_start_k": cfg.cold_start_k,
        "cold_item_quota": cfg.cold_item_quota,
        "static_metadata_features": list(best_model.metadata.feature_names),
    }
    (artifact_dir / "model_metadata.json").write_text(
        json.dumps(model_metadata, indent=2) + "\n", encoding="utf-8"
    )

    route_rows = []
    for user_id in sorted(set(validation["user_id"].astype(int))):
        route_rows.append({"user_id": user_id, "route": best_model.route_for_user(user_id)})
    route_frame = pd.DataFrame(route_rows)
    route_summary = route_frame.groupby("route").size().rename("users").reset_index()
    route_summary.to_csv(report_dir / "routing_summary.csv", index=False)

    comparison = pd.DataFrame(
        [
            {
                "model": "m3_puresvd",
                ndcg_key: m3_ndcg,
                candidate_key: m3_candidate,
            },
            {
                "model": "m4_hybrid",
                ndcg_key: float(best[ndcg_key]),
                candidate_key: float(best[candidate_key]),
            },
        ]
    )
    comparison.to_csv(report_dir / "m3_vs_m4.csv", index=False)
    comparison.set_index("model")[[ndcg_key, candidate_key]].plot(kind="bar")
    plt.title("M3 vs M4 validation retrieval")
    plt.ylabel("Mean user metric")
    plt.xlabel("Model")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(report_dir / "m3_vs_m4.png", dpi=160, bbox_inches="tight")
    plt.close()

    manifest: dict[str, Any] = {
        "milestone": "M4",
        "fit_split": "train",
        "tuning_split": "validation",
        "test_split_accessed": False,
        "primary_segment": "warm_user_warm_item",
        "selection_objective": candidate_key,
        "ndcg_guardrail_ratio": cfg.ndcg_guardrail_ratio,
        "m3_fixed": {
            "rank": rank,
            "popularity_blend": popularity_blend,
            ndcg_key: m3_ndcg,
            candidate_key: m3_candidate,
        },
        "selected": {
            "metadata_blend": best_blend,
            ndcg_key: float(best[ndcg_key]),
            candidate_key: float(best[candidate_key]),
            "passes_ndcg_guardrail": bool(best["passes_ndcg_guardrail"]),
        },
        "warm_validation": warm_diag,
        "cold_start": cold_metrics,
        "routing": {str(row["route"]): int(row["users"]) for _, row in route_summary.iterrows()},
        "candidate_recall_lift_vs_m3_pct": (
            100.0 * (float(best[candidate_key]) - m3_candidate) / m3_candidate
            if m3_candidate
            else 0.0
        ),
        "ndcg_lift_vs_m3_pct": (
            100.0 * (float(best[ndcg_key]) - m3_ndcg) / m3_ndcg if m3_ndcg else 0.0
        ),
        "generated_reports": sorted(p.name for p in report_dir.iterdir() if p.is_file()),
    }
    (artifact_dir / "m4_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest
