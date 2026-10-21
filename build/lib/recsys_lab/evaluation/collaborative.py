from __future__ import annotations

import json
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from recsys_lab.config import Settings
from recsys_lab.evaluation.metrics import ranking_metrics
from recsys_lab.evaluation.offline import build_ground_truth, evaluate_one_model
from recsys_lab.evaluation.split import load_split
from recsys_lab.models.baselines import PopularityRecommender, build_history
from recsys_lab.models.collaborative import PureSVDRecommender, fit_puresvd


def _mean_metric_at_k(
    model: object,
    ground_truth: dict[int, set[int]],
    k: int,
) -> dict[str, float]:
    rows: list[dict[str, float]] = []
    for user_id, relevant in ground_truth.items():
        recs = model.recommend(user_id, k)  # type: ignore[attr-defined]
        rows.append(ranking_metrics(recs, relevant, k).as_dict())
    frame = pd.DataFrame(rows)
    return {
        "recall": float(frame["recall"].mean()),
        "precision": float(frame["precision"].mean()),
        "hit_rate": float(frame["hit_rate"].mean()),
        "ndcg": float(frame["ndcg"].mean()),
        "map": float(frame["map"].mean()),
        "mrr": float(frame["mrr"].mean()),
    }


def _candidate_recall(
    model: object,
    ground_truth: dict[int, set[int]],
    ks: tuple[int, ...],
) -> dict[int, float]:
    max_k = max(ks)
    recs = {
        user_id: model.recommend(user_id, max_k)  # type: ignore[attr-defined]
        for user_id in ground_truth
    }
    result: dict[int, float] = {}
    for k in ks:
        values = []
        for user_id, relevant in ground_truth.items():
            retrieved = set(recs[user_id][:k])
            values.append(len(retrieved & relevant) / len(relevant) if relevant else 0.0)
        result[k] = float(sum(values) / len(values)) if values else 0.0
    return result


def run_m3_collaborative(settings: Settings) -> dict[str, Any]:
    """Tune collaborative retrieval on validation only and persist the selected artifact."""
    train = load_split(settings, "train")
    validation = load_split(settings, "validation")
    ground_truth, cohort = build_ground_truth(train, validation)
    if not ground_truth:
        raise ValueError("M3 found no warm positive validation targets")
    if settings.collaborative is None:
        raise ValueError("Missing [collaborative] configuration")

    cfg = settings.collaborative
    max_rank = max(cfg.ranks)
    decomposition = fit_puresvd(train, max_rank=max_rank, seed=settings.seed)
    history = build_history(train)

    report_dir = settings.paths.reports / "m3"
    artifact_dir = settings.paths.artifacts / "m3"
    report_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # Popularity is the M2 champion and is intentionally recomputed from train only.
    popularity = PopularityRecommender(history)
    pop_primary = _mean_metric_at_k(popularity, ground_truth, cfg.tuning_k)
    pop_candidate = _candidate_recall(popularity, ground_truth, cfg.candidate_ks)

    tuning_rows: list[dict[str, Any]] = []
    model_cache: dict[tuple[int, float], PureSVDRecommender] = {}
    for rank in cfg.ranks:
        for blend in cfg.popularity_blends:
            model = PureSVDRecommender(decomposition, rank=rank, popularity_blend=blend)
            model_cache[(rank, blend)] = model
            primary = _mean_metric_at_k(model, ground_truth, cfg.tuning_k)
            candidate = _candidate_recall(model, ground_truth, cfg.candidate_ks)
            tuning_rows.append(
                {
                    "model": model.name,
                    "rank": int(rank),
                    "popularity_blend": float(blend),
                    f"ndcg@{cfg.tuning_k}": primary["ndcg"],
                    f"recall@{cfg.tuning_k}": primary["recall"],
                    f"map@{cfg.tuning_k}": primary["map"],
                    f"hit_rate@{cfg.tuning_k}": primary["hit_rate"],
                    **{f"candidate_recall@{k}": value for k, value in candidate.items()},
                }
            )

    tuning = pd.DataFrame(tuning_rows).sort_values(
        [f"ndcg@{cfg.tuning_k}", f"candidate_recall@{max(cfg.candidate_ks)}"],
        ascending=[False, False],
    )
    tuning.to_csv(report_dir / "puresvd_tuning.csv", index=False)
    best_row = tuning.iloc[0]
    best_rank = int(best_row["rank"])
    best_blend = float(best_row["popularity_blend"])
    best_model = model_cache[(best_rank, best_blend)]

    # Full M2-style ranking table for the selected model, preserving comparable columns.
    sample_rows: list[dict[str, Any]] = []
    selected_rows = evaluate_one_model(
        model=best_model,
        model_name="puresvd_selected",
        ground_truth=ground_truth,
        history=history,
        ks=tuple(settings.evaluation.ks if settings.evaluation else (5, 10, 20)),
        report_dir=report_dir,
        sample_rows=sample_rows,
    )
    pd.DataFrame(selected_rows).to_csv(report_dir / "selected_validation_metrics.csv", index=False)
    pd.DataFrame(sample_rows).to_csv(report_dir / "sample_recommendations.csv", index=False)

    selected_primary = _mean_metric_at_k(best_model, ground_truth, cfg.tuning_k)
    selected_candidate = _candidate_recall(best_model, ground_truth, cfg.candidate_ks)
    comparison_rows = [
        {
            "model": "popularity_m2_champion",
            f"ndcg@{cfg.tuning_k}": pop_primary["ndcg"],
            f"recall@{cfg.tuning_k}": pop_primary["recall"],
            **{f"candidate_recall@{k}": v for k, v in pop_candidate.items()},
        },
        {
            "model": "puresvd_selected",
            f"ndcg@{cfg.tuning_k}": selected_primary["ndcg"],
            f"recall@{cfg.tuning_k}": selected_primary["recall"],
            **{f"candidate_recall@{k}": v for k, v in selected_candidate.items()},
        },
    ]
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(report_dir / "m2_vs_m3.csv", index=False)

    plot_cols = [f"ndcg@{cfg.tuning_k}", f"candidate_recall@{max(cfg.candidate_ks)}"]
    comparison.set_index("model")[plot_cols].plot(kind="bar")
    plt.title("M2 popularity vs M3 collaborative retrieval")
    plt.ylabel("Mean user metric")
    plt.xlabel("")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(report_dir / "m2_vs_m3.png", dpi=160, bbox_inches="tight")
    plt.close()

    best_model.save(artifact_dir / "puresvd_selected.npz")
    metadata = {
        "model_type": "PureSVDRecommender",
        "fit_split": "train",
        "tuning_split": "validation",
        "test_split_accessed": False,
        "interaction_semantics": "binary implicit positives only",
        "rank": best_rank,
        "popularity_blend": best_blend,
        "positive_threshold": settings.data.implicit_positive_threshold,
        "users": len(decomposition.user_ids),
        "items": len(decomposition.item_ids),
        "max_fitted_rank": int(decomposition.max_rank),
    }
    (artifact_dir / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    primary_ndcg_key = f"ndcg@{cfg.tuning_k}"
    candidate_key = f"candidate_recall@{max(cfg.candidate_ks)}"
    manifest = {
        "milestone": "M3",
        "fit_split": "train",
        "tuning_split": "validation",
        "test_split_accessed": False,
        "primary_segment": "warm_user_warm_item",
        "evaluation_users": len(ground_truth),
        "validation_cohort": cohort,
        "search_space": {
            "ranks": list(cfg.ranks),
            "popularity_blends": list(cfg.popularity_blends),
            "tuning_k": cfg.tuning_k,
            "candidate_ks": list(cfg.candidate_ks),
        },
        "selected": metadata,
        "m2_popularity": comparison_rows[0],
        "m3_selected": comparison_rows[1],
        "ndcg_lift_vs_popularity_pct": float(
            100.0
            * (comparison_rows[1][primary_ndcg_key] - comparison_rows[0][primary_ndcg_key])
            / comparison_rows[0][primary_ndcg_key]
        )
        if comparison_rows[0][primary_ndcg_key] != 0
        else 0.0,
        "candidate_recall_lift_vs_popularity_pct": float(
            100.0
            * (comparison_rows[1][candidate_key] - comparison_rows[0][candidate_key])
            / comparison_rows[0][candidate_key]
        )
        if comparison_rows[0][candidate_key] != 0
        else 0.0,
        "generated_reports": sorted(p.name for p in report_dir.iterdir() if p.is_file()),
    }
    (artifact_dir / "m3_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest
