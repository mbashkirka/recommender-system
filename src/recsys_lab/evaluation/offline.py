from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from recsys_lab.config import Settings
from recsys_lab.evaluation.metrics import ranking_metrics
from recsys_lab.evaluation.split import load_split
from recsys_lab.models.baselines import FittedHistory, build_recommenders


def _load_movies(settings: Settings) -> pd.DataFrame:
    path = settings.paths.processed / "movies.csv.gz"
    if not path.exists():
        raise FileNotFoundError(f"Processed movies missing at {path}. Run `prepare` first.")
    return pd.read_csv(path)


def build_ground_truth(
    train: pd.DataFrame,
    evaluation: pd.DataFrame,
) -> tuple[dict[int, set[int]], dict[str, Any]]:
    """Build the primary warm-user/warm-item positive evaluation cohort."""
    train_users = set(train["user_id"].astype(int).unique())
    train_items = set(train["item_id"].astype(int).unique())
    positive = evaluation.loc[evaluation["is_positive"].astype(bool)].copy()
    positive["user_id"] = positive["user_id"].astype(int)
    positive["item_id"] = positive["item_id"].astype(int)

    warm_mask = positive["user_id"].isin(train_users) & positive["item_id"].isin(train_items)
    warm = positive.loc[warm_mask].copy()

    # A target already observed for the same user in fit history cannot be recommended because
    # seen items are excluded. MovieLens has one rating per user-item, but keep the evaluator
    # correct for generic repeated-interaction datasets too.
    seen_pairs = set(
        zip(train["user_id"].astype(int), train["item_id"].astype(int), strict=False)
    )
    repeated_mask = pd.Series(
        [
            (int(user_id), int(item_id)) in seen_pairs
            for user_id, item_id in zip(warm["user_id"], warm["item_id"], strict=False)
        ],
        index=warm.index,
        dtype=bool,
    )
    repeated_positive_events = int(repeated_mask.sum())
    warm = warm.loc[~repeated_mask]
    ground_truth = {
        int(user_id): set(group["item_id"].astype(int))
        for user_id, group in warm.groupby("user_id", sort=True)
    }

    diagnostics = {
        "evaluation_rows": len(evaluation),
        "positive_events": len(positive),
        "warm_positive_events": len(warm),
        "repeated_seen_positive_events_excluded": repeated_positive_events,
        "warm_positive_event_rate": float(len(warm) / len(positive)) if len(positive) else 0.0,
        "evaluation_users_with_warm_positive": len(ground_truth),
        "mean_relevant_items_per_eval_user": float(np.mean([len(v) for v in ground_truth.values()]))
        if ground_truth
        else 0.0,
        "median_relevant_items_per_eval_user": float(
            np.median([len(v) for v in ground_truth.values()])
        )
        if ground_truth
        else 0.0,
    }
    return ground_truth, diagnostics


def _aggregate_user_metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    if frame.empty:
        return {
            "users": 0,
            "recall": 0.0,
            "precision": 0.0,
            "hit_rate": 0.0,
            "ndcg": 0.0,
            "map": 0.0,
            "mrr": 0.0,
        }
    return {
        "users": len(frame),
        "recall": float(frame["recall"].mean()),
        "precision": float(frame["precision"].mean()),
        "hit_rate": float(frame["hit_rate"].mean()),
        "ndcg": float(frame["ndcg"].mean()),
        "map": float(frame["map"].mean()),
        "mrr": float(frame["mrr"].mean()),
    }


def _catalog_metrics(
    all_recommendations: list[list[int]],
    history: FittedHistory,
) -> dict[str, float | int]:
    flattened = [item for recs in all_recommendations for item in recs]
    unique_items = set(flattened)
    catalog_size = len(history.catalog)
    coverage = len(unique_items) / catalog_size if catalog_size else 0.0

    total_positive = sum(history.popularity_counts.values())
    alpha = 1.0
    denominator = total_positive + alpha * max(catalog_size, 1)
    novelty_values = [
        -np.log2((history.popularity_counts.get(item, 0) + alpha) / denominator)
        for item in flattened
    ]
    popularity_values = [history.popularity_counts.get(item, 0) for item in flattened]
    return {
        "catalog_size": int(catalog_size),
        "unique_recommended_items": len(unique_items),
        "catalog_coverage": float(coverage),
        "mean_train_positive_popularity": float(np.mean(popularity_values))
        if popularity_values
        else 0.0,
        "mean_novelty_bits": float(np.mean(novelty_values)) if novelty_values else 0.0,
    }


def evaluate_one_model(
    model: object,
    model_name: str,
    ground_truth: dict[int, set[int]],
    history: FittedHistory,
    ks: tuple[int, ...],
    report_dir: Path,
    sample_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    max_k = max(ks)
    recommendations: dict[int, list[int]] = {}
    for user_id in ground_truth:
        recommendations[user_id] = model.recommend(user_id, max_k)  # type: ignore[attr-defined]

    rows: list[dict[str, Any]] = []
    for k in ks:
        user_rows: list[dict[str, Any]] = []
        for user_id, relevant in ground_truth.items():
            recs = recommendations[user_id][:k]
            metrics = ranking_metrics(recs, relevant, k)
            user_rows.append(
                {
                    "user_id": user_id,
                    "k": k,
                    **metrics.as_dict(),
                }
            )
        user_frame = pd.DataFrame(user_rows)
        aggregate = _aggregate_user_metrics(user_frame)
        catalog = _catalog_metrics(
            [recommendations[user_id][:k] for user_id in ground_truth], history
        )
        rows.append({"model": model_name, "k": k, **aggregate, **catalog})

        if k == 10:
            user_frame.to_csv(
                report_dir / f"per_user_{model_name}_k10.csv.gz",
                index=False,
                compression="gzip",
            )

    for user_id in sorted(ground_truth)[:5]:
        sample_rows.append(
            {
                "model": model_name,
                "user_id": user_id,
                "relevant_items": "|".join(str(x) for x in sorted(ground_truth[user_id])),
                "top_10": "|".join(str(x) for x in recommendations[user_id][:10]),
            }
        )
    return rows


def evaluate_m2_baselines(settings: Settings) -> dict[str, Any]:
    """Evaluate M2 baselines on validation only; the test split remains sealed."""
    train = load_split(settings, "train")
    validation = load_split(settings, "validation")
    movies = _load_movies(settings)

    evaluation_cfg = settings.evaluation
    ks = tuple(evaluation_cfg.ks if evaluation_cfg is not None else (5, 10, 20))
    primary_k = evaluation_cfg.primary_k if evaluation_cfg is not None else 10
    if primary_k not in ks:
        raise ValueError("evaluation.primary_k must be included in evaluation.ks")

    ground_truth, cohort = build_ground_truth(train, validation)
    if not ground_truth:
        raise ValueError("No warm-user/warm-item positive validation targets were found")

    history, recommenders = build_recommenders(train, movies=movies, seed=settings.seed)
    report_dir = settings.paths.reports / "m2"
    report_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = settings.paths.artifacts / "m2"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    metric_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    for model_name, model in recommenders.items():
        metric_rows.extend(
            evaluate_one_model(
                model=model,
                model_name=model_name,
                ground_truth=ground_truth,
                history=history,
                ks=ks,
                report_dir=report_dir,
                sample_rows=sample_rows,
            )
        )

    metrics = pd.DataFrame(metric_rows).sort_values(
        ["k", "ndcg", "recall"], ascending=[True, False, False]
    )
    metrics.to_csv(report_dir / "validation_metrics.csv", index=False)
    (report_dir / "validation_metrics.json").write_text(
        json.dumps(metric_rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    pd.DataFrame(sample_rows).to_csv(report_dir / "sample_recommendations.csv", index=False)

    primary = metrics.loc[metrics["k"].eq(primary_k)].copy().sort_values("ndcg", ascending=False)
    primary.to_csv(report_dir / f"model_comparison_at_{primary_k}.csv", index=False)

    plot_metrics = primary.set_index("model")[["recall", "ndcg", "map", "hit_rate"]]
    plot_metrics.plot(kind="bar")
    plt.title(f"M2 validation ranking metrics @ {primary_k}")
    plt.ylabel("Mean user metric")
    plt.xlabel("Baseline")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(report_dir / f"model_comparison_at_{primary_k}.png", dpi=160, bbox_inches="tight")
    plt.close()

    cohort_path = report_dir / "validation_cohort.json"
    cohort_path.write_text(json.dumps(cohort, indent=2) + "\n", encoding="utf-8")

    manifest = {
        "milestone": "M2",
        "evaluation_split": "validation",
        "test_split_accessed": False,
        "primary_segment": "warm_user_warm_item",
        "positive_threshold": settings.data.implicit_positive_threshold,
        "ks": list(ks),
        "primary_k": primary_k,
        "baselines": list(recommenders),
        "train_rows": len(train),
        "fit_catalog_items": len(history.catalog),
        "fit_users": int(train["user_id"].nunique()),
        "validation_cohort": cohort,
        "best_by_ndcg_at_primary_k": str(primary.iloc[0]["model"]),
        "generated_reports": sorted(p.name for p in report_dir.iterdir() if p.is_file()),
    }
    (artifact_dir / "baseline_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    primary_records = primary.to_dict(orient="records")
    return {"manifest": manifest, "primary_metrics": primary_records}
