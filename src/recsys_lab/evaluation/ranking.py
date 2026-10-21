from __future__ import annotations

import json
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from recsys_lab.config import Settings
from recsys_lab.evaluation.hybrid import run_m4_hybrid
from recsys_lab.evaluation.metrics import ranking_metrics
from recsys_lab.evaluation.offline import build_ground_truth
from recsys_lab.evaluation.split import load_split
from recsys_lab.models.collaborative import fit_puresvd
from recsys_lab.models.hybrid import HybridSVDMetadataRecommender
from recsys_lab.models.ranking import CandidateFeatureBuilder, PairwiseLinearRanker


def _load_movies(settings: Settings) -> pd.DataFrame:
    path = settings.paths.processed / "movies.csv.gz"
    if not path.exists():
        raise FileNotFoundError(f"Processed movies missing at {path}. Run `prepare` first.")
    return pd.read_csv(path)


def _read_or_build_m4(settings: Settings) -> dict[str, Any]:
    path = settings.paths.artifacts / "m4" / "m4_manifest.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return run_m4_hybrid(settings)


def split_ranker_validation(
    validation: pd.DataFrame,
    train_fraction: float,
    tune_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("rank_train_fraction must be in (0, 1)")
    if not 0.0 < tune_fraction < 1.0:
        raise ValueError("rank_tune_fraction must be in (0, 1)")
    if train_fraction + tune_fraction >= 1.0:
        raise ValueError("rank_train_fraction + rank_tune_fraction must be < 1")
    ordered = validation.sort_values(["timestamp", "user_id", "item_id"]).reset_index(drop=True)
    n = len(ordered)
    i = int(n * train_fraction)
    j = int(n * (train_fraction + tune_fraction))
    if i <= 0 or j <= i or j >= n:
        raise ValueError("Ranker validation windows are empty; adjust ranking fractions")
    frames = ordered.iloc[:i].copy(), ordered.iloc[i:j].copy(), ordered.iloc[j:].copy()
    rank_train, rank_tune, rank_eval = frames
    strict = (
        rank_train["timestamp"].max() <= rank_tune["timestamp"].min()
        and rank_tune["timestamp"].max() <= rank_eval["timestamp"].min()
    )
    manifest = {
        "protocol": "chronological_partition_inside_validation",
        "rank_train_rows": len(rank_train),
        "rank_tune_rows": len(rank_tune),
        "rank_eval_rows": len(rank_eval),
        "rank_train_end": rank_train["timestamp"].max().isoformat(),
        "rank_tune_end": rank_tune["timestamp"].max().isoformat(),
        "rank_eval_end": rank_eval["timestamp"].max().isoformat(),
        "chronological_order_check": bool(strict),
    }
    return rank_train, rank_tune, rank_eval, manifest


def _build_frozen_m4(settings: Settings, train: pd.DataFrame, movies: pd.DataFrame, m4: dict[str, Any]) -> HybridSVDMetadataRecommender:
    if settings.collaborative is None or settings.hybrid is None:
        raise ValueError("M5 requires collaborative and hybrid configuration")
    rank = int(m4["m3_fixed"]["rank"])
    popularity_blend = float(m4["m3_fixed"]["popularity_blend"])
    metadata_blend = float(m4["selected"]["metadata_blend"])
    # A single fit recreates the frozen M4 train snapshot. M5 never retunes retrieval.
    decomposition = fit_puresvd(train, max_rank=rank, seed=settings.seed)
    return HybridSVDMetadataRecommender(
        decomposition=decomposition,
        movies=movies,
        rank=rank,
        popularity_blend=popularity_blend,
        metadata_blend=metadata_blend,
        sparse_positive_threshold=settings.hybrid.sparse_positive_threshold,
        sparse_metadata_multiplier=settings.hybrid.sparse_metadata_multiplier,
        cold_item_penalty=settings.hybrid.cold_item_penalty,
    )


def _candidate_labeled_frame(
    model: HybridSVDMetadataRecommender,
    feature_builder: CandidateFeatureBuilder,
    ground_truth: dict[int, set[int]],
    candidate_k: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frames: list[pd.DataFrame] = []
    users_with_hit = 0
    relevant_total = 0
    relevant_retrieved = 0
    for user_id, relevant in ground_truth.items():
        candidates = model.recommend(user_id, candidate_k)
        feat = feature_builder.build(user_id, candidates)
        if feat.empty:
            continue
        feat["label"] = feat["item_id"].isin(relevant).astype(np.int8)
        hit_count = int(feat["label"].sum())
        if hit_count:
            users_with_hit += 1
        relevant_total += len(relevant)
        relevant_retrieved += hit_count
        frames.append(feat)
    if not frames:
        return pd.DataFrame(), {
            "users": len(ground_truth),
            "users_with_candidate_hit": 0,
            "candidate_positive_recall": 0.0,
            "rows": 0,
        }
    frame = pd.concat(frames, ignore_index=True)
    diagnostics = {
        "users": len(ground_truth),
        "users_with_candidate_hit": int(users_with_hit),
        "candidate_positive_recall": float(relevant_retrieved / relevant_total) if relevant_total else 0.0,
        "rows": len(frame),
        "positive_rows": int(frame["label"].sum()),
    }
    return frame, diagnostics


def _evaluate_orderings(
    candidate_frame: pd.DataFrame,
    ground_truth: dict[int, set[int]],
    ranker: PairwiseLinearRanker | None,
    blend: float,
    k: int,
) -> dict[str, float | int]:
    rows: list[dict[str, float]] = []
    for user_id, relevant in ground_truth.items():
        frame = candidate_frame.loc[candidate_frame["user_id"].eq(user_id)].copy()
        if frame.empty:
            recs: list[int] = []
        elif ranker is None or blend == 0.0:
            recs = [int(x) for x in frame.sort_values("base_rank")["item_id"]]
        else:
            recs = ranker.rerank(frame, blend=blend)
        metric = ranking_metrics(recs[:k], relevant, k)
        rows.append(metric.as_dict())
    if not rows:
        return {"users": 0, "recall": 0.0, "ndcg": 0.0, "map": 0.0, "hit_rate": 0.0, "mrr": 0.0, "precision": 0.0}
    df = pd.DataFrame(rows)
    return {
        "users": len(df),
        "recall": float(df["recall"].mean()),
        "precision": float(df["precision"].mean()),
        "ndcg": float(df["ndcg"].mean()),
        "map": float(df["map"].mean()),
        "hit_rate": float(df["hit_rate"].mean()),
        "mrr": float(df["mrr"].mean()),
    }


def _combine_ground_truth(a: dict[int, set[int]], b: dict[int, set[int]]) -> dict[int, set[int]]:
    out: dict[int, set[int]] = {u: set(items) for u, items in a.items()}
    for user_id, items in b.items():
        out.setdefault(user_id, set()).update(items)
    return out


def run_m5_ranking(settings: Settings) -> dict[str, Any]:
    """Train a pairwise LTR second stage over frozen M4 candidates; test remains sealed."""
    if settings.ranking is None or settings.evaluation is None:
        raise ValueError("M5 requires [ranking] and [evaluation] configuration")
    cfg = settings.ranking
    train = load_split(settings, "train")
    validation = load_split(settings, "validation")
    movies = _load_movies(settings)
    m4 = _read_or_build_m4(settings)
    model = _build_frozen_m4(settings, train, movies, m4)
    feature_builder = CandidateFeatureBuilder(model)

    rank_train, rank_tune, rank_eval, window_manifest = split_ranker_validation(
        validation, cfg.rank_train_fraction, cfg.rank_tune_fraction
    )
    gt_train, diag_train_gt = build_ground_truth(train, rank_train)
    gt_tune, diag_tune_gt = build_ground_truth(train, rank_tune)
    gt_eval, diag_eval_gt = build_ground_truth(train, rank_eval)
    if not gt_train or not gt_tune or not gt_eval:
        raise ValueError("M5 ranker windows need warm positive targets in train/tune/eval")

    train_candidates, train_diag = _candidate_labeled_frame(model, feature_builder, gt_train, cfg.candidate_k)
    tune_candidates, tune_diag = _candidate_labeled_frame(model, feature_builder, gt_tune, cfg.candidate_k)
    eval_candidates, eval_diag = _candidate_labeled_frame(model, feature_builder, gt_eval, cfg.candidate_k)

    report_dir = settings.paths.reports / "m5"
    artifact_dir = settings.paths.artifacts / "m5"
    report_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    tuning_rows: list[dict[str, Any]] = []
    fitted_by_c: dict[float, PairwiseLinearRanker] = {}
    pair_diag_by_c: dict[float, dict[str, int]] = {}
    for c in cfg.c_values:
        ranker = PairwiseLinearRanker(c=c, seed=settings.seed)
        pair_diag = ranker.fit(train_candidates, cfg.negatives_per_positive)
        fitted_by_c[c] = ranker
        pair_diag_by_c[c] = pair_diag
        for blend in cfg.ltr_blends:
            metrics = _evaluate_orderings(tune_candidates, gt_tune, ranker, blend, cfg.primary_k)
            tuning_rows.append(
                {
                    "c": float(c),
                    "ltr_blend": float(blend),
                    f"ndcg@{cfg.primary_k}": metrics["ndcg"],
                    f"recall@{cfg.primary_k}": metrics["recall"],
                    f"map@{cfg.primary_k}": metrics["map"],
                    f"hit_rate@{cfg.primary_k}": metrics["hit_rate"],
                }
            )
    tuning = pd.DataFrame(tuning_rows)
    ndcg_key = f"ndcg@{cfg.primary_k}"
    recall_key = f"recall@{cfg.primary_k}"
    # Stable tie-breaking favors a smaller LTR perturbation, then stronger regularization (smaller C).
    tuning = tuning.sort_values([ndcg_key, recall_key, "ltr_blend", "c"], ascending=[False, False, True, True])
    tuning.to_csv(report_dir / "ranker_tuning.csv", index=False)
    selected = tuning.iloc[0]
    selected_c = float(selected["c"])
    selected_blend = float(selected["ltr_blend"])

    # Refit selected hyperparameter on the first 75% of validation labels, then report once on the
    # final chronological rank-eval window.
    combined_gt = _combine_ground_truth(gt_train, gt_tune)
    combined_candidates, combined_diag = _candidate_labeled_frame(
        model, feature_builder, combined_gt, cfg.candidate_k
    )
    final_ranker = PairwiseLinearRanker(c=selected_c, seed=settings.seed)
    final_pair_diag = final_ranker.fit(combined_candidates, cfg.negatives_per_positive)

    baseline_eval = _evaluate_orderings(eval_candidates, gt_eval, None, 0.0, cfg.primary_k)
    ltr_eval = _evaluate_orderings(eval_candidates, gt_eval, final_ranker, selected_blend, cfg.primary_k)
    comparison = pd.DataFrame(
        [
            {"model": "m4_candidate_order", **baseline_eval},
            {"model": "m5_pairwise_ltr", **ltr_eval},
        ]
    )
    comparison.to_csv(report_dir / "m4_vs_m5_rank_eval.csv", index=False)
    comparison.set_index("model")[["ndcg", "recall", "map", "hit_rate"]].plot(kind="bar")
    plt.title(f"M4 candidate order vs M5 LTR on rank-eval @ {cfg.primary_k}")
    plt.ylabel("Mean user metric")
    plt.xlabel("Ordering")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(report_dir / "m4_vs_m5_rank_eval.png", dpi=160, bbox_inches="tight")
    plt.close()

    importance = final_ranker.feature_importance()
    importance.to_csv(report_dir / "feature_importance.csv", index=False)
    importance.head(10).sort_values("abs_coefficient").plot.barh(x="feature", y="abs_coefficient", legend=False)
    plt.title("M5 pairwise LTR coefficient magnitude")
    plt.xlabel("|standardized coefficient|")
    plt.tight_layout()
    plt.savefig(report_dir / "feature_importance.png", dpi=160, bbox_inches="tight")
    plt.close()

    # Per-user audit + small human-readable sample.
    per_user_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    for user_id, relevant in gt_eval.items():
        frame = eval_candidates.loc[eval_candidates["user_id"].eq(user_id)].copy()
        base_recs = [int(x) for x in frame.sort_values("base_rank")["item_id"]]
        ltr_recs = final_ranker.rerank(frame, selected_blend) if not frame.empty else []
        base_m = ranking_metrics(base_recs[: cfg.primary_k], relevant, cfg.primary_k)
        ltr_m = ranking_metrics(ltr_recs[: cfg.primary_k], relevant, cfg.primary_k)
        per_user_rows.append(
            {
                "user_id": int(user_id),
                "base_ndcg": base_m.ndcg,
                "ltr_ndcg": ltr_m.ndcg,
                "ndcg_delta": ltr_m.ndcg - base_m.ndcg,
                "base_recall": base_m.recall,
                "ltr_recall": ltr_m.recall,
            }
        )
        if len(sample_rows) < 8:
            sample_rows.append(
                {
                    "user_id": int(user_id),
                    "relevant_items": "|".join(str(x) for x in sorted(relevant)),
                    "m4_top10": "|".join(str(x) for x in base_recs[:10]),
                    "m5_top10": "|".join(str(x) for x in ltr_recs[:10]),
                }
            )
    pd.DataFrame(per_user_rows).to_csv(report_dir / "per_user_rank_eval.csv.gz", index=False, compression="gzip")
    pd.DataFrame(sample_rows).to_csv(report_dir / "sample_rerankings.csv", index=False)

    final_ranker.save(artifact_dir / "pairwise_ltr_selected.npz", blend=selected_blend)
    model_metadata = {
        "model_type": "PairwiseLinearRanker",
        "candidate_source": "frozen M4 HybridSVDMetadataRecommender",
        "candidate_k": cfg.candidate_k,
        "feature_names": list(feature_builder.feature_names),
        "selected_c": selected_c,
        "selected_ltr_blend": selected_blend,
        "fit_labels": "rank_train + rank_tune (chronological first 75% of validation)",
        "rank_eval_labels": "chronological final 25% of validation",
        "test_split_accessed": False,
    }
    (artifact_dir / "model_metadata.json").write_text(json.dumps(model_metadata, indent=2) + "\n", encoding="utf-8")

    base_ndcg = float(baseline_eval["ndcg"])
    ltr_ndcg = float(ltr_eval["ndcg"])
    manifest: dict[str, Any] = {
        "milestone": "M5",
        "candidate_fit_split": "train",
        "ranker_development_split": "validation_only",
        "test_split_accessed": False,
        "candidate_model_frozen_from": "M4",
        "candidate_k": cfg.candidate_k,
        "primary_k": cfg.primary_k,
        "ranker_windows": window_manifest,
        "ground_truth_diagnostics": {
            "rank_train": diag_train_gt,
            "rank_tune": diag_tune_gt,
            "rank_eval": diag_eval_gt,
        },
        "candidate_diagnostics": {
            "rank_train": train_diag,
            "rank_tune": tune_diag,
            "rank_eval": eval_diag,
            "rank_train_plus_tune": combined_diag,
        },
        "selected": {
            "c": selected_c,
            "ltr_blend": selected_blend,
            f"rank_tune_ndcg@{cfg.primary_k}": float(selected[ndcg_key]),
            f"rank_tune_recall@{cfg.primary_k}": float(selected[recall_key]),
        },
        "rank_eval": {
            "m4_candidate_order": baseline_eval,
            "m5_pairwise_ltr": ltr_eval,
            "ndcg_lift_vs_m4_pct": 100.0 * (ltr_ndcg - base_ndcg) / base_ndcg if base_ndcg else 0.0,
            "candidate_recall_unchanged_by_construction": True,
        },
        "pairwise_training": final_pair_diag,
        "generated_reports": sorted(p.name for p in report_dir.iterdir() if p.is_file()),
    }
    (artifact_dir / "m5_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest
