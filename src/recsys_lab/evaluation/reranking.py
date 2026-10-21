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
from recsys_lab.evaluation.offline import build_ground_truth
from recsys_lab.evaluation.ranking import (
    _build_frozen_m4,
    _candidate_labeled_frame,
    _load_movies,
    _read_or_build_m4,
    split_ranker_validation,
)
from recsys_lab.evaluation.split import load_split
from recsys_lab.models.ranking import FEATURE_NAMES, CandidateFeatureBuilder
from recsys_lab.models.reranking import DiversityNoveltyReranker, intra_list_diversity, novelty_bits


def _read_or_build_m5(settings: Settings) -> dict[str, Any]:
    path = settings.paths.artifacts / "m5" / "m5_manifest.json"
    artifact = settings.paths.artifacts / "m5" / "pairwise_ltr_selected.npz"
    if path.exists() and artifact.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    from recsys_lab.evaluation.ranking import run_m5_ranking

    return run_m5_ranking(settings)


def _split_rerank_window(frame: pd.DataFrame, tune_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if not 0.0 < tune_fraction < 1.0:
        raise ValueError("reranking.tune_fraction must be in (0, 1)")
    ordered = frame.sort_values(["timestamp", "user_id", "item_id"]).reset_index(drop=True)
    cut = int(len(ordered) * tune_fraction)
    if cut <= 0 or cut >= len(ordered):
        raise ValueError("M6 rerank tune/eval windows are empty")
    tune, evaluation = ordered.iloc[:cut].copy(), ordered.iloc[cut:].copy()
    strict = tune["timestamp"].max() <= evaluation["timestamp"].min()
    manifest = {
        "protocol": "chronological_partition_inside_m5_rank_eval",
        "rerank_tune_rows": len(tune),
        "rerank_eval_rows": len(evaluation),
        "rerank_tune_end": tune["timestamp"].max().isoformat(),
        "rerank_eval_end": evaluation["timestamp"].max().isoformat(),
        "chronological_order_check": bool(strict),
    }
    return tune, evaluation, manifest


def _load_m5_orderer(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"M5 serving artifact missing at {path}. Run `python -m recsys_lab m5` first.")
    saved = np.load(path)
    return {
        "feature_names": tuple(str(x) for x in saved["feature_names"].tolist()),
        "mean": np.asarray(saved["scaler_mean"], dtype=np.float64),
        "scale": np.asarray(saved["scaler_scale"], dtype=np.float64),
        "coef": np.asarray(saved["coefficients"], dtype=np.float64),
        "blend": float(saved["blend"][0]),
    }


def _z(x: np.ndarray) -> np.ndarray:
    std = float(x.std())
    return np.zeros_like(x) if std <= 1e-12 else (x - float(x.mean())) / std


def _m5_order(frame: pd.DataFrame, orderer: dict[str, Any]) -> list[int]:
    if frame.empty:
        return []
    baseline = frame.sort_values("base_rank").copy()
    blend = float(orderer["blend"])
    if blend == 0.0:
        return [int(x) for x in baseline["item_id"]]
    feature_names = orderer["feature_names"]
    if tuple(feature_names) != FEATURE_NAMES:
        raise ValueError("M5 artifact feature contract does not match current code")
    x = baseline.loc[:, feature_names].to_numpy(dtype=np.float64)
    scale = np.where(np.asarray(orderer["scale"]) == 0, 1.0, np.asarray(orderer["scale"]))
    ltr = ((x - np.asarray(orderer["mean"])) / scale) @ np.asarray(orderer["coef"])
    base = -baseline["base_rank"].to_numpy(dtype=np.float64)
    score = (1.0 - blend) * _z(base) + blend * _z(ltr)
    tmp = baseline[["item_id", "base_rank"]].copy()
    tmp["score"] = score
    tmp = tmp.sort_values(["score", "base_rank", "item_id"], ascending=[False, True, True])
    return [int(x) for x in tmp["item_id"]]


def _evaluate(
    candidates: pd.DataFrame,
    ground_truth: dict[int, set[int]],
    orderer: dict[str, Any],
    reranker: DiversityNoveltyReranker | None,
    metadata: Any,
    popularity_counts: dict[int, int],
    catalog_size: int,
    k: int,
) -> tuple[dict[str, float | int], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    all_recs: list[list[int]] = []
    for user_id, relevant in ground_truth.items():
        frame = candidates.loc[candidates["user_id"].eq(user_id)].copy()
        base_order = _m5_order(frame, orderer)
        recs = base_order if reranker is None else reranker.rerank(base_order, k)
        top = recs[:k]
        all_recs.append(top)
        m = ranking_metrics(top, relevant, k)
        rows.append(
            {
                "user_id": int(user_id),
                **m.as_dict(),
                "intra_list_diversity": intra_list_diversity(metadata, top),
                "novelty_bits": novelty_bits(top, popularity_counts, catalog_size),
            }
        )
    if not rows:
        return {
            "users": 0, "recall": 0.0, "precision": 0.0, "ndcg": 0.0, "map": 0.0,
            "hit_rate": 0.0, "mrr": 0.0, "intra_list_diversity": 0.0,
            "mean_novelty_bits": 0.0, "catalog_coverage": 0.0,
        }, []
    df = pd.DataFrame(rows)
    unique = {item for recs in all_recs for item in recs}
    catalog_size = max(int(catalog_size), 1)
    return {
        "users": len(df),
        "recall": float(df["recall"].mean()),
        "precision": float(df["precision"].mean()),
        "ndcg": float(df["ndcg"].mean()),
        "map": float(df["map"].mean()),
        "hit_rate": float(df["hit_rate"].mean()),
        "mrr": float(df["mrr"].mean()),
        "intra_list_diversity": float(df["intra_list_diversity"].mean()),
        "mean_novelty_bits": float(df["novelty_bits"].mean()),
        "catalog_coverage": float(len(unique) / catalog_size),
        "unique_recommended_items": len(unique),
    }, rows


def _passes_quality_guardrails(
    candidate: dict[str, float | int],
    baseline: dict[str, float | int],
    ndcg_ratio: float,
    recall_ratio: float,
) -> bool:
    return bool(
        float(candidate["ndcg"]) >= float(baseline["ndcg"]) * ndcg_ratio
        and float(candidate["recall"]) >= float(baseline["recall"]) * recall_ratio
    )


def run_m6_reranking(settings: Settings) -> dict[str, Any]:
    """Tune diversity/novelty re-ranking downstream of frozen M5; global test remains sealed."""
    if settings.reranking is None or settings.ranking is None:
        raise ValueError("M6 requires [reranking] and [ranking] configuration")
    cfg = settings.reranking
    train = load_split(settings, "train")
    validation = load_split(settings, "validation")
    movies = _load_movies(settings)
    m4 = _read_or_build_m4(settings)
    m5 = _read_or_build_m5(settings)
    model = _build_frozen_m4(settings, train, movies, m4)
    feature_builder = CandidateFeatureBuilder(model)

    _, _, rank_eval, rank_windows = split_ranker_validation(
        validation, settings.ranking.rank_train_fraction, settings.ranking.rank_tune_fraction
    )
    rerank_tune, rerank_eval, rerank_windows = _split_rerank_window(rank_eval, cfg.tune_fraction)
    gt_tune, tune_gt_diag = build_ground_truth(train, rerank_tune)
    gt_eval, eval_gt_diag = build_ground_truth(train, rerank_eval)
    if not gt_tune or not gt_eval:
        raise ValueError("M6 needs warm positive targets in both rerank tune/eval windows")

    tune_candidates, tune_diag = _candidate_labeled_frame(
        model, feature_builder, gt_tune, settings.ranking.candidate_k
    )
    eval_candidates, eval_diag = _candidate_labeled_frame(
        model, feature_builder, gt_eval, settings.ranking.candidate_k
    )
    orderer = _load_m5_orderer(settings.paths.artifacts / "m5" / "pairwise_ltr_selected.npz")

    catalog_size = len(model.history.catalog)
    popularity_counts = dict(model.history.popularity_counts)

    baseline_tune, _ = _evaluate(
        tune_candidates, gt_tune, orderer, None, model.metadata, popularity_counts, catalog_size,
        cfg.primary_k
    )
    tuning_rows: list[dict[str, Any]] = []
    for dw in cfg.diversity_weights:
        for nw in cfg.novelty_weights:
            reranker = DiversityNoveltyReranker(
                metadata=model.metadata,
                popularity_counts=model.history.popularity_counts,
                diversity_weight=dw,
                novelty_weight=nw,
                pool_k=cfg.pool_k,
                max_per_genre=cfg.max_per_genre,
            )
            metrics, _ = _evaluate(
                tune_candidates, gt_tune, orderer, reranker, model.metadata, popularity_counts,
                catalog_size, cfg.primary_k
            )
            passes = _passes_quality_guardrails(
                metrics, baseline_tune, cfg.ndcg_guardrail_ratio, cfg.recall_guardrail_ratio
            )
            tuning_rows.append(
                {
                    "diversity_weight": float(dw),
                    "novelty_weight": float(nw),
                    f"ndcg@{cfg.primary_k}": metrics["ndcg"],
                    f"recall@{cfg.primary_k}": metrics["recall"],
                    "intra_list_diversity": metrics["intra_list_diversity"],
                    "mean_novelty_bits": metrics["mean_novelty_bits"],
                    "catalog_coverage": metrics["catalog_coverage"],
                    "passes_guardrails": bool(passes),
                }
            )

    tuning = pd.DataFrame(tuning_rows)
    ndcg_col, recall_col = f"ndcg@{cfg.primary_k}", f"recall@{cfg.primary_k}"
    eligible = tuning.loc[tuning["passes_guardrails"]].copy()
    if eligible.empty:
        raise RuntimeError("M6 no-regression anchor unexpectedly failed its own guardrails")
    # Business objective after quality guardrails: maximize genre diversity, then novelty and coverage.
    eligible = eligible.sort_values(
        ["intra_list_diversity", "mean_novelty_bits", "catalog_coverage", ndcg_col, recall_col,
         "diversity_weight", "novelty_weight"],
        ascending=[False, False, False, False, False, True, True],
    )
    selected = eligible.iloc[0]
    selected_dw = float(selected["diversity_weight"])
    selected_nw = float(selected["novelty_weight"])
    selected_reranker = DiversityNoveltyReranker(
        metadata=model.metadata,
        popularity_counts=model.history.popularity_counts,
        diversity_weight=selected_dw,
        novelty_weight=selected_nw,
        pool_k=cfg.pool_k,
        max_per_genre=cfg.max_per_genre,
    )

    baseline_eval, baseline_rows = _evaluate(
        eval_candidates, gt_eval, orderer, None, model.metadata, popularity_counts, catalog_size,
        cfg.primary_k
    )
    proposed_eval, proposed_rows = _evaluate(
        eval_candidates, gt_eval, orderer, selected_reranker, model.metadata, popularity_counts,
        catalog_size, cfg.primary_k
    )
    deployment_gate_passed = _passes_quality_guardrails(
        proposed_eval, baseline_eval, cfg.ndcg_guardrail_ratio, cfg.recall_guardrail_ratio
    )
    if deployment_gate_passed:
        deployed_dw, deployed_nw = selected_dw, selected_nw
        deployed_reranker = selected_reranker
        deployed_eval, deployed_rows = proposed_eval, proposed_rows
        deployment_action = "deploy_tune_selected_reranker"
    else:
        deployed_dw, deployed_nw = 0.0, 0.0
        deployed_reranker = DiversityNoveltyReranker(
            metadata=model.metadata,
            popularity_counts=model.history.popularity_counts,
            diversity_weight=0.0,
            novelty_weight=0.0,
            pool_k=cfg.pool_k,
            max_per_genre=cfg.max_per_genre,
        )
        deployed_eval, deployed_rows = baseline_eval, baseline_rows
        deployment_action = "rollback_to_exact_m5_anchor"

    report_dir = settings.paths.reports / "m6"
    artifact_dir = settings.paths.artifacts / "m6"
    report_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    tuning.sort_values(["passes_guardrails", "intra_list_diversity"], ascending=[False, False]).to_csv(
        report_dir / "reranker_tuning.csv", index=False
    )
    comparison = pd.DataFrame(
        [
            {"model": "m5_selected_order", **baseline_eval},
            {"model": "m6_tune_selected_candidate", **proposed_eval},
            {"model": "m6_deployed", **deployed_eval},
        ]
    )
    comparison.to_csv(report_dir / "m5_vs_m6_rerank_eval.csv", index=False)
    comparison.set_index("model")[["ndcg", "recall", "intra_list_diversity", "catalog_coverage"]].plot(kind="bar")
    plt.title(f"M5 vs M6 on rerank-eval @ {cfg.primary_k}")
    plt.ylabel("Metric")
    plt.xlabel("Ordering")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(report_dir / "m5_vs_m6_rerank_eval.png", dpi=160, bbox_inches="tight")
    plt.close()

    baseline_by_user = {int(row["user_id"]): row for row in baseline_rows}
    proposed_by_user = {int(row["user_id"]): row for row in proposed_rows}
    deployed_by_user = {int(row["user_id"]): row for row in deployed_rows}
    audit_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    for user_id, relevant in gt_eval.items():
        frame = eval_candidates.loc[eval_candidates["user_id"].eq(user_id)].copy()
        m5_order = _m5_order(frame, orderer)
        proposed_order = selected_reranker.rerank(m5_order, cfg.primary_k)
        deployed_order = deployed_reranker.rerank(m5_order, cfg.primary_k)
        b = baseline_by_user[int(user_id)]
        p = proposed_by_user[int(user_id)]
        d = deployed_by_user[int(user_id)]
        audit_rows.append(
            {
                "user_id": int(user_id),
                "m5_ndcg": b["ndcg"],
                "proposed_m6_ndcg": p["ndcg"],
                "deployed_m6_ndcg": d["ndcg"],
                "deployed_ndcg_delta": d["ndcg"] - b["ndcg"],
                "m5_diversity": b["intra_list_diversity"],
                "proposed_m6_diversity": p["intra_list_diversity"],
                "deployed_m6_diversity": d["intra_list_diversity"],
                "deployed_diversity_delta": d["intra_list_diversity"] - b["intra_list_diversity"],
                "m5_novelty_bits": b["novelty_bits"],
                "proposed_m6_novelty_bits": p["novelty_bits"],
                "deployed_m6_novelty_bits": d["novelty_bits"],
            }
        )
        if len(sample_rows) < 8:
            sample_rows.append(
                {
                    "user_id": int(user_id),
                    "relevant_items": "|".join(str(x) for x in sorted(relevant)),
                    "m5_top10": "|".join(str(x) for x in m5_order[: cfg.primary_k]),
                    "m6_proposed_top10": "|".join(str(x) for x in proposed_order[: cfg.primary_k]),
                    "m6_deployed_top10": "|".join(str(x) for x in deployed_order[: cfg.primary_k]),
                }
            )
    pd.DataFrame(audit_rows).to_csv(report_dir / "per_user_rerank_eval.csv.gz", index=False, compression="gzip")
    pd.DataFrame(sample_rows).to_csv(report_dir / "sample_rerankings.csv", index=False)

    artifact = {
        "model_type": "DiversityNoveltyReranker",
        "candidate_source": "M5 selected order over frozen M4 Top-100",
        "development_selected_diversity_weight": selected_dw,
        "development_selected_novelty_weight": selected_nw,
        "deployed_diversity_weight": deployed_dw,
        "deployed_novelty_weight": deployed_nw,
        "deployment_gate_passed": deployment_gate_passed,
        "deployment_action": deployment_action,
        "pool_k": cfg.pool_k,
        "primary_k": cfg.primary_k,
        "max_per_genre": cfg.max_per_genre,
        "m5_ltr_blend": float(m5["selected"]["ltr_blend"]),
        "test_split_accessed": False,
    }
    (artifact_dir / "reranker_selected.json").write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    base_ndcg = float(baseline_eval["ndcg"])
    new_ndcg = float(deployed_eval["ndcg"])
    base_div = float(baseline_eval["intra_list_diversity"])
    new_div = float(deployed_eval["intra_list_diversity"])
    manifest: dict[str, Any] = {
        "milestone": "M6",
        "candidate_fit_split": "train",
        "reranker_development_split": "validation_only",
        "test_split_accessed": False,
        "m5_selected_ltr_blend": float(m5["selected"]["ltr_blend"]),
        "candidate_k": settings.ranking.candidate_k,
        "primary_k": cfg.primary_k,
        "m5_rank_windows": rank_windows,
        "rerank_windows": rerank_windows,
        "ground_truth_diagnostics": {"rerank_tune": tune_gt_diag, "rerank_eval": eval_gt_diag},
        "candidate_diagnostics": {"rerank_tune": tune_diag, "rerank_eval": eval_diag},
        "selection_guardrails": {
            "ndcg_ratio": cfg.ndcg_guardrail_ratio,
            "recall_ratio": cfg.recall_guardrail_ratio,
            "no_regression_anchor": {"diversity_weight": 0.0, "novelty_weight": 0.0},
        },
        "development_selected": {
            "diversity_weight": selected_dw,
            "novelty_weight": selected_nw,
            f"rerank_tune_ndcg@{cfg.primary_k}": float(selected[ndcg_col]),
            f"rerank_tune_recall@{cfg.primary_k}": float(selected[recall_col]),
            "rerank_tune_intra_list_diversity": float(selected["intra_list_diversity"]),
            "rerank_tune_mean_novelty_bits": float(selected["mean_novelty_bits"]),
            "passes_guardrails": bool(selected["passes_guardrails"]),
        },
        "deployment_gate": {
            "passed": deployment_gate_passed,
            "action": deployment_action,
            "ndcg_ratio": cfg.ndcg_guardrail_ratio,
            "recall_ratio": cfg.recall_guardrail_ratio,
            "proposed_candidate": proposed_eval,
        },
        "deployed": {
            "diversity_weight": deployed_dw,
            "novelty_weight": deployed_nw,
        },
        "rerank_eval": {
            "m5_selected_order": baseline_eval,
            "m6_tune_selected_candidate": proposed_eval,
            "m6_deployed": deployed_eval,
            "ndcg_lift_vs_m5_pct": 100.0 * (new_ndcg - base_ndcg) / base_ndcg if base_ndcg else 0.0,
            "diversity_lift_vs_m5_pct": 100.0 * (new_div - base_div) / base_div if base_div else 0.0,
            "candidate_recall_unchanged_by_construction": True,
        },
        "generated_reports": sorted(p.name for p in report_dir.iterdir() if p.is_file()),
    }
    (artifact_dir / "m6_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest
