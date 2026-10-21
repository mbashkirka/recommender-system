from __future__ import annotations

import argparse
import json

from recsys_lab.config import load_settings
from recsys_lab.data.download import download_movielens
from recsys_lab.data.ingest import prepare_processed
from recsys_lab.data.profile import write_profile
from recsys_lab.evaluation.collaborative import run_m3_collaborative
from recsys_lab.evaluation.eda import build_m1_eda
from recsys_lab.evaluation.hybrid import run_m4_hybrid
from recsys_lab.evaluation.offline import evaluate_m2_baselines
from recsys_lab.evaluation.ranking import run_m5_ranking
from recsys_lab.evaluation.reranking import run_m6_reranking
from recsys_lab.evaluation.split import build_temporal_splits
from recsys_lab.experiments.assignment import assign_two_arm
from recsys_lab.experiments.pipeline import run_m7_ab_testing
from recsys_lab.portfolio import run_m10_portfolio
from recsys_lab.serving.bundle import build_serving_bundle
from recsys_lab.serving.engine import RecommendationEngine
from recsys_lab.serving.operations import run_m9_preflight


def _cmd_download(args: argparse.Namespace) -> None:
    settings = load_settings()
    path = download_movielens(settings, force=args.force)
    print(path)


def _cmd_prepare(_: argparse.Namespace) -> None:
    settings = load_settings()
    outputs = prepare_processed(settings)
    for name, path in outputs.items():
        print(f"{name}: {path}")


def _cmd_profile(_: argparse.Namespace) -> None:
    settings = load_settings()
    profile = write_profile(settings)
    print(json.dumps(profile, indent=2, ensure_ascii=False))
    if not all(profile["quality_checks"].values()):
        raise SystemExit("One or more M0 data quality checks failed.")


def _cmd_smoke(args: argparse.Namespace) -> None:
    settings = load_settings()
    download_movielens(settings, force=args.force)
    prepare_processed(settings)
    profile = write_profile(settings)
    if not all(profile["quality_checks"].values()):
        print(json.dumps(profile, indent=2, ensure_ascii=False))
        raise SystemExit("M0 smoke test failed: data quality checks did not all pass.")

    assignment = assign_two_arm(
        experiment=settings.experiment.default_name,
        unit_id="smoke-user-42",
        salt=settings.experiment.default_salt,
        control_weight=settings.experiment.control_weight,
    )
    print("M0 smoke test: PASS")
    print(
        json.dumps(
            {
                "interactions": profile["n_interactions"],
                "users": profile["n_users_in_ratings"],
                "items": profile["n_items_in_ratings"],
                "positive_rate": profile["positive_rate"],
                "ab_assignment_smoke": assignment.variant,
            },
            indent=2,
        )
    )


def _cmd_split(_: argparse.Namespace) -> None:
    settings = load_settings()
    manifest = build_temporal_splits(settings)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    if not all(manifest["leakage_checks"].values()):
        raise SystemExit("M1 split failed one or more leakage checks.")


def _cmd_eda(_: argparse.Namespace) -> None:
    settings = load_settings()
    summary = build_m1_eda(settings)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def _cmd_m1(_: argparse.Namespace) -> None:
    settings = load_settings()
    manifest = build_temporal_splits(settings)
    if not all(manifest["leakage_checks"].values()):
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
        raise SystemExit("M1 failed: leakage checks did not all pass.")
    eda = build_m1_eda(settings)
    print("M1 temporal split + EDA: PASS")
    print(
        json.dumps(
            {
                "train_rows": manifest["splits"]["train"]["rows"],
                "validation_rows": manifest["splits"]["validation"]["rows"],
                "test_rows": manifest["splits"]["test"]["rows"],
                "train_end": manifest["cutoffs"]["train_end"],
                "validation_end": manifest["cutoffs"]["validation_end"],
                "validation_warm_eval_rate": manifest["evaluation_segments"]["validation"][
                    "warm_user_warm_item_rate"
                ],
                "test_warm_eval_rate": manifest["evaluation_segments"]["test"][
                    "warm_user_warm_item_rate"
                ],
                "top_1pct_item_share": eda["top_1pct_item_share"],
            },
            indent=2,
        )
    )


def _cmd_m2(_: argparse.Namespace) -> None:
    settings = load_settings()
    result = evaluate_m2_baselines(settings)
    manifest = result["manifest"]
    primary_k = manifest["primary_k"]
    metrics = result["primary_metrics"]
    print("M2 offline baselines: PASS")
    print(
        json.dumps(
            {
                "evaluation_split": manifest["evaluation_split"],
                "test_split_accessed": manifest["test_split_accessed"],
                "primary_segment": manifest["primary_segment"],
                "evaluation_users": manifest["validation_cohort"][
                    "evaluation_users_with_warm_positive"
                ],
                "warm_positive_event_rate": manifest["validation_cohort"][
                    "warm_positive_event_rate"
                ],
                "candidate_catalog_items": manifest["fit_catalog_items"],
                "primary_k": primary_k,
                "best_by_ndcg": manifest["best_by_ndcg_at_primary_k"],
                "metrics_at_primary_k": [
                    {
                        "model": row["model"],
                        f"recall@{primary_k}": row["recall"],
                        f"ndcg@{primary_k}": row["ndcg"],
                        f"map@{primary_k}": row["map"],
                        f"hit_rate@{primary_k}": row["hit_rate"],
                        "catalog_coverage": row["catalog_coverage"],
                    }
                    for row in metrics
                ],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def _cmd_m3(_: argparse.Namespace) -> None:
    settings = load_settings()
    manifest = run_m3_collaborative(settings)
    selected = manifest["selected"]
    print("M3 collaborative retrieval: PASS")
    print(
        json.dumps(
            {
                "fit_split": manifest["fit_split"],
                "tuning_split": manifest["tuning_split"],
                "test_split_accessed": manifest["test_split_accessed"],
                "evaluation_users": manifest["evaluation_users"],
                "selected_rank": selected["rank"],
                "selected_popularity_blend": selected["popularity_blend"],
                "m2_popularity": manifest["m2_popularity"],
                "m3_selected": manifest["m3_selected"],
                "ndcg_lift_vs_popularity_pct": manifest["ndcg_lift_vs_popularity_pct"],
                "candidate_recall_lift_vs_popularity_pct": manifest[
                    "candidate_recall_lift_vs_popularity_pct"
                ],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def _cmd_m4(_: argparse.Namespace) -> None:
    settings = load_settings()
    manifest = run_m4_hybrid(settings)
    selected = manifest["selected"]
    print("M4 hybrid retrieval + cold start: PASS")
    print(
        json.dumps(
            {
                "fit_split": manifest["fit_split"],
                "tuning_split": manifest["tuning_split"],
                "test_split_accessed": manifest["test_split_accessed"],
                "selected_metadata_blend": selected["metadata_blend"],
                "m3_fixed": manifest["m3_fixed"],
                "m4_selected": selected,
                "candidate_recall_lift_vs_m3_pct": manifest[
                    "candidate_recall_lift_vs_m3_pct"
                ],
                "ndcg_lift_vs_m3_pct": manifest["ndcg_lift_vs_m3_pct"],
                "cold_start": manifest["cold_start"],
                "routing": manifest["routing"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def _cmd_m5(_: argparse.Namespace) -> None:
    settings = load_settings()
    manifest = run_m5_ranking(settings)
    primary_k = manifest["primary_k"]
    print("M5 learning-to-rank: PASS")
    print(
        json.dumps(
            {
                "candidate_fit_split": manifest["candidate_fit_split"],
                "ranker_development_split": manifest["ranker_development_split"],
                "test_split_accessed": manifest["test_split_accessed"],
                "candidate_k": manifest["candidate_k"],
                "primary_k": primary_k,
                "ranker_windows": manifest["ranker_windows"],
                "selected": manifest["selected"],
                "m4_candidate_order": manifest["rank_eval"]["m4_candidate_order"],
                "m5_pairwise_ltr": manifest["rank_eval"]["m5_pairwise_ltr"],
                "ndcg_lift_vs_m4_pct": manifest["rank_eval"]["ndcg_lift_vs_m4_pct"],
                "candidate_recall_unchanged_by_construction": manifest["rank_eval"][
                    "candidate_recall_unchanged_by_construction"
                ],
                "pairwise_training": manifest["pairwise_training"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def _cmd_m6(_: argparse.Namespace) -> None:
    settings = load_settings()
    manifest = run_m6_reranking(settings)
    print("M6 diversity/novelty re-ranking: PASS")
    print(
        json.dumps(
            {
                "candidate_fit_split": manifest["candidate_fit_split"],
                "reranker_development_split": manifest["reranker_development_split"],
                "test_split_accessed": manifest["test_split_accessed"],
                "m5_selected_ltr_blend": manifest["m5_selected_ltr_blend"],
                "candidate_k": manifest["candidate_k"],
                "primary_k": manifest["primary_k"],
                "rerank_windows": manifest["rerank_windows"],
                "development_selected": manifest["development_selected"],
                "deployment_gate": manifest["deployment_gate"],
                "deployed": manifest["deployed"],
                "m5_selected_order": manifest["rerank_eval"]["m5_selected_order"],
                "m6_deployed": manifest["rerank_eval"]["m6_deployed"],
                "ndcg_lift_vs_m5_pct": manifest["rerank_eval"]["ndcg_lift_vs_m5_pct"],
                "diversity_lift_vs_m5_pct": manifest["rerank_eval"]["diversity_lift_vs_m5_pct"],
                "candidate_recall_unchanged_by_construction": manifest["rerank_eval"][
                    "candidate_recall_unchanged_by_construction"
                ],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def _cmd_m7(_: argparse.Namespace) -> None:
    settings = load_settings()
    manifest = run_m7_ab_testing(settings)
    analysis = manifest["analysis"]
    print("M7 A/B experimentation: PASS")
    print(
        json.dumps(
            {
                "experiment": manifest["experiment"],
                "demo_only": manifest["demo_only"],
                "test_split_accessed": manifest["test_split_accessed"],
                "event_validation_passed": manifest["event_validation_passed"],
                "serving_state": manifest["serving_state"],
                "srm": analysis["srm"],
                "primary": analysis["primary"],
                "cuped_clicks_per_user": analysis["cuped_clicks_per_user"],
                "power": analysis["power"],
                "guardrails": analysis["guardrails"],
                "decision": analysis["decision"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )



def _cmd_m8(_: argparse.Namespace) -> None:
    settings = load_settings()
    manifest = build_serving_bundle(settings)
    engine = RecommendationEngine(settings.paths.artifacts / "m8" / "serving_bundle.npz")
    known_user = int(engine.user_ids[0]) if len(engine.user_ids) else 1
    warm_recs, warm_meta = engine.recommend(known_user, min(10, engine.primary_k), policy="control")
    cold_user = int(max(engine.user_ids) + 1) if len(engine.user_ids) else 999999
    cold_recs, cold_meta = engine.recommend(cold_user, min(10, engine.primary_k), policy="control")
    if not warm_recs or not cold_recs:
        raise SystemExit("M8 preflight failed: serving engine returned an empty recommendation list")
    report = {
        "milestone": "M8",
        "status": "PASS",
        "artifact_fingerprint": engine.fingerprint,
        "validation_split_accessed": False,
        "test_split_accessed": False,
        "catalog_items": len(engine.all_item_ids),
        "known_user_smoke": {"user_id": known_user, "route": warm_meta["route"], "items": [x.item_id for x in warm_recs]},
        "cold_user_smoke": {"user_id": cold_user, "route": cold_meta["route"], "items": [x.item_id for x in cold_recs]},
        "experiment": manifest["experiment"],
        "model_policy": manifest["model_policy"],
    }
    report_dir = settings.paths.reports / "m8"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "preflight.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("M8 FastAPI serving preflight: PASS")
    print(json.dumps(report, indent=2, ensure_ascii=False))



def _cmd_m9(_: argparse.Namespace) -> None:
    settings = load_settings()
    report = run_m9_preflight(settings)
    print("M9 production reliability preflight: PASS")
    print(
        json.dumps(
            {
                "milestone": report["milestone"],
                "status": report["status"],
                "model_version": report["model_version"],
                "artifact_fingerprint": report["artifact_fingerprint"],
                "validation_split_accessed": report["validation_split_accessed"],
                "test_split_accessed": report["test_split_accessed"],
                "integrity": report["integrity"],
                "benchmark": report["benchmark"],
                "checks": report["checks"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )

def _cmd_m10(_: argparse.Namespace) -> None:
    settings = load_settings()
    manifest = run_m10_portfolio(settings)
    final_test = manifest["final_test"]
    print("M10 portfolio finalization + single final test: PASS")
    print(
        json.dumps(
            {
                "release": manifest["release"],
                "model_version": manifest["model_version"],
                "artifact_fingerprint": manifest["artifact_fingerprint"],
                "evaluation_source": manifest["evaluation_source"],
                "test_split_accessed_this_run": manifest["test_split_accessed_this_run"],
                "protocol": final_test["protocol"],
                "primary_segment": final_test["primary_segment"],
                "evaluation_users": final_test["deployed_policy"]["users"],
                "popularity_test": final_test["popularity_baseline"],
                "deployed_policy_test": final_test["deployed_policy"],
                "lift_vs_popularity_pct": final_test["lift_vs_popularity_pct"],
                "portfolio_checks": manifest["portfolio_checks"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def _cmd_serve(args: argparse.Namespace) -> None:
    import uvicorn

    uvicorn.run(
        "recsys_lab.serving.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )

def _cmd_assign(args: argparse.Namespace) -> None:
    settings = load_settings()
    assignment = assign_two_arm(
        experiment=args.experiment or settings.experiment.default_name,
        unit_id=args.unit_id,
        salt=settings.experiment.default_salt,
        control_weight=args.control_weight,
    )
    print(json.dumps(assignment.__dict__, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="recsys", description="Recsys Lab CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_download = sub.add_parser("download", help="Download and verify MovieLens 1M")
    p_download.add_argument("--force", action="store_true")
    p_download.set_defaults(func=_cmd_download)

    p_prepare = sub.add_parser("prepare", help="Convert raw MovieLens files to compressed CSV")
    p_prepare.set_defaults(func=_cmd_prepare)

    p_profile = sub.add_parser("profile", help="Validate and profile prepared data")
    p_profile.set_defaults(func=_cmd_profile)

    p_smoke = sub.add_parser("smoke", help="Run the complete M0 pipeline")
    p_smoke.add_argument("--force", action="store_true")
    p_smoke.set_defaults(func=_cmd_smoke)

    p_split = sub.add_parser("split", help="Create leakage-safe global chronological splits")
    p_split.set_defaults(func=_cmd_split)

    p_eda = sub.add_parser("eda", help="Generate M1 EDA from temporal splits")
    p_eda.set_defaults(func=_cmd_eda)

    p_m1 = sub.add_parser("m1", help="Run M1 temporal split, leakage checks and EDA")
    p_m1.set_defaults(func=_cmd_m1)

    p_m2 = sub.add_parser("m2", help="Run M2 validation-only offline baseline evaluation")
    p_m2.set_defaults(func=_cmd_m2)

    p_m3 = sub.add_parser("m3", help="Run M3 collaborative retrieval tuning on validation")
    p_m3.set_defaults(func=_cmd_m3)

    p_m4 = sub.add_parser("m4", help="Run M4 hybrid retrieval and cold-start evaluation")
    p_m4.set_defaults(func=_cmd_m4)

    p_m5 = sub.add_parser("m5", help="Run M5 pairwise learning-to-rank over frozen M4 candidates")
    p_m5.set_defaults(func=_cmd_m5)

    p_m6 = sub.add_parser("m6", help="Run M6 diversity/novelty-aware re-ranking")
    p_m6.set_defaults(func=_cmd_m6)

    p_m7 = sub.add_parser("m7", help="Run M7 A/B experiment simulation and statistical analysis")
    p_m7.set_defaults(func=_cmd_m7)


    p_m8 = sub.add_parser("m8", help="Build and preflight the frozen M8 serving bundle")
    p_m8.set_defaults(func=_cmd_m8)

    p_m9 = sub.add_parser("m9", help="Run M9 artifact-integrity and operational reliability preflight")
    p_m9.set_defaults(func=_cmd_m9)

    p_m10 = sub.add_parser("m10", help="Run the single final test and portfolio finalization")
    p_m10.set_defaults(func=_cmd_m10)

    p_serve = sub.add_parser("serve", help="Run the production-hardened FastAPI recommendation service")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--reload", action="store_true")
    p_serve.set_defaults(func=_cmd_serve)

    p_assign = sub.add_parser("assign", help="Smoke-test deterministic A/B bucketing")
    p_assign.add_argument("--unit-id", required=True)
    p_assign.add_argument("--experiment")
    p_assign.add_argument("--control-weight", type=float, default=0.5)
    p_assign.set_defaults(func=_cmd_assign)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
