# Changelog

## 1.0.2 — M10.1.1 final lint hotfix

- Fixed the final Ruff `I001` import-block formatting issue in `models/ranking.py`.
- No model, evaluation, serving-artifact, or locked-test-result changes.

## 1.0.1 — M10.1 release-quality hotfix

- Cleared the Ruff release-gate debt without weakening lint configuration.
- Fixed import ordering, unused imports, redundant casts, modern typing/datetime usage and small code-quality issues.
- Fixed the `Path` annotation/import issue in the hybrid model persistence path.
- Preserved all recommender algorithms, frozen serving parameters and the M10 test lock.
- Published the exact sealed-test metrics in the portfolio README and final-results document.

## 1.0.0 — M10

- Added a single final evaluation of the frozen serving policy on the previously sealed test split.
- Added lock/caching semantics so repeated M10 runs do not reopen test.
- Reworked README into a GitHub portfolio landing page.
- Added architecture, runbook, reproducibility, benchmark and interview documentation.
- Added demo and release-check scripts.
- Serving model remains `recsys-m9`; no model refit or test-based tuning.

## 0.10.1 — M9.1

- Fixed Docker runtime config resolution with explicit `RECSYS_CONFIG`.
- Added regression coverage for installed/container layouts.

## 0.10.0 — M9

- Added artifact-integrity readiness, runtime metrics, operational preflight, load testing, Docker and CI.

## 0.9.0 — M8

- Added frozen serving bundle, FastAPI recommendation endpoints and exposure/outcome logging.

## 0.8.0 — M7

- Added sticky A/B assignment, SRM, CI, power/MDE, CUPED and guardrails.

## 0.7.0 — M6

- Added diversity/novelty reranking and deployment rollback gate.

## 0.6.0 — M5

- Added pairwise learning-to-rank experiment.

## 0.5.0 — M4

- Added hybrid/cold-start routing.

## 0.4.0 — M3

- Added PureSVD collaborative retrieval.

## 0.3.0 — M2

- Added offline baseline framework and ranking metrics.

## 0.2.0 — M1

- Added global chronological split and leakage-safe EDA.

## 0.1.0 — M0

- Added project bootstrap, MovieLens ingestion and deterministic A/B bucketing primitive.
