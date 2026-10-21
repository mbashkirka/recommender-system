# Project Card — Production Recommender System

## Business framing

Build a personalized recommendation service that ranks a catalog for each user while balancing relevance, coverage, cold-start robustness and operational safety. MovieLens is the public benchmark domain, but the architecture mirrors an e-commerce/content recommendation stack.

## Data and label

Primary benchmark: MovieLens 1M. Ratings are converted into an implicit positive signal using a configurable threshold (`rating >= 4` in the completed run). Data is split globally by event time into train/validation/test. This avoids random-split leakage and keeps test sealed through all model-selection milestones.

## Offline objectives

Primary metrics are `NDCG@10`, `Recall@10` and `Candidate Recall@100`. Secondary metrics include MAP, HitRate, MRR, catalog coverage, novelty, diversity and explicit cold-start cohorts.

## Model stack

1. Train-popularity and content baselines.
2. PureSVD collaborative candidate generation.
3. Explicit warm/sparse/cold-user routing plus metadata cold-item support.
4. Pairwise LTR experiment over frozen candidates.
5. Diversity/novelty re-ranking with relevance guardrails and rollback.
6. Frozen serving bundle used by the API and the final test evaluation.

The final deployed policy deliberately keeps `ltr_blend=0.0` and zero diversity/novelty weights because the more complex challengers did not clear the relevant offline gates.

## Online objectives

The experiment layer supports sticky user-level randomization, exposure/outcome event contracts, SRM detection, confidence intervals, power/MDE, CUPED and latency/error guardrails. MovieLens does not contain randomized online impressions, so M7 uses a clearly labeled synthetic traffic harness rather than claiming a real online lift.

## Serving and reliability

FastAPI serves a frozen artifact, not a model trained at startup. Readiness is fail-closed on SHA-256 artifact integrity and train-only provenance. Runtime metrics expose latency, errors, routing and assigned-versus-served variants. The project includes load testing, Docker health checks, event persistence and GitHub Actions CI.

## Portfolio signal

The project demonstrates the full recommender lifecycle: data contracts, temporal leakage control, retrieval versus ranking, matrix factorization, cold start, rejected model experiments, deployment guardrails, A/B methodology, serving, observability, reproducibility and final holdout evaluation.
