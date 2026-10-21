# Recommender System Roadmap

The project is deliberately built milestone by milestone. Every milestone archive contains the complete project state.

| Milestone | Scope | Main deliverable |
|---|---|---|
| M0 | Bootstrap, dataset acquisition, data contract, A/B bucketing primitive | Reproducible local project + smoke pipeline |
| M1 | Time-aware data split, leakage controls, EDA, implicit-feedback definition | Frozen train/validation/test protocol |
| M2 | Ranking metrics and baselines | Popularity/random/personal-history baselines; Recall@K, NDCG@K, MAP@K, coverage |
| M3 | Collaborative candidate generation | Sparse PureSVD retrieval + persisted latent factors |
| M4 | Retrieval strengthening and cold-start | PureSVD + metadata hybrid, NDCG guardrail, explicit cold-user/new-item routing |
| M5 | Learning-to-rank | Feature pipeline + ranker over retrieved candidates |
| M6 | Re-ranking | Diversity, novelty, deduplication, catalog/business constraints |
| M7 | A/B experimentation | Sticky assignment, exposure/event logging, SRM, CI, power/MDE, guardrails, optional CUPED |
| M8 | Online serving | FastAPI endpoints, model registry, versioning, cache, experiment integration |
| M9 | Reliability | Monitoring, latency/load tests, Docker, CI, reproducibility checks |
| M10 | Portfolio polish | Final README, diagrams, demo, interview notes, experiment report |

## Target architecture

1. **Candidate generation** retrieves a few hundred plausible items per user.
2. **Ranking** predicts user-item utility using collaborative and metadata features.
3. **Re-ranking** applies diversity/novelty/business constraints and returns Top-K.
4. **Experiment layer** assigns a user to control/treatment before model selection.
5. **Event layer** records exposure and downstream behavior for online evaluation.

## A/B principle

The online experiment is not a notebook-only t-test. The system will include a stable hash-based assignment API and event contract so the recommendation service can serve model A or B consistently. M7 adds analysis tooling for sample-ratio mismatch, confidence intervals, practical/relative lift, power and minimum detectable effect, plus guardrail metrics.


Current implementation status: **M10.1 / release 1.0.2 implemented**.
