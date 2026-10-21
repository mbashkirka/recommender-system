# Final Results — Frozen Production Policy

## Protocol

M10 performed the first and final evaluation of the already frozen production policy on the sealed test split. No model was refit and no hyperparameter was selected using test. The evaluated serving artifact is the same artifact validated by FastAPI, M9 reliability checks, and Docker.

- **Release:** `1.0.2`
- **Serving model:** `recsys-m9`
- **Artifact SHA-256:** `6847221923010464582cd83358963051d500f9f6ee36e647a20e13a8abd2a74f`
- **Evaluation source:** `fresh_single_final_test_evaluation`
- **Primary segment:** `warm_user_warm_item`
- **Evaluation users:** `992`

## Held-out test metrics

| Metric | Popularity baseline | Deployed policy | Relative lift |
|---|---:|---:|---:|
| NDCG@10 | 0.197313 | 0.211687 | **+7.29%** |
| Recall@10 | 0.053533 | 0.064354 | **+20.21%** |
| Precision@10 | 0.181855 | 0.189718 | +4.32% |
| HitRate@10 | 0.622984 | 0.657258 | +5.50% |
| MAP@10 | 0.115971 | 0.120176 | +3.63% |
| MRR@10 | 0.343678 | 0.383245 | +11.51% |
| Candidate Recall@100 | 0.272698 | 0.311579 | **+14.26%** |
| Catalog Coverage@10 | 0.035227 | 0.145549 | **+313.18%** |

The deployed policy improves all reported ranking/retrieval metrics while recommending from a substantially broader portion of the train catalog.

## Model-selection record

The final policy preserves the project's negative experimental outcomes rather than hiding them:

- M4 selected `metadata_blend=0.0` for warm retrieval.
- M5 selected `ltr_blend=0.0`; the pairwise LTR model was not promoted.
- M6's diversity/novelty challenger failed the later relevance deployment gate and was rolled back to the exact M5 anchor.
- M7 validates the experimentation stack using synthetic online traffic only; its uplift is `demo_only=true` and is not presented as observed recommender lift.

## Lock semantics

The first successful M10 run writes the locked final-test result under `artifacts/m10/`. Subsequent M10 runs reuse the cached result rather than reopening test, preventing the holdout from becoming another tuning set.
