# M3 — Collaborative Candidate Generation

## Objective

Move from heuristic baselines to a learned collaborative representation while keeping the M1
chronological protocol and the M2 sealed-test rule unchanged.

## Fit data

Only `data/splits/train.csv.gz` is used to learn factors. Ratings are converted to the already
frozen implicit-positive signal (`rating >= 4` by default). Negative/low ratings do not become
positive edges. The sparse matrix is binary even if a generic future interaction log contains
repeated positive events.

## Model

PureSVD factorizes the sparse positive interaction matrix approximately as:

`X ≈ U Σ Vᵀ`.

The user representation is `UΣ`; item representation is `V`. Candidate scores are reconstructed
latent similarities. The maximum requested rank is fitted once and lower ranks use truncated
components from the same decomposition.

## Popularity blend

A small popularity prior is allowed because a globally chronological benchmark can exhibit temporal
catalog/popularity drift. Collaborative scores are standardized per user; log train-positive item
popularity is standardized globally. The configured blend is selected only on validation.

A blend of zero is pure collaborative retrieval. Values are restricted to `[0, 1)` so M3 cannot
silently collapse into a pure popularity recommender.

## Evaluation

M3 uses the same primary `warm_user_warm_item` positive validation cohort as M2. Seen train items are
removed from recommendation candidates.

Two kinds of metrics are reported:

1. ranking diagnostics at Top-10 (`NDCG`, `Recall`, `MAP`, `HitRate`);
2. candidate-stage `Recall@50` and `Recall@100`.

The selected model is tuned by NDCG@10 with Recall@100 as a deterministic tie-breaker. M4/M5 will
make candidate recall increasingly important as the architecture becomes an explicit two-stage
retrieval → ranking system.

## Leakage contract

- factor fit: train only;
- hyperparameter selection: validation only;
- baseline popularity statistics: train only;
- test split: never opened in M3;
- selected model artifact records `test_split_accessed: false`.
