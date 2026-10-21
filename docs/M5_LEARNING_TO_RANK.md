# M5 — Learning-to-Rank

M5 turns the recommender into a true two-stage system. The M4 retrieval model is frozen and returns
Top-100 candidates. A pairwise linear learning-to-rank model then reorders only those candidates.

## Leakage contract

The global `test` split remains sealed. M5 partitions the existing validation period
chronologically into:

- first 50%: `rank_train` labels;
- next 25%: `rank_tune` for `C` and LTR blend selection;
- final 25%: `rank_eval` for a one-shot M5 development report.

This is an internal development holdout, not the final test benchmark. M4 candidate hyperparameters
remain frozen; M5 does not retune retrieval.

## Pairwise objective

For each user, retrieved relevant candidates are paired with high-ranked non-relevant candidates.
The ranker learns standardized feature differences with logistic loss. Symmetric positive/negative
pairs make the learned utility suitable for scoring individual candidates.

Features are train-snapshot or static-metadata features only: retrieval score/rank, collaborative
base score, metadata affinity, popularity, user history size, genre overlap, and cold-item flag.

## No-regression anchor

`ltr_blend=0.0` exactly preserves M4 candidate order. Tuning may choose a larger blend only when it
improves the chronological `rank_tune` objective. Candidate Recall@100 is unchanged by construction
because M5 only reorders the same Top-100 pool.

## Outputs

`reports/m5/` contains tuning, feature importance, per-user deltas, sample rerankings, and M4-vs-M5
rank-eval comparison. `artifacts/m5/pairwise_ltr_selected.npz` contains serving coefficients and
feature scaling state.
