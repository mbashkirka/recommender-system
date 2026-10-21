# M6 — Diversity/Novelty-Aware Re-ranking

M6 adds the third stage of the recommender pipeline. M4 retrieves Top-100 candidates, M5 optionally
changes their utility order, and M6 greedily re-ranks the prefix for list-level quality.

## Objective

Pointwise/pairwise relevance alone tends to create homogeneous lists. M6 therefore optimizes two
train-safe list properties:

- **Intra-list diversity (ILD):** mean pairwise genre dissimilarity, `1 - Jaccard(genres_i, genres_j)`.
- **Novelty:** mean self-information in bits from Laplace-smoothed positive train popularity.

The selected configuration is chosen only after relevance guardrails pass. The development
objective is: maximize ILD, then novelty and catalog coverage.

## Leakage protocol

The global test split remains unopened. M5 already partitioned validation into rank-train,
rank-tune and rank-eval. M6 takes only the M5 rank-eval window and chronologically partitions it
again:

```text
M5 rank-eval
├── first 50%  -> rerank-tune (select diversity/novelty weights)
└── last 50%   -> rerank-eval (report M6 once)

Global test -> sealed
```

M5's candidate/ranking artifacts are frozen. M6 cannot add a relevant candidate that retrieval
missed, so candidate Recall@100 is unchanged by construction.

## Re-ranker

For an M5-ordered candidate pool, greedy utility is:

```text
utility(item) = rank_relevance
                - diversity_weight * max_genre_similarity_to_selected
                + novelty_weight * standardized_novelty
```

Only the first `pool_k=50` candidates are considered for Top-10 re-ranking. The full candidate list
is preserved after the diversified prefix.

A configurable `max_per_genre=4` soft business constraint avoids excessive concentration in one
primary genre. If the constraint would make the list too short it is relaxed deterministically.

## No-regression contract

The grid includes `(diversity_weight=0, novelty_weight=0)`. That pair bypasses all M6 changes and
returns the M5 order exactly.

Non-anchor configurations are eligible only when on rerank-tune:

- `NDCG@10 >= 0.97 * M5 NDCG@10`
- `Recall@10 >= 0.95 * M5 Recall@10`

If diversification cannot satisfy those guardrails on rerank-tune, the exact M5 order remains selectable.

## Deployment gate

The tune-selected candidate is evaluated once on the later `rerank-eval` window. This holdout is
not used to search another weight pair; it is only a binary deployment safety gate:

```text
rerank-eval passes NDCG/Recall guardrails -> deploy tune-selected M6 parameters
rerank-eval fails guardrails             -> rollback to exact M5 (0,0) anchor
```

This prevents a diversity configuration that overfits rerank-tune from becoming the serving
artifact. The manifest preserves both the development-selected candidate and the actually deployed
parameters for auditability.

## Outputs

```text
reports/m6/
├── reranker_tuning.csv
├── m5_vs_m6_rerank_eval.csv
├── m5_vs_m6_rerank_eval.png
├── per_user_rerank_eval.csv.gz
└── sample_rerankings.csv

artifacts/m6/
├── reranker_selected.json
└── m6_manifest.json
```

The JSON artifact contains only serving-time M6 parameters; catalog metadata and popularity state
remain owned by the frozen M4 artifact.
