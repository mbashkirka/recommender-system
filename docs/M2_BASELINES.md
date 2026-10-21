# M2 — Offline Ranking Baselines

## Goal

M2 establishes a reproducible lower bound before collaborative retrieval is introduced. All model
selection is performed on **validation only**. The test split remains sealed.

## Evaluation cohort

The primary cohort contains validation positive events (`rating >= 4`) where both the user and the
item appeared in train. Users must have at least one such relevant item. Candidate items are the
train catalog, and every item already rated by that user in train is excluded before ranking.

## Baselines

1. **Random** — deterministic user-specific random ranking. This is a sanity floor and should not
   outperform meaningful recommenders consistently.
2. **Popularity** — items ranked by positive train interaction count, with item id as deterministic
   tie-breaker.
3. **Genre affinity** — a personalized content baseline. The user's positive train history is
   converted to a genre distribution; candidate films are scored by overlap with that profile.
   Popularity is used only as a tiny deterministic tie-breaker. Users without a usable genre
   history fall back to popularity.

No statistic is computed from validation labels before ranking.

## Metrics

For K in `{5, 10, 20}`:

- Recall@K — fraction of a user's relevant validation items retrieved in Top-K;
- Precision@K — fraction of Top-K recommendations that are relevant;
- HitRate@K — fraction of users with at least one hit;
- NDCG@K — position-sensitive ranking quality under binary relevance;
- MAP@K — mean average precision with `min(|relevant|, K)` denominator;
- MRR@K — reciprocal rank of the first relevant recommendation;
- catalog coverage — unique recommended train items / train catalog size;
- mean novelty bits — smoothed self-information based only on train positive popularity.

The primary comparison point is K=10 and the primary ranking metric is NDCG@10. Recall@10 is the
main companion metric.

## Why these baselines matter

M3 collaborative retrieval must beat these baselines on validation under the same cohort and
candidate-exclusion rules. If a more sophisticated model cannot beat popularity or genre affinity,
there is no justification for deploying it.
