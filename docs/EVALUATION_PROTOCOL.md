# Offline Evaluation Protocol

## Objective

The project is evaluated as an **implicit-feedback Top-N recommender**. A rating is a positive preference signal when `rating >= 4.0`. The original explicit rating is retained for analysis, but candidate-generation models use the binary preference signal unless a milestone explicitly states otherwise.

## Global chronological split

Random splitting leaks future behavior. The project uses a **global chronological 80/10/10 split**:

- earliest ~80% of events: train;
- next ~10%: validation;
- latest ~10%: test.

All events sharing a cutoff timestamp remain on the earlier side, preserving strict temporal ordering across periods.

## Evaluation segments

The primary comparable benchmark is **warm-user / warm-item** positive events: both user and item must have appeared in train. Cold-user and cold-item events are tracked separately and receive explicit fallback policies in M4.

MovieLens 1M chronology makes validation notably colder than test under a global split. This is a dataset property, not a reason to switch to random splitting. Cohort coverage is therefore reported next to primary metrics.

## Model-selection discipline

1. Fit the candidate model and train-derived statistics on train.
2. Select/tune architecture and hyperparameters using validation only.
3. Never use test results for hyperparameter decisions.
4. Build one frozen serving bundle after validation decisions are complete.
5. M8/M9 serve and verify this exact frozen bundle.
6. M10 evaluates that same bundle on test once; there is **no train+validation refit** in this project because the portfolio intentionally evaluates the exact artifact that was operationally validated and deployed.
7. Learned features/statistics (popularity, priors, encoders, normalization constants) must come only from the fitting period encoded in the frozen artifact.

This choice trades some possible final-test performance for a stronger artifact-consistency claim: the model measured in M10 is byte-for-byte the model whose SHA-256 was exercised by the API, reliability checks and Docker runtime.

## Seen-item exclusion

Recommendations exclude items already observed for a user in train, independent of whether the historical rating was positive or negative. M10 keeps this exact serving-time exclusion behavior rather than changing history semantics specifically for the final benchmark.

## M2 ground truth

For each evaluation user, relevant items are all positive validation items in the warm-user / warm-item segment. Users with no evaluable warm positive item are not included in the primary ranking average; their exclusion is captured in cohort diagnostics rather than hidden.

## Metrics

Primary development metrics: **NDCG@10** and **Recall@10**.

Retrieval metric: **Candidate Recall@100**.

Secondary: Precision@K, HitRate@K, MAP@K, MRR@K, catalog coverage and novelty/diversity measures. Online A/B metrics remain separate from offline ranking metrics.

## M5 ranker-development protocol

M5 keeps the global test split sealed. The validation period is chronologically subdivided into rank-train (50%), rank-tune (25%), and rank-eval (25%). M4 retrieval is frozen. Pairwise LTR uses rank-train labels, hyperparameters are selected on rank-tune, and the selected candidate is checked on the later rank-eval window. In the completed run, the no-regression anchor (`ltr_blend=0.0`) won.

## M6 reranker protocol

M6 further subdivides the later development window into rerank-tune and rerank-eval. Diversity/novelty weights are selected only on rerank-tune. The later rerank-eval acts as a deployment safety gate; a failed relevance guardrail causes an exact rollback to the anchor rather than another round of tuning on the gate window.

## M10 final test protocol

M10 is the only milestone that intentionally reads test. It evaluates:

- the already frozen control/deployed serving policy from M8/M9;
- a train-only popularity baseline for context;
- the same primary warm-user/warm-item cohort definition.

M10 does not pick a winner from test and does not alter serving parameters afterward. The first successful evaluation is locked in `artifacts/m10/final_test_locked.json`; subsequent M10 runs reuse the cached result rather than reopening test.
