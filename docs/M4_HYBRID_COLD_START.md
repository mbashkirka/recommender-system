# M4 — Hybrid Retrieval and Cold Start

## Why M4 exists

M3 proves that collaborative latent factors beat the popularity baseline on the warm validation
cohort. Pure collaborative filtering still has two structural limitations: it cannot score items
that never appeared in train, and it has weak behavior for users with little or no interaction
history. M4 adds explicit mechanisms for both without opening the test split.

## Fixed M3 anchor

M4 does not re-tune the M3 rank/popularity pair. It reads the selected M3 manifest and keeps those
parameters fixed. The SVD decomposition is rebuilt from train using the same maximum-rank contract,
so `metadata_blend=0.0` is the no-regression anchor.

## Metadata channel

Static MovieLens genres are encoded into a normalized multi-hot item matrix. For a known user, the
metadata profile is the normalized sum of genre vectors from positive train interactions. Static
metadata is not derived from validation/test behavior and can represent train-unseen items.

For warm train items, M4 blends standardized collaborative/popularity score with standardized genre
affinity. For train-unseen items, no collaborative score exists; metadata becomes the only
personalization signal.

## Selection objective and guardrail

M4 is a retrieval milestone, so the main selection objective is Recall@100. However, a candidate
model is eligible only when its NDCG@10 remains at least the configured fraction of the M3 value.
The default guardrail is 97%. If no hybrid variant passes, the mandatory zero-metadata anchor is the
fallback rather than silently accepting a ranking collapse.

## Routing

`HybridSVDMetadataRecommender.route_for_user` exposes inference behavior explicitly:

- `warm_user_hybrid`: sufficient positive train history;
- `sparse_user_hybrid`: limited positive history, so the metadata contribution is increased;
- `cold_user_popularity`: no train history, so only train-derived popularity is trusted.

This route label will later be logged as part of the serving/experiment event schema.

## Cold-item exploration

A brand-new item has no interaction evidence. M4 therefore separates ordinary warm retrieval from
a bounded cold-item exploration policy. `recommend_with_cold_item_quota` reserves a configured
number of candidate slots for metadata-ranked train-unseen items. This makes cold-item exposure
measurable and controllable instead of allowing new items to remain permanently invisible.

The quota is evaluated only on the `warm_user_cold_item` validation cohort and does not alter the
primary warm-user/warm-item metric table.

## Leakage contract

- SVD fit: train only;
- train popularity: train only;
- user genre profiles: positive train interactions only;
- item genres: static catalog metadata;
- M4 tuning: validation only;
- test split: never opened.

## Artifact contract

`artifacts/m4/hybrid_selected.npz` stores:

- selected SVD components and singular values;
- full metadata item catalog and feature matrix;
- popularity priors;
- train seen-user/item pairs;
- positive train history pairs;
- selected hybrid/routing parameters.

This gives M8 a serving-oriented artifact without requiring a model refit.
