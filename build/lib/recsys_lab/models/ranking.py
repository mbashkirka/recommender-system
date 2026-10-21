from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from recsys_lab.models.hybrid import HybridSVDMetadataRecommender


FEATURE_NAMES = (
    "retrieval_score",
    "reciprocal_rank",
    "warm_base_score",
    "metadata_score",
    "popularity_z",
    "log_positive_popularity",
    "log_user_seen_count",
    "log_user_positive_count",
    "genre_overlap",
    "is_cold_item",
)


def _finite_or(value: float, default: float = 0.0) -> float:
    return float(value) if np.isfinite(value) else float(default)


class CandidateFeatureBuilder:
    """Build leakage-safe user-item features from the frozen M4 train snapshot.

    All dynamic statistics come from M4's train-derived history/artifact. Movie genres are static
    metadata. Validation labels are never used while computing features.
    """

    def __init__(self, model: HybridSVDMetadataRecommender) -> None:
        self.model = model
        self.feature_names = FEATURE_NAMES
        self._warm_items = {int(x) for x in model.decomposition.item_ids}

    def build(self, user_id: int, candidate_ids: Iterable[int]) -> pd.DataFrame:
        user_id = int(user_id)
        candidates = [int(x) for x in candidate_ids]
        if not candidates:
            return pd.DataFrame(columns=("user_id", "item_id", "base_rank", *FEATURE_NAMES))

        retrieval = self.model.score_all(user_id)
        warm = self.model._warm_base_scores(user_id)  # train-only model component
        metadata = self.model._metadata_scores(user_id)  # train-history profile + static genres
        profile = self.model._metadata_profile(user_id)
        seen_count = len(self.model.history.seen_by_user.get(user_id, set()))
        positive_count = len(self.model.history.positive_by_user.get(user_id, set()))

        rows: list[dict[str, float | int]] = []
        for rank, item_id in enumerate(candidates, start=1):
            idx = self.model.metadata.item_index.get(item_id)
            if idx is None:
                continue
            meta_score = 0.0 if metadata is None else _finite_or(float(metadata[idx]))
            warm_score = 0.0 if warm is None else _finite_or(float(warm[idx]))
            genre_overlap = 0.0
            if profile is not None:
                genre_overlap = float(self.model.metadata.features[idx] @ profile)
            popularity = self.model.history.popularity_counts.get(item_id, 0)
            rows.append(
                {
                    "user_id": user_id,
                    "item_id": item_id,
                    "base_rank": rank,
                    "retrieval_score": _finite_or(float(retrieval[idx]), default=-8.0),
                    "reciprocal_rank": 1.0 / rank,
                    "warm_base_score": warm_score,
                    "metadata_score": meta_score,
                    "popularity_z": _finite_or(float(self.model._all_popularity_z[idx])),
                    "log_positive_popularity": float(np.log1p(popularity)),
                    "log_user_seen_count": float(np.log1p(seen_count)),
                    "log_user_positive_count": float(np.log1p(positive_count)),
                    "genre_overlap": genre_overlap,
                    "is_cold_item": float(item_id not in self._warm_items),
                }
            )
        return pd.DataFrame(rows)


@dataclass
class PairwiseLinearRanker:
    """Linear pairwise LTR model trained with logistic loss on feature differences."""

    c: float = 1.0
    seed: int = 42

    def __post_init__(self) -> None:
        if self.c <= 0:
            raise ValueError("c must be > 0")
        self.scaler = StandardScaler()
        self.model = LogisticRegression(
            C=float(self.c),
            solver="lbfgs",
            max_iter=2000,
            random_state=int(self.seed),
        )
        self.is_fitted = False

    @staticmethod
    def build_pairwise_training_data(
        frame: pd.DataFrame,
        negatives_per_positive: int,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
        if negatives_per_positive <= 0:
            raise ValueError("negatives_per_positive must be > 0")
        required = {"user_id", "label", "base_rank", *FEATURE_NAMES}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Missing ranker training columns: {sorted(missing)}")

        diffs: list[np.ndarray] = []
        labels: list[int] = []
        users_with_pairs = 0
        positive_candidates = 0
        for _, group in frame.groupby("user_id", sort=True):
            positives = group.loc[group["label"].eq(1)].sort_values("base_rank")
            negatives = group.loc[group["label"].eq(0)].sort_values("base_rank")
            if positives.empty or negatives.empty:
                continue
            users_with_pairs += 1
            positive_candidates += len(positives)
            neg = negatives.head(int(negatives_per_positive))
            neg_x = neg.loc[:, FEATURE_NAMES].to_numpy(dtype=np.float64)
            for _, pos in positives.iterrows():
                pos_x = pos.loc[list(FEATURE_NAMES)].to_numpy(dtype=np.float64)
                for neg_row in neg_x:
                    diff = pos_x - neg_row
                    # Symmetric examples force the decision boundary through the pairwise origin
                    # and remove arbitrary label imbalance/intercept effects.
                    diffs.append(diff)
                    labels.append(1)
                    diffs.append(-diff)
                    labels.append(0)
        if not diffs:
            raise ValueError("No positive-vs-negative candidate pairs available for LTR training")
        x = np.vstack(diffs)
        y = np.asarray(labels, dtype=np.int8)
        diagnostics = {
            "users_with_pairs": int(users_with_pairs),
            "positive_candidates": int(positive_candidates),
            "pairwise_examples": len(y),
        }
        return x, y, diagnostics

    def fit(self, frame: pd.DataFrame, negatives_per_positive: int) -> dict[str, int]:
        x, y, diagnostics = self.build_pairwise_training_data(frame, negatives_per_positive)
        x_scaled = self.scaler.fit_transform(x)
        self.model.fit(x_scaled, y)
        self.is_fitted = True
        return diagnostics

    def score_frame(self, frame: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Ranker must be fitted before scoring")
        x = frame.loc[:, FEATURE_NAMES].to_numpy(dtype=np.float64)
        x_scaled = self.scaler.transform(x)
        # For pairwise ranking only relative utility matters; intercept cancels for a user's items.
        return np.asarray(x_scaled @ self.model.coef_[0], dtype=np.float64)

    def rerank(self, frame: pd.DataFrame, blend: float) -> list[int]:
        if not 0.0 <= blend <= 1.0:
            raise ValueError("blend must be in [0, 1]")
        if frame.empty:
            return []
        base = -frame["base_rank"].to_numpy(dtype=np.float64)
        if blend == 0.0:
            return [int(x) for x in frame.sort_values("base_rank")["item_id"]]
        ltr = self.score_frame(frame)

        def z(x: np.ndarray) -> np.ndarray:
            std = float(x.std())
            return np.zeros_like(x) if std <= 1e-12 else (x - float(x.mean())) / std

        score = (1.0 - blend) * z(base) + blend * z(ltr)
        tmp = frame[["item_id", "base_rank"]].copy()
        tmp["score"] = score
        tmp = tmp.sort_values(["score", "base_rank", "item_id"], ascending=[False, True, True])
        return [int(x) for x in tmp["item_id"]]

    def feature_importance(self) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("Ranker must be fitted before reading coefficients")
        coef = np.asarray(self.model.coef_[0], dtype=float)
        return pd.DataFrame(
            {
                "feature": FEATURE_NAMES,
                "coefficient": coef,
                "abs_coefficient": np.abs(coef),
            }
        ).sort_values("abs_coefficient", ascending=False)

    def save(self, path: Path, blend: float) -> None:
        if not self.is_fitted:
            raise RuntimeError("Ranker must be fitted before saving")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            feature_names=np.asarray(FEATURE_NAMES, dtype="U64"),
            scaler_mean=self.scaler.mean_,
            scaler_scale=self.scaler.scale_,
            coefficients=self.model.coef_[0],
            intercept=self.model.intercept_,
            c=np.asarray([self.c], dtype=np.float64),
            blend=np.asarray([blend], dtype=np.float64),
        )
