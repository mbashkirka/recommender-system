from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from recsys_lab.models.baselines import FittedHistory
from recsys_lab.models.collaborative import PureSVDDecomposition


def _zscore(values: np.ndarray, finite_mask: np.ndarray | None = None) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if finite_mask is None:
        finite_mask = np.isfinite(values)
    out = np.full(values.shape, -np.inf, dtype=np.float32)
    if not finite_mask.any():
        return out
    selected = values[finite_mask]
    std = float(selected.std())
    if std <= 1e-12:
        out[finite_mask] = 0.0
    else:
        out[finite_mask] = (selected - float(selected.mean())) / std
    return out


@dataclass(frozen=True)
class MetadataIndex:
    item_ids: np.ndarray
    features: np.ndarray
    feature_names: tuple[str, ...]
    item_index: dict[int, int]


def build_metadata_index(movies: pd.DataFrame) -> MetadataIndex:
    """Build a leakage-safe item metadata matrix from static MovieLens movie metadata.

    Features are genres only. Static item metadata is not derived from validation/test interactions,
    so it can safely represent both warm and train-unseen (cold) items.
    """
    frame = movies[["item_id", "genres"]].copy()
    frame["item_id"] = frame["item_id"].astype(int)
    frame["genres"] = frame["genres"].fillna("").astype(str)
    frame = frame.drop_duplicates("item_id").sort_values("item_id").reset_index(drop=True)
    genres = sorted(
        {
            genre
            for value in frame["genres"]
            for genre in value.split("|")
            if genre and genre != "(no genres listed)"
        }
    )
    genre_index = {genre: idx for idx, genre in enumerate(genres)}
    features = np.zeros((len(frame), len(genres)), dtype=np.float32)
    for row_idx, value in enumerate(frame["genres"]):
        item_genres = [genre for genre in value.split("|") if genre in genre_index]
        if not item_genres:
            continue
        weight = 1.0 / len(item_genres)
        for genre in item_genres:
            features[row_idx, genre_index[genre]] = weight
    item_ids = frame["item_id"].to_numpy(dtype=np.int64)
    return MetadataIndex(
        item_ids=item_ids,
        features=features,
        feature_names=tuple(genres),
        item_index={int(item): idx for idx, item in enumerate(item_ids)},
    )


class HybridSVDMetadataRecommender:
    """Hybrid candidate generator for warm retrieval and cold-item support.

    Warm train items receive a PureSVD+popularity score. A user metadata profile is derived only
    from positive train history and scored against the complete static movie catalog, allowing
    train-unseen items to enter the candidate set. Unknown users deterministically fall back to
    train popularity; sparse users can receive an increased metadata weight.
    """

    def __init__(
        self,
        decomposition: PureSVDDecomposition,
        movies: pd.DataFrame,
        rank: int,
        popularity_blend: float,
        metadata_blend: float,
        sparse_positive_threshold: int = 5,
        sparse_metadata_multiplier: float = 1.5,
        cold_item_penalty: float = 0.25,
    ) -> None:
        if rank <= 0 or rank > decomposition.max_rank:
            raise ValueError(f"rank must be within [1, {decomposition.max_rank}]")
        if not 0.0 <= popularity_blend < 1.0:
            raise ValueError("popularity_blend must be in [0, 1)")
        if not 0.0 <= metadata_blend < 1.0:
            raise ValueError("metadata_blend must be in [0, 1)")
        if sparse_positive_threshold < 0:
            raise ValueError("sparse_positive_threshold must be >= 0")
        if sparse_metadata_multiplier <= 0:
            raise ValueError("sparse_metadata_multiplier must be > 0")
        if cold_item_penalty < 0:
            raise ValueError("cold_item_penalty must be >= 0")

        self.decomposition = decomposition
        self.history: FittedHistory = decomposition.history
        self.rank = int(rank)
        self.popularity_blend = float(popularity_blend)
        self.metadata_blend = float(metadata_blend)
        self.sparse_positive_threshold = int(sparse_positive_threshold)
        self.sparse_metadata_multiplier = float(sparse_metadata_multiplier)
        self.cold_item_penalty = float(cold_item_penalty)
        self.name = f"hybrid_r{rank}_p{popularity_blend:g}_m{metadata_blend:g}"

        self.metadata = build_metadata_index(movies)
        self._all_item_ids = self.metadata.item_ids
        self._all_item_index = self.metadata.item_index
        self._svd_user_index = {
            int(user_id): idx for idx, user_id in enumerate(decomposition.user_ids)
        }
        self._svd_item_index = {
            int(item_id): idx for idx, item_id in enumerate(decomposition.item_ids)
        }

        # Map train-catalog SVD positions into the complete metadata catalog once.
        self._svd_to_all = np.array(
            [self._all_item_index.get(int(item_id), -1) for item_id in decomposition.item_ids],
            dtype=np.int64,
        )
        counts = np.array(
            [self.history.popularity_counts.get(int(item), 0) for item in self._all_item_ids],
            dtype=np.float32,
        )
        log_pop = np.log1p(counts)
        warm_mask = counts > 0
        pop_z = _zscore(log_pop, warm_mask)
        # Metadata-only cold items get a conservative prior rather than -inf.
        floor = float(np.min(pop_z[warm_mask])) if warm_mask.any() else 0.0
        pop_z[~warm_mask] = floor - self.cold_item_penalty
        self._all_popularity_z = pop_z

    def _metadata_profile(self, user_id: int) -> np.ndarray | None:
        positives = self.history.positive_by_user.get(int(user_id), set())
        rows = [self._all_item_index[item] for item in positives if item in self._all_item_index]
        if not rows:
            return None
        profile = self.metadata.features[rows].sum(axis=0)
        norm = float(profile.sum())
        if norm <= 1e-12:
            return None
        return (profile / norm).astype(np.float32)

    def _metadata_scores(self, user_id: int) -> np.ndarray | None:
        profile = self._metadata_profile(user_id)
        if profile is None:
            return None
        raw = self.metadata.features @ profile
        return _zscore(raw)

    def _warm_base_scores(self, user_id: int) -> np.ndarray | None:
        user_idx = self._svd_user_index.get(int(user_id))
        if user_idx is None:
            return None
        r = self.rank
        weighted_user = (
            self.decomposition.user_components[user_idx, :r]
            * self.decomposition.singular_values[:r]
        )
        raw_cf = weighted_user @ self.decomposition.item_components[:, :r].T
        cf_z = _zscore(np.asarray(raw_cf, dtype=np.float32))
        warm = (1.0 - self.popularity_blend) * cf_z + (
            self.popularity_blend * self.decomposition.popularity_z
        )
        all_scores = np.full(len(self._all_item_ids), -np.inf, dtype=np.float32)
        valid = self._svd_to_all >= 0
        all_scores[self._svd_to_all[valid]] = warm[valid]
        return all_scores

    def route_for_user(self, user_id: int) -> str:
        user_id = int(user_id)
        if user_id not in self.history.seen_by_user:
            return "cold_user_popularity"
        positive_count = len(self.history.positive_by_user.get(user_id, set()))
        if positive_count <= self.sparse_positive_threshold:
            return "sparse_user_hybrid"
        return "warm_user_hybrid"

    def score_all(self, user_id: int) -> np.ndarray:
        user_id = int(user_id)
        base = self._warm_base_scores(user_id)
        metadata = self._metadata_scores(user_id)
        route = self.route_for_user(user_id)

        if route == "cold_user_popularity" or base is None:
            scores = self._all_popularity_z.copy()
        elif metadata is None or self.metadata_blend <= 0.0:
            scores = base.copy()
        else:
            blend = self.metadata_blend
            if route == "sparse_user_hybrid":
                blend = min(0.95, blend * self.sparse_metadata_multiplier)

            warm_mask = np.isfinite(base)
            # For warm items, blend calibrated SVD and metadata scores.
            scores = np.full(len(self._all_item_ids), -np.inf, dtype=np.float32)
            scores[warm_mask] = (1.0 - blend) * base[warm_mask] + blend * metadata[warm_mask]

            # Cold items cannot receive an SVD score; metadata is the candidate-generation channel.
            cold_mask = ~warm_mask & np.isfinite(metadata)
            if cold_mask.any() and blend > 0.0:
                scores[cold_mask] = (
                    blend * metadata[cold_mask]
                    + (1.0 - blend) * self._all_popularity_z[cold_mask]
                )

        seen = self.history.seen_by_user.get(user_id, set())
        for item_id in seen:
            idx = self._all_item_index.get(int(item_id))
            if idx is not None:
                scores[idx] = -np.inf
        return scores

    def recommend(self, user_id: int, k: int) -> list[int]:
        if k <= 0:
            return []
        scores = self.score_all(int(user_id))
        finite_idx = np.flatnonzero(np.isfinite(scores))
        if len(finite_idx) == 0:
            return []
        take = min(int(k), len(finite_idx))
        if take == len(finite_idx):
            candidate_idx = finite_idx
        else:
            local = np.argpartition(scores[finite_idx], -take)[-take:]
            candidate_idx = finite_idx[local]
        candidate_items = self._all_item_ids[candidate_idx]
        candidate_scores = scores[candidate_idx]
        order = np.lexsort((candidate_items, -candidate_scores))
        return [int(item) for item in candidate_items[order]]

    def recommend_with_cold_item_quota(self, user_id: int, k: int, cold_item_quota: int) -> list[int]:
        """Inject a bounded metadata-ranked quota of train-unseen items for catalog exploration.

        This policy is used only when evaluating/serving new-item discovery. It does not contaminate
        the primary warm retrieval benchmark, where the M3-compatible path remains the anchor.
        """
        if k <= 0:
            return []
        quota = min(max(int(cold_item_quota), 0), int(k))
        if quota == 0:
            return self.recommend(user_id, k)
        metadata = self._metadata_scores(int(user_id))
        if metadata is None:
            return self.recommend(user_id, k)

        train_items = {int(x) for x in self.decomposition.item_ids}
        seen = self.history.seen_by_user.get(int(user_id), set())
        cold_idx = np.array(
            [
                idx
                for idx, item in enumerate(self._all_item_ids)
                if int(item) not in train_items and int(item) not in seen and np.isfinite(metadata[idx])
            ],
            dtype=np.int64,
        )
        if len(cold_idx) == 0:
            return self.recommend(user_id, k)
        take_cold = min(quota, len(cold_idx))
        if take_cold == len(cold_idx):
            chosen_idx = cold_idx
        else:
            local = np.argpartition(metadata[cold_idx], -take_cold)[-take_cold:]
            chosen_idx = cold_idx[local]
        cold_items = self._all_item_ids[chosen_idx]
        cold_scores = metadata[chosen_idx]
        order = np.lexsort((cold_items, -cold_scores))
        injected = [int(x) for x in cold_items[order]]

        base = [item for item in self.recommend(user_id, k) if item not in set(injected)]
        # Keep most capacity for the proven warm retriever and reserve only the explicit quota.
        return base[: max(0, k - len(injected))] + injected
    def save(self, path: str | Path) -> None:
        """Persist the inference state needed to reconstruct M4 without refitting."""
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        seen_pairs = [
            (user_id, item_id)
            for user_id, items in self.history.seen_by_user.items()
            for item_id in items
        ]
        positive_pairs = [
            (user_id, item_id)
            for user_id, items in self.history.positive_by_user.items()
            for item_id in items
        ]
        seen_users = np.array([x[0] for x in seen_pairs], dtype=np.int64)
        seen_items = np.array([x[1] for x in seen_pairs], dtype=np.int64)
        positive_users = np.array([x[0] for x in positive_pairs], dtype=np.int64)
        positive_items = np.array([x[1] for x in positive_pairs], dtype=np.int64)
        np.savez_compressed(
            output,
            user_ids=self.decomposition.user_ids,
            svd_item_ids=self.decomposition.item_ids,
            user_components=self.decomposition.user_components[:, : self.rank],
            singular_values=self.decomposition.singular_values[: self.rank],
            item_components=self.decomposition.item_components[:, : self.rank],
            svd_popularity_z=self.decomposition.popularity_z,
            all_item_ids=self._all_item_ids,
            metadata_features=self.metadata.features,
            feature_names=np.asarray(self.metadata.feature_names, dtype="U64"),
            all_popularity_z=self._all_popularity_z,
            seen_user_ids=seen_users,
            seen_item_ids=seen_items,
            positive_user_ids=positive_users,
            positive_item_ids=positive_items,
            rank=np.array([self.rank], dtype=np.int32),
            popularity_blend=np.array([self.popularity_blend], dtype=np.float32),
            metadata_blend=np.array([self.metadata_blend], dtype=np.float32),
            sparse_positive_threshold=np.array([self.sparse_positive_threshold], dtype=np.int32),
            sparse_metadata_multiplier=np.array([self.sparse_metadata_multiplier], dtype=np.float32),
            cold_item_penalty=np.array([self.cold_item_penalty], dtype=np.float32),
        )

