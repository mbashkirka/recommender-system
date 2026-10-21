from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import svds

from recsys_lab.models.baselines import FittedHistory, build_history


@dataclass(frozen=True)
class PureSVDDecomposition:
    user_ids: np.ndarray
    item_ids: np.ndarray
    user_components: np.ndarray
    singular_values: np.ndarray
    item_components: np.ndarray
    popularity_z: np.ndarray
    history: FittedHistory

    @property
    def max_rank(self) -> int:
        return len(self.singular_values)


def _zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    std = float(values.std())
    if std <= 1e-12:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - float(values.mean())) / std).astype(np.float32)


def build_positive_matrix(
    train: pd.DataFrame,
) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray]:
    positive = train.loc[train["is_positive"].astype(bool), ["user_id", "item_id"]].copy()
    if positive.empty:
        raise ValueError("No positive interactions available for collaborative training")

    user_ids = np.sort(train["user_id"].astype(int).unique()).astype(np.int64)
    item_ids = np.sort(train["item_id"].astype(int).unique()).astype(np.int64)
    user_index = {int(user_id): idx for idx, user_id in enumerate(user_ids)}
    item_index = {int(item_id): idx for idx, item_id in enumerate(item_ids)}

    rows = positive["user_id"].astype(int).map(user_index).to_numpy(dtype=np.int32)
    cols = positive["item_id"].astype(int).map(item_index).to_numpy(dtype=np.int32)
    data = np.ones(len(positive), dtype=np.float32)
    matrix = sparse.coo_matrix(
        (data, (rows, cols)), shape=(len(user_ids), len(item_ids)), dtype=np.float32
    ).tocsr()
    # Generic interaction logs may contain repeats. Binary implicit feedback is the M3 contract.
    if matrix.nnz:
        matrix.data[:] = 1.0
    matrix.eliminate_zeros()
    return matrix, user_ids, item_ids


def fit_puresvd(train: pd.DataFrame, max_rank: int, seed: int = 42) -> PureSVDDecomposition:
    matrix, user_ids, item_ids = build_positive_matrix(train)
    min_dim = min(matrix.shape)
    if max_rank <= 0 or max_rank >= min_dim:
        raise ValueError(f"max_rank must be in [1, {min_dim - 1}], got {max_rank}")

    # svds returns singular triplets in ascending singular-value order.
    u, s, vt = svds(matrix, k=max_rank, which="LM", random_state=int(seed))
    order = np.argsort(s)[::-1]
    u = np.asarray(u[:, order], dtype=np.float32)
    s = np.asarray(s[order], dtype=np.float32)
    vt = np.asarray(vt[order, :], dtype=np.float32)

    history = build_history(train)
    popularity = np.log1p(
        np.array([history.popularity_counts.get(int(item), 0) for item in item_ids], dtype=np.float32)
    )
    popularity_z = _zscore(popularity)
    return PureSVDDecomposition(
        user_ids=user_ids,
        item_ids=item_ids,
        user_components=u,
        singular_values=s,
        item_components=vt.T,
        popularity_z=popularity_z,
        history=history,
    )


class PureSVDRecommender:
    """Latent-factor candidate generator with optional popularity blending.

    The SVD is fitted only on binary positive train interactions. For each user, reconstructed
    collaborative scores are standardized across the catalog before blending with a standardized
    log-popularity prior. ``popularity_blend=0`` is pure collaborative retrieval.
    """

    def __init__(
        self,
        decomposition: PureSVDDecomposition,
        rank: int,
        popularity_blend: float = 0.0,
    ):
        if rank <= 0 or rank > decomposition.max_rank:
            raise ValueError(f"rank must be within [1, {decomposition.max_rank}]")
        if not 0.0 <= popularity_blend < 1.0:
            raise ValueError("popularity_blend must be in [0, 1)")
        self.decomposition = decomposition
        self.rank = int(rank)
        self.popularity_blend = float(popularity_blend)
        self.name = f"puresvd_r{rank}_b{popularity_blend:g}"
        self._user_index = {
            int(user_id): idx for idx, user_id in enumerate(decomposition.user_ids)
        }
        self._item_index = {
            int(item_id): idx for idx, item_id in enumerate(decomposition.item_ids)
        }

    def _collaborative_scores(self, user_id: int) -> np.ndarray | None:
        idx = self._user_index.get(int(user_id))
        if idx is None:
            return None
        r = self.rank
        weighted_user = (
            self.decomposition.user_components[idx, :r] * self.decomposition.singular_values[:r]
        )
        scores = weighted_user @ self.decomposition.item_components[:, :r].T
        return _zscore(np.asarray(scores, dtype=np.float32))

    def recommend(self, user_id: int, k: int) -> list[int]:
        if k <= 0:
            return []
        cf_scores = self._collaborative_scores(int(user_id))
        if cf_scores is None:
            # M3 primary cohort is warm-user only; this fallback makes the artifact serving-safe.
            scores = self.decomposition.popularity_z.copy()
        else:
            blend = self.popularity_blend
            scores = (1.0 - blend) * cf_scores + blend * self.decomposition.popularity_z

        seen = self.decomposition.history.seen_by_user.get(int(user_id), set())
        for item_id in seen:
            idx = self._item_index.get(int(item_id))
            if idx is not None:
                scores[idx] = -np.inf

        available = int(np.isfinite(scores).sum())
        if available == 0:
            return []
        take = min(int(k), available)
        if take == available:
            candidate_idx = np.flatnonzero(np.isfinite(scores))
        else:
            candidate_idx = np.argpartition(scores, -take)[-take:]
        candidate_items = self.decomposition.item_ids[candidate_idx]
        candidate_scores = scores[candidate_idx]
        order = np.lexsort((candidate_items, -candidate_scores))
        return [int(item) for item in candidate_items[order]]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            user_ids=self.decomposition.user_ids,
            item_ids=self.decomposition.item_ids,
            user_components=self.decomposition.user_components[:, : self.rank],
            singular_values=self.decomposition.singular_values[: self.rank],
            item_components=self.decomposition.item_components[:, : self.rank],
            popularity_z=self.decomposition.popularity_z,
            rank=np.array([self.rank], dtype=np.int32),
            popularity_blend=np.array([self.popularity_blend], dtype=np.float32),
        )
