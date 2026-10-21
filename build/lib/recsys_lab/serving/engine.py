from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from recsys_lab.models.ranking import FEATURE_NAMES
from recsys_lab.serving.bundle import sha256_file


def _zscore(values: np.ndarray, finite_mask: np.ndarray | None = None) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if finite_mask is None:
        finite_mask = np.isfinite(values)
    out = np.full(values.shape, -np.inf, dtype=np.float64)
    if not finite_mask.any():
        return out
    selected = values[finite_mask]
    std = float(selected.std())
    out[finite_mask] = 0.0 if std <= 1e-12 else (selected - float(selected.mean())) / std
    return out


def _z(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    std = float(values.std())
    return np.zeros_like(values) if std <= 1e-12 else (values - float(values.mean())) / std


def _pairs_to_dict(users: np.ndarray, items: np.ndarray) -> dict[int, set[int]]:
    out: dict[int, set[int]] = {}
    for user, item in zip(users, items, strict=True):
        out.setdefault(int(user), set()).add(int(item))
    return out


def _genre_jaccard(features: np.ndarray, a: int, b: int) -> float:
    va, vb = features[a] > 0, features[b] > 0
    union = int(np.logical_or(va, vb).sum())
    return 0.0 if union == 0 else float(np.logical_and(va, vb).sum() / union)


@dataclass(frozen=True)
class Recommendation:
    item_id: int
    title: str
    genres: str
    rank: int
    score: float


class RecommendationEngine:
    """Inference-only recommender reconstructed from the M8 serving bundle."""

    def __init__(self, bundle_path: Path) -> None:
        self.bundle_path = Path(bundle_path)
        if not self.bundle_path.exists():
            raise FileNotFoundError(f"Serving bundle missing at {self.bundle_path}; run `python -m recsys_lab m8`.")
        data = np.load(self.bundle_path, allow_pickle=False)
        self.fingerprint = sha256_file(self.bundle_path)

        self.user_ids = np.asarray(data["user_ids"], dtype=np.int64)
        self.svd_item_ids = np.asarray(data["svd_item_ids"], dtype=np.int64)
        self.user_components = np.asarray(data["user_components"], dtype=np.float64)
        self.singular_values = np.asarray(data["singular_values"], dtype=np.float64)
        self.item_components = np.asarray(data["item_components"], dtype=np.float64)
        self.svd_popularity_z = np.asarray(data["svd_popularity_z"], dtype=np.float64)
        self.all_item_ids = np.asarray(data["all_item_ids"], dtype=np.int64)
        self.metadata_features = np.asarray(data["metadata_features"], dtype=np.float64)
        self.feature_names = tuple(str(x) for x in data["feature_names"].tolist())
        self.all_popularity_z = np.asarray(data["all_popularity_z"], dtype=np.float64)
        self.popularity_counts = np.asarray(data["positive_popularity_counts"], dtype=np.int64)
        self.movie_titles = np.asarray(data["movie_titles"]).astype(str)
        self.movie_genres = np.asarray(data["movie_genres"]).astype(str)

        self.rank = int(data["rank"][0])
        self.popularity_blend = float(data["popularity_blend"][0])
        self.metadata_blend = float(data["metadata_blend"][0])
        self.sparse_positive_threshold = int(data["sparse_positive_threshold"][0])
        self.sparse_metadata_multiplier = float(data["sparse_metadata_multiplier"][0])

        self.ltr_feature_names = tuple(str(x) for x in data["ltr_feature_names"].tolist())
        if self.ltr_feature_names != FEATURE_NAMES:
            raise ValueError("M8 bundle LTR feature contract does not match current code")
        self.ltr_mean = np.asarray(data["ltr_scaler_mean"], dtype=np.float64)
        self.ltr_scale = np.asarray(data["ltr_scaler_scale"], dtype=np.float64)
        self.ltr_coef = np.asarray(data["ltr_coefficients"], dtype=np.float64)
        self.ltr_blend = float(data["ltr_blend"][0])

        self.deployed_dw = float(data["m6_deployed_diversity_weight"][0])
        self.deployed_nw = float(data["m6_deployed_novelty_weight"][0])
        self.dev_dw = float(data["m6_development_diversity_weight"][0])
        self.dev_nw = float(data["m6_development_novelty_weight"][0])
        self.pool_k = int(data["m6_pool_k"][0])
        self.primary_k = int(data["m6_primary_k"][0])
        self.max_per_genre = int(data["m6_max_per_genre"][0])
        self.challenger_eligible = bool(int(data["challenger_eligible"][0]))

        self.user_index = {int(v): i for i, v in enumerate(self.user_ids)}
        self.all_item_index = {int(v): i for i, v in enumerate(self.all_item_ids)}
        self.svd_to_all = np.asarray([self.all_item_index.get(int(v), -1) for v in self.svd_item_ids], dtype=np.int64)
        self.seen_by_user = _pairs_to_dict(data["seen_user_ids"], data["seen_item_ids"])
        self.positive_by_user = _pairs_to_dict(data["positive_user_ids"], data["positive_item_ids"])
        self.warm_items = {int(x) for x in self.svd_item_ids}

    def route_for_user(self, user_id: int) -> str:
        if int(user_id) not in self.seen_by_user:
            return "cold_user_popularity"
        if len(self.positive_by_user.get(int(user_id), set())) <= self.sparse_positive_threshold:
            return "sparse_user_hybrid"
        return "warm_user_hybrid"

    def _metadata_profile(self, user_id: int) -> np.ndarray | None:
        rows = [self.all_item_index[x] for x in self.positive_by_user.get(int(user_id), set()) if x in self.all_item_index]
        if not rows:
            return None
        profile = self.metadata_features[rows].sum(axis=0)
        norm = float(profile.sum())
        return None if norm <= 1e-12 else profile / norm

    def _metadata_scores(self, user_id: int) -> np.ndarray | None:
        profile = self._metadata_profile(user_id)
        return None if profile is None else _zscore(self.metadata_features @ profile)

    def _warm_base_scores(self, user_id: int) -> np.ndarray | None:
        idx = self.user_index.get(int(user_id))
        if idx is None:
            return None
        r = self.rank
        raw = (self.user_components[idx, :r] * self.singular_values[:r]) @ self.item_components[:, :r].T
        cf_z = _zscore(raw)
        warm = (1.0 - self.popularity_blend) * cf_z + self.popularity_blend * self.svd_popularity_z
        scores = np.full(len(self.all_item_ids), -np.inf, dtype=np.float64)
        valid = self.svd_to_all >= 0
        scores[self.svd_to_all[valid]] = warm[valid]
        return scores

    def score_all(self, user_id: int) -> np.ndarray:
        user_id = int(user_id)
        base = self._warm_base_scores(user_id)
        metadata = self._metadata_scores(user_id)
        route = self.route_for_user(user_id)
        if route == "cold_user_popularity" or base is None:
            scores = self.all_popularity_z.copy()
        elif metadata is None or self.metadata_blend <= 0.0:
            scores = base.copy()
        else:
            blend = self.metadata_blend
            if route == "sparse_user_hybrid":
                blend = min(0.95, blend * self.sparse_metadata_multiplier)
            warm_mask = np.isfinite(base)
            scores = np.full(len(self.all_item_ids), -np.inf, dtype=np.float64)
            scores[warm_mask] = (1.0 - blend) * base[warm_mask] + blend * metadata[warm_mask]
            cold_mask = ~warm_mask & np.isfinite(metadata)
            scores[cold_mask] = blend * metadata[cold_mask] + (1.0 - blend) * self.all_popularity_z[cold_mask]
        for item_id in self.seen_by_user.get(user_id, set()):
            idx = self.all_item_index.get(int(item_id))
            if idx is not None:
                scores[idx] = -np.inf
        return scores

    def _top_candidates(self, user_id: int, k: int) -> tuple[list[int], np.ndarray, np.ndarray | None, np.ndarray | None]:
        scores = self.score_all(user_id)
        finite = np.flatnonzero(np.isfinite(scores))
        take = min(int(k), len(finite))
        if take == 0:
            return [], scores, None, None
        if take < len(finite):
            idx = finite[np.argpartition(scores[finite], -take)[-take:]]
        else:
            idx = finite
        order = np.lexsort((self.all_item_ids[idx], -scores[idx]))
        items = [int(x) for x in self.all_item_ids[idx][order]]
        return items, scores, self._warm_base_scores(user_id), self._metadata_scores(user_id)

    def _ltr_order(self, user_id: int, candidates: list[int], retrieval: np.ndarray, warm: np.ndarray | None, metadata: np.ndarray | None) -> list[int]:
        if self.ltr_blend == 0.0 or not candidates:
            return candidates
        profile = self._metadata_profile(user_id)
        seen_count = len(self.seen_by_user.get(int(user_id), set()))
        positive_count = len(self.positive_by_user.get(int(user_id), set()))
        rows: list[list[float]] = []
        for rank, item_id in enumerate(candidates, start=1):
            idx = self.all_item_index[item_id]
            meta = 0.0 if metadata is None or not np.isfinite(metadata[idx]) else float(metadata[idx])
            warm_score = 0.0 if warm is None or not np.isfinite(warm[idx]) else float(warm[idx])
            overlap = 0.0 if profile is None else float(self.metadata_features[idx] @ profile)
            rows.append([
                float(retrieval[idx]) if np.isfinite(retrieval[idx]) else -8.0,
                1.0 / rank,
                warm_score,
                meta,
                float(self.all_popularity_z[idx]) if np.isfinite(self.all_popularity_z[idx]) else 0.0,
                float(np.log1p(self.popularity_counts[idx])),
                float(np.log1p(seen_count)),
                float(np.log1p(positive_count)),
                overlap,
                float(item_id not in self.warm_items),
            ])
        x = np.asarray(rows, dtype=np.float64)
        scale = np.where(self.ltr_scale == 0, 1.0, self.ltr_scale)
        ltr = ((x - self.ltr_mean) / scale) @ self.ltr_coef
        base = -np.arange(1, len(candidates) + 1, dtype=np.float64)
        score = (1.0 - self.ltr_blend) * _z(base) + self.ltr_blend * _z(ltr)
        return [candidates[i] for i in np.lexsort((np.asarray(candidates), np.arange(len(candidates)), -score))]

    def _rerank(self, ordered: list[int], k: int, diversity_weight: float, novelty_weight: float) -> list[int]:
        if not ordered or (diversity_weight == 0.0 and novelty_weight == 0.0):
            return ordered
        pool = ordered[: min(self.pool_k, len(ordered))]
        tail = ordered[len(pool):]
        relevance = np.ones(len(pool)) if len(pool) == 1 else 1.0 - np.arange(len(pool)) / (len(pool) - 1)
        total = float(self.popularity_counts.sum()) + max(len(self.all_item_ids), 1)
        raw_novelty = np.asarray([-np.log2((self.popularity_counts[self.all_item_index[i]] + 1.0) / total) for i in pool])
        novelty = _z(raw_novelty)
        positions = {item: pos for pos, item in enumerate(pool)}
        selected: list[int] = []
        genre_counts: dict[int, int] = {}
        remaining = set(pool)
        target = min(k, len(pool))
        while remaining and len(selected) < target:
            scored: list[tuple[float, int, int, bool]] = []
            for item in remaining:
                pos = positions[item]
                idx = self.all_item_index[item]
                max_sim = max((_genre_jaccard(self.metadata_features, idx, self.all_item_index[x]) for x in selected), default=0.0)
                nz = np.flatnonzero(self.metadata_features[idx] > 0)
                primary = int(nz[0]) if len(nz) else None
                violates = bool(self.max_per_genre > 0 and primary is not None and genre_counts.get(primary, 0) >= self.max_per_genre)
                utility = relevance[pos] - diversity_weight * max_sim + novelty_weight * novelty[pos]
                scored.append((float(utility), -pos, int(item), violates))
            feasible = [x for x in scored if not x[3]]
            best = max(feasible if feasible else scored, key=lambda x: (x[0], x[1], -x[2]))
            item = int(best[2])
            selected.append(item)
            remaining.remove(item)
            idx = self.all_item_index[item]
            nz = np.flatnonzero(self.metadata_features[idx] > 0)
            if len(nz):
                primary = int(nz[0])
                genre_counts[primary] = genre_counts.get(primary, 0) + 1
        selected_set = set(selected)
        return selected + [x for x in pool if x not in selected_set] + tail

    def recommend(self, user_id: int, k: int, policy: str = "control", candidate_k: int = 100) -> tuple[list[Recommendation], dict[str, Any]]:
        if k <= 0:
            return [], {"route": self.route_for_user(user_id), "policy": policy}
        candidate_k = max(int(candidate_k), int(k))
        candidates, retrieval, warm, metadata = self._top_candidates(user_id, candidate_k)
        ordered = self._ltr_order(user_id, candidates, retrieval, warm, metadata)
        if policy == "treatment" and self.challenger_eligible:
            dw, nw = self.dev_dw, self.dev_nw
            effective_policy = "m6_development_selected"
        else:
            dw, nw = self.deployed_dw, self.deployed_nw
            effective_policy = "m6_deployed"
        ordered = self._rerank(ordered, k, dw, nw)
        top = ordered[:k]
        recs = [
            Recommendation(
                item_id=item,
                title=self.movie_titles[self.all_item_index[item]],
                genres=self.movie_genres[self.all_item_index[item]],
                rank=rank,
                score=float(retrieval[self.all_item_index[item]]) if np.isfinite(retrieval[self.all_item_index[item]]) else -8.0,
            )
            for rank, item in enumerate(top, start=1)
        ]
        return recs, {
            "route": self.route_for_user(user_id),
            "requested_policy": policy,
            "effective_policy": effective_policy,
            "treatment_offline_eligible": self.challenger_eligible,
            "ltr_blend": self.ltr_blend,
            "diversity_weight": dw,
            "novelty_weight": nw,
        }
