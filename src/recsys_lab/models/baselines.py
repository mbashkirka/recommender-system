from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FittedHistory:
    catalog: np.ndarray
    seen_by_user: dict[int, set[int]]
    positive_by_user: dict[int, set[int]]
    popularity_counts: dict[int, int]
    popularity_order: np.ndarray


def build_history(train: pd.DataFrame) -> FittedHistory:
    if train.empty:
        raise ValueError("Training interactions are empty")

    catalog = np.sort(train["item_id"].astype(int).unique())
    seen_by_user = {
        int(user_id): set(group["item_id"].astype(int))
        for user_id, group in train.groupby("user_id", sort=False)
    }
    positive = train.loc[train["is_positive"].astype(bool)]
    positive_by_user = {
        int(user_id): set(group["item_id"].astype(int))
        for user_id, group in positive.groupby("user_id", sort=False)
    }
    counts = positive.groupby("item_id").size().astype(int)
    popularity_counts = {int(item): int(count) for item, count in counts.items()}
    popularity_order = np.array(
        sorted(
            (int(item) for item in catalog),
            key=lambda item: (-popularity_counts.get(item, 0), item),
        ),
        dtype=np.int64,
    )
    return FittedHistory(
        catalog=catalog.astype(np.int64),
        seen_by_user=seen_by_user,
        positive_by_user=positive_by_user,
        popularity_counts=popularity_counts,
        popularity_order=popularity_order,
    )


class PopularityRecommender:
    name = "popularity"

    def __init__(self, history: FittedHistory):
        self.history = history

    def recommend(self, user_id: int, k: int) -> list[int]:
        seen = self.history.seen_by_user.get(int(user_id), set())
        recs: list[int] = []
        for item in self.history.popularity_order:
            item_id = int(item)
            if item_id not in seen:
                recs.append(item_id)
                if len(recs) == k:
                    break
        return recs


class RandomRecommender:
    name = "random"

    def __init__(self, history: FittedHistory, seed: int = 42):
        self.history = history
        self.seed = int(seed)

    def _user_seed(self, user_id: int) -> int:
        payload = f"{self.seed}:{int(user_id)}".encode()
        digest = hashlib.sha256(payload).digest()
        return int.from_bytes(digest[:8], byteorder="big", signed=False)

    def recommend(self, user_id: int, k: int) -> list[int]:
        seen = self.history.seen_by_user.get(int(user_id), set())
        if seen:
            candidates = np.array(
                [item for item in self.history.catalog if int(item) not in seen], dtype=np.int64
            )
        else:
            candidates = self.history.catalog.copy()
        if len(candidates) == 0:
            return []
        rng = np.random.default_rng(self._user_seed(int(user_id)))
        take = min(k, len(candidates))
        selected = rng.choice(candidates, size=take, replace=False)
        return [int(item) for item in selected]


class GenreAffinityRecommender:
    """Simple personalized content baseline derived only from positive train history.

    User profiles are distributions over MovieLens genres. Candidate item score is the dot product
    between the user profile and the item's genre vector normalized by its number of genres. A
    tiny train-popularity term provides deterministic tie breaking without overwhelming affinity.
    Users without positive history fall back to global popularity.
    """

    name = "genre_affinity"

    def __init__(
        self,
        history: FittedHistory,
        movies: pd.DataFrame,
        popularity_tie_weight: float = 1e-6,
    ):
        self.history = history
        self.popularity_tie_weight = float(popularity_tie_weight)
        movie_map = movies.copy()
        movie_map["item_id"] = movie_map["item_id"].astype(int)
        movie_map["genres"] = movie_map["genres"].fillna("")
        movie_map = movie_map.drop_duplicates("item_id").set_index("item_id")

        genres = sorted(
            {
                genre
                for value in movie_map["genres"].astype(str)
                for genre in value.split("|")
                if genre and genre != "(no genres listed)"
            }
        )
        self.genres = genres
        genre_index = {genre: idx for idx, genre in enumerate(genres)}
        matrix = np.zeros((len(history.catalog), len(genres)), dtype=np.float32)
        for row, item in enumerate(history.catalog):
            item_id = int(item)
            if item_id not in movie_map.index:
                continue
            item_genres = [
                genre
                for genre in str(movie_map.at[item_id, "genres"]).split("|")
                if genre in genre_index
            ]
            if not item_genres:
                continue
            weight = 1.0 / len(item_genres)
            for genre in item_genres:
                matrix[row, genre_index[genre]] = weight
        self.item_features = matrix
        self.catalog_index = {int(item): idx for idx, item in enumerate(history.catalog)}

        max_pop = max(history.popularity_counts.values(), default=1)
        self.popularity_prior = np.array(
            [history.popularity_counts.get(int(item), 0) / max_pop for item in history.catalog],
            dtype=np.float32,
        )
        self._popularity_fallback = PopularityRecommender(history)

    def _profile(self, user_id: int) -> np.ndarray | None:
        positive_items = self.history.positive_by_user.get(int(user_id), set())
        rows = [self.catalog_index[item] for item in positive_items if item in self.catalog_index]
        if not rows:
            return None
        profile = self.item_features[rows].sum(axis=0)
        norm = float(profile.sum())
        if norm <= 0:
            return None
        return profile / norm

    def recommend(self, user_id: int, k: int) -> list[int]:
        profile = self._profile(int(user_id))
        if profile is None:
            return self._popularity_fallback.recommend(int(user_id), k)

        scores = self.item_features @ profile
        scores = scores + self.popularity_tie_weight * self.popularity_prior
        seen = self.history.seen_by_user.get(int(user_id), set())
        if seen:
            for item in seen:
                idx = self.catalog_index.get(int(item))
                if idx is not None:
                    scores[idx] = -np.inf

        available = int(np.isfinite(scores).sum())
        if available == 0:
            return []
        take = min(k, available)
        # argpartition avoids sorting the full catalog; lexsort gives stable score/item order.
        candidate_idx = np.argpartition(scores, -take)[-take:]
        candidate_items = self.history.catalog[candidate_idx]
        candidate_scores = scores[candidate_idx]
        order = np.lexsort((candidate_items, -candidate_scores))
        return [int(item) for item in candidate_items[order]]


def build_recommenders(
    train: pd.DataFrame,
    movies: pd.DataFrame,
    seed: int,
) -> tuple[FittedHistory, dict[str, object]]:
    history = build_history(train)
    recommenders: dict[str, object] = {
        "random": RandomRecommender(history, seed=seed),
        "popularity": PopularityRecommender(history),
        "genre_affinity": GenreAffinityRecommender(history, movies=movies),
    }
    return history, recommenders
