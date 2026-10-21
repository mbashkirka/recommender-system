import pandas as pd

from recsys_lab.models.baselines import (
    GenreAffinityRecommender,
    PopularityRecommender,
    RandomRecommender,
    build_history,
)


def _train() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2, 3, 3, 4],
            "item_id": [10, 11, 10, 12, 10, 13, 14],
            "rating": [5, 2, 5, 5, 4, 5, 5],
            "is_positive": [True, False, True, True, True, True, True],
        }
    )


def _movies() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "item_id": [10, 11, 12, 13, 14],
            "title": ["A", "B", "C", "D", "E"],
            "genres": ["Action", "Drama", "Action", "Drama", "Comedy"],
        }
    )


def test_popularity_excludes_seen_items() -> None:
    history = build_history(_train())
    model = PopularityRecommender(history)
    recs = model.recommend(1, 3)
    assert 10 not in recs
    assert 11 not in recs
    assert recs[0] in {12, 13, 14}


def test_random_is_deterministic_per_user() -> None:
    history = build_history(_train())
    model = RandomRecommender(history, seed=42)
    assert model.recommend(1, 3) == model.recommend(1, 3)


def test_random_excludes_seen_items() -> None:
    history = build_history(_train())
    model = RandomRecommender(history, seed=42)
    recs = model.recommend(1, 10)
    assert set(recs).isdisjoint({10, 11})


def test_genre_affinity_prefers_matching_genre() -> None:
    history = build_history(_train())
    model = GenreAffinityRecommender(history, _movies())
    recs = model.recommend(1, 3)
    # User 1's only positive train item is Action item 10; item 12 is unseen Action.
    assert recs[0] == 12


def test_genre_affinity_falls_back_for_unknown_user() -> None:
    history = build_history(_train())
    genre = GenreAffinityRecommender(history, _movies())
    popularity = PopularityRecommender(history)
    assert genre.recommend(999, 3) == popularity.recommend(999, 3)
