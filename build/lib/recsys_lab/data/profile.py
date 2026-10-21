from __future__ import annotations

import json
from typing import Any

import pandas as pd

from recsys_lab.config import Settings


def _load(settings: Settings) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ratings_path = settings.paths.processed / "ratings.csv.gz"
    movies_path = settings.paths.processed / "movies.csv.gz"
    users_path = settings.paths.processed / "users.csv.gz"
    missing = [p for p in (ratings_path, movies_path, users_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Processed data missing: " + ", ".join(str(p) for p in missing) + ". Run `prepare`."
        )
    return (
        pd.read_csv(ratings_path, parse_dates=["timestamp"]),
        pd.read_csv(movies_path),
        pd.read_csv(users_path),
    )


def build_profile(settings: Settings) -> dict[str, Any]:
    ratings, movies, users = _load(settings)

    duplicate_interactions = int(ratings.duplicated(["user_id", "item_id"]).sum())
    unknown_users = int((~ratings["user_id"].isin(users["user_id"])).sum())
    unknown_items = int((~ratings["item_id"].isin(movies["item_id"])).sum())

    n_users = int(ratings["user_id"].nunique())
    n_items = int(ratings["item_id"].nunique())
    n_interactions = len(ratings)
    density = n_interactions / (n_users * n_items)

    profile: dict[str, Any] = {
        "dataset": settings.data.dataset,
        "n_interactions": n_interactions,
        "n_users_in_ratings": n_users,
        "n_items_in_ratings": n_items,
        "n_users_catalog": len(users),
        "n_items_catalog": len(movies),
        "matrix_density": density,
        "matrix_sparsity": 1.0 - density,
        "rating_mean": float(ratings["rating"].mean()),
        "positive_threshold": settings.data.implicit_positive_threshold,
        "positive_rate": float(ratings["is_positive"].mean()),
        "min_timestamp": ratings["timestamp"].min().isoformat(),
        "max_timestamp": ratings["timestamp"].max().isoformat(),
        "duplicate_user_item_pairs": duplicate_interactions,
        "unknown_user_interactions": unknown_users,
        "unknown_item_interactions": unknown_items,
        "quality_checks": {
            "enough_interactions": n_interactions >= 900_000,
            "enough_users": n_users >= 5_000,
            "enough_items": n_items >= 3_000,
            "rating_range_ok": bool(ratings["rating"].between(1, 5).all()),
            "no_unknown_users": unknown_users == 0,
            "no_unknown_items": unknown_items == 0,
            "no_duplicate_user_item_pairs": duplicate_interactions == 0,
        },
    }
    return profile


def write_profile(settings: Settings) -> dict[str, Any]:
    settings.paths.reports.mkdir(parents=True, exist_ok=True)
    profile = build_profile(settings)
    output = settings.paths.reports / "m0_data_profile.json"
    output.write_text(json.dumps(profile, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return profile
