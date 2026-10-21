from __future__ import annotations

from pathlib import Path

import pandas as pd

from recsys_lab.config import Settings

RATINGS_COLUMNS = ["user_id", "item_id", "rating", "timestamp"]
MOVIES_COLUMNS = ["item_id", "title", "genres"]
USERS_COLUMNS = ["user_id", "gender", "age", "occupation", "zip_code"]


def _read_double_colon(path: Path, names: list[str]) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep="::",
        engine="python",
        names=names,
        header=None,
        encoding="latin-1",
    )


def prepare_processed(settings: Settings) -> dict[str, Path]:
    source = settings.paths.raw / settings.data.extracted_dir
    if not source.exists():
        raise FileNotFoundError(
            f"Raw dataset missing at {source}. Run `python -m recsys_lab download` first."
        )

    settings.paths.processed.mkdir(parents=True, exist_ok=True)

    ratings = _read_double_colon(source / "ratings.dat", RATINGS_COLUMNS)
    movies = _read_double_colon(source / "movies.dat", MOVIES_COLUMNS)
    users = _read_double_colon(source / "users.dat", USERS_COLUMNS)

    ratings["timestamp"] = pd.to_datetime(ratings["timestamp"], unit="s", utc=True)
    ratings["is_positive"] = ratings["rating"].ge(settings.data.implicit_positive_threshold)
    movies["genres"] = movies["genres"].fillna("")

    outputs = {
        "ratings": settings.paths.processed / "ratings.csv.gz",
        "movies": settings.paths.processed / "movies.csv.gz",
        "users": settings.paths.processed / "users.csv.gz",
    }
    ratings.to_csv(outputs["ratings"], index=False, compression="gzip")
    movies.to_csv(outputs["movies"], index=False, compression="gzip")
    users.to_csv(outputs["users"], index=False, compression="gzip")

    return outputs
