from __future__ import annotations

import json
from typing import Any

import pandas as pd

from recsys_lab.config import Settings

SPLIT_NAMES = ("train", "validation", "test")


def _read_ratings(settings: Settings) -> pd.DataFrame:
    path = settings.paths.processed / "ratings.csv.gz"
    if not path.exists():
        raise FileNotFoundError(f"Processed ratings missing at {path}. Run `prepare` first.")
    ratings = pd.read_csv(path, parse_dates=["timestamp"])
    ratings["timestamp"] = pd.to_datetime(ratings["timestamp"], utc=True)
    return ratings.sort_values(["timestamp", "user_id", "item_id"], kind="mergesort").reset_index(
        drop=True
    )


def chronological_cutoffs(
    ratings: pd.DataFrame,
    train_fraction: float,
    validation_fraction: float,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return global time cutoffs based only on event order.

    All events sharing a boundary timestamp stay on the earlier side of that boundary,
    preventing identical timestamps from being split across periods.
    """
    if ratings.empty:
        raise ValueError("Cannot split an empty ratings table")
    timestamps = pd.to_datetime(ratings["timestamp"], utc=True).sort_values().reset_index(drop=True)
    n_rows = len(timestamps)
    train_index = min(max(int(n_rows * train_fraction) - 1, 0), n_rows - 1)
    validation_index = min(
        max(int(n_rows * (train_fraction + validation_fraction)) - 1, train_index),
        n_rows - 1,
    )
    return timestamps.iloc[train_index], timestamps.iloc[validation_index]


def assign_periods(
    ratings: pd.DataFrame,
    train_cutoff: pd.Timestamp,
    validation_cutoff: pd.Timestamp,
) -> pd.Series:
    timestamps = pd.to_datetime(ratings["timestamp"], utc=True)
    periods = pd.Series("test", index=ratings.index, dtype="string")
    periods.loc[timestamps <= validation_cutoff] = "validation"
    periods.loc[timestamps <= train_cutoff] = "train"
    return periods


def _segment_stats(eval_frame: pd.DataFrame, train_frame: pd.DataFrame) -> dict[str, Any]:
    positive = eval_frame.loc[eval_frame["is_positive"].astype(bool)].copy()
    train_users = set(train_frame["user_id"].unique())
    train_items = set(train_frame["item_id"].unique())

    if positive.empty:
        return {
            "positive_events": 0,
            "warm_user_rate": 0.0,
            "warm_item_rate": 0.0,
            "warm_user_warm_item_rate": 0.0,
            "warm_user_warm_item_events": 0,
            "cold_user_events": 0,
            "cold_item_events": 0,
        }

    warm_user = positive["user_id"].isin(train_users)
    warm_item = positive["item_id"].isin(train_items)
    fully_warm = warm_user & warm_item
    return {
        "positive_events": len(positive),
        "warm_user_rate": float(warm_user.mean()),
        "warm_item_rate": float(warm_item.mean()),
        "warm_user_warm_item_rate": float(fully_warm.mean()),
        "warm_user_warm_item_events": int(fully_warm.sum()),
        "cold_user_events": int((~warm_user).sum()),
        "cold_item_events": int((~warm_item).sum()),
    }


def _frame_stats(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": len(frame),
        "positive_rows": int(frame["is_positive"].astype(bool).sum()),
        "positive_rate": float(frame["is_positive"].astype(bool).mean()) if len(frame) else 0.0,
        "users": int(frame["user_id"].nunique()),
        "items": int(frame["item_id"].nunique()),
        "min_timestamp": frame["timestamp"].min().isoformat() if len(frame) else None,
        "max_timestamp": frame["timestamp"].max().isoformat() if len(frame) else None,
    }


def build_temporal_splits(settings: Settings) -> dict[str, Any]:
    ratings = _read_ratings(settings)
    train_cutoff, validation_cutoff = chronological_cutoffs(
        ratings,
        train_fraction=settings.split.train_fraction,
        validation_fraction=settings.split.validation_fraction,
    )
    periods = assign_periods(ratings, train_cutoff, validation_cutoff)

    frames = {
        "train": ratings.loc[periods.eq("train")].copy(),
        "validation": ratings.loc[periods.eq("validation")].copy(),
        "test": ratings.loc[periods.eq("test")].copy(),
    }

    settings.paths.splits.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}
    for name, frame in frames.items():
        path = settings.paths.splits / f"{name}.csv.gz"
        frame.to_csv(path, index=False, compression="gzip")
        outputs[name] = str(path.relative_to(settings.paths.root))

        positive_path = settings.paths.splits / f"{name}_positive.csv.gz"
        frame.loc[frame["is_positive"].astype(bool)].to_csv(
            positive_path, index=False, compression="gzip"
        )
        outputs[f"{name}_positive"] = str(positive_path.relative_to(settings.paths.root))

    train = frames["train"]
    validation = frames["validation"]
    test = frames["test"]

    all_partitioned = sum(len(frame) for frame in frames.values()) == len(ratings)
    disjoint_indices = (
        set(train.index).isdisjoint(validation.index)
        and set(train.index).isdisjoint(test.index)
        and set(validation.index).isdisjoint(test.index)
    )
    temporal_order = (
        train["timestamp"].max() < validation["timestamp"].min()
        and validation["timestamp"].max() < test["timestamp"].min()
    )

    manifest: dict[str, Any] = {
        "dataset": settings.data.dataset,
        "protocol": "global_chronological_80_10_10",
        "implicit_positive_threshold": settings.data.implicit_positive_threshold,
        "primary_eval_segment": settings.split.primary_eval_segment,
        "fractions_requested": {
            "train": settings.split.train_fraction,
            "validation": settings.split.validation_fraction,
            "test": settings.split.test_fraction,
        },
        "cutoffs": {
            "train_end": train_cutoff.isoformat(),
            "validation_end": validation_cutoff.isoformat(),
        },
        "splits": {name: _frame_stats(frame) for name, frame in frames.items()},
        "evaluation_segments": {
            "validation": _segment_stats(validation, train),
            "test": _segment_stats(test, train),
        },
        "leakage_checks": {
            "all_rows_partitioned": bool(all_partitioned),
            "split_rows_disjoint": bool(disjoint_indices),
            "strict_global_time_order": bool(temporal_order),
            "train_cutoff_respected": bool((train["timestamp"] <= train_cutoff).all()),
            "validation_after_train": bool((validation["timestamp"] > train_cutoff).all()),
            "test_after_validation": bool((test["timestamp"] > validation_cutoff).all()),
        },
        "outputs": outputs,
    }

    settings.paths.reports.mkdir(parents=True, exist_ok=True)
    manifest_path = settings.paths.reports / "m1_split_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


def load_split(settings: Settings, name: str) -> pd.DataFrame:
    if name not in SPLIT_NAMES:
        raise ValueError(f"Unknown split {name!r}; expected one of {SPLIT_NAMES}")
    path = settings.paths.splits / f"{name}.csv.gz"
    if not path.exists():
        raise FileNotFoundError(f"Missing split at {path}. Run `python -m recsys_lab m1`.")
    return pd.read_csv(path, parse_dates=["timestamp"])
