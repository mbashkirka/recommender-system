from __future__ import annotations

import json
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from recsys_lab.config import Settings
from recsys_lab.evaluation.split import load_split


def _save_current(path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def build_m1_eda(settings: Settings) -> dict[str, Any]:
    train = load_split(settings, "train")
    validation = load_split(settings, "validation")
    test = load_split(settings, "test")
    full = pd.concat(
        [
            train.assign(split="train"),
            validation.assign(split="validation"),
            test.assign(split="test"),
        ],
        ignore_index=True,
    )

    report_dir = settings.paths.reports / "m1"
    report_dir.mkdir(parents=True, exist_ok=True)

    # 1) Split volume and positives.
    split_table = pd.DataFrame(
        {
            "interactions": [len(train), len(validation), len(test)],
            "positive_interactions": [
                int(train["is_positive"].astype(bool).sum()),
                int(validation["is_positive"].astype(bool).sum()),
                int(test["is_positive"].astype(bool).sum()),
            ],
        },
        index=["train", "validation", "test"],
    )
    split_table.to_csv(report_dir / "split_counts.csv")
    split_table.plot(kind="bar")
    plt.title("Global chronological split")
    plt.ylabel("Interactions")
    plt.xlabel("Split")
    _save_current(report_dir / "split_counts.png")

    # 2) Rating distribution.
    rating_counts = full["rating"].value_counts().sort_index()
    rating_counts.to_csv(report_dir / "rating_distribution.csv", header=["count"])
    rating_counts.plot(kind="bar")
    plt.title("MovieLens rating distribution")
    plt.xlabel("Rating")
    plt.ylabel("Interactions")
    _save_current(report_dir / "rating_distribution.png")

    # 3) Time trend.
    monthly = (
        full.assign(month=full["timestamp"].dt.tz_localize(None).dt.to_period("M").astype(str))
        .groupby("month", observed=True)
        .size()
        .rename("interactions")
    )
    monthly.to_csv(report_dir / "monthly_interactions.csv")
    monthly.plot()
    plt.title("Interactions over time")
    plt.xlabel("Month")
    plt.ylabel("Interactions")
    plt.xticks(rotation=45, ha="right")
    _save_current(report_dir / "monthly_interactions.png")

    # 4) Train item popularity long tail (positive events only).
    train_positive = train.loc[train["is_positive"].astype(bool)]
    item_popularity = (
        train_positive.groupby("item_id")
        .size()
        .sort_values(ascending=False)
        .rename("positive_events")
    )
    item_popularity.to_csv(report_dir / "train_item_popularity.csv")
    ranks = np.arange(1, len(item_popularity) + 1)
    plt.figure()
    plt.loglog(ranks, item_popularity.to_numpy())
    plt.title("Train item popularity long tail")
    plt.xlabel("Item popularity rank")
    plt.ylabel("Positive interactions")
    _save_current(report_dir / "train_item_popularity_long_tail.png")

    # 5) Train user activity.
    user_activity = train_positive.groupby("user_id").size().rename("positive_events")
    user_activity.to_csv(report_dir / "train_user_activity.csv")
    plt.figure()
    plt.hist(user_activity.to_numpy(), bins=50)
    plt.title("Train user activity distribution")
    plt.xlabel("Positive interactions per user")
    plt.ylabel("Users")
    _save_current(report_dir / "train_user_activity.png")

    summary = {
        "train_positive_events": len(train_positive),
        "train_active_users": int(train_positive["user_id"].nunique()),
        "train_active_items": int(train_positive["item_id"].nunique()),
        "median_positive_events_per_train_user": float(user_activity.median()),
        "p95_positive_events_per_train_user": float(user_activity.quantile(0.95)),
        "median_positive_events_per_train_item": float(item_popularity.median()),
        "p95_positive_events_per_train_item": float(item_popularity.quantile(0.95)),
        "top_1pct_item_share": float(
            item_popularity.head(max(1, int(np.ceil(len(item_popularity) * 0.01)))).sum()
            / item_popularity.sum()
        ),
        "generated_files": sorted(path.name for path in report_dir.iterdir() if path.is_file()),
    }
    (report_dir / "eda_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary
