from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from recsys_lab.config import Settings


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _required(path: Path, hint: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required M8 input missing: {path}. {hint}")
    return path


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_serving_bundle(settings: Settings) -> dict[str, Any]:
    """Build a self-contained online artifact from frozen M4/M5/M6 outputs.

    M8 reads train-derived/static data only to fill fields missing from earlier artifacts.
    It never reads validation or test and does not refit or retune any model.
    """
    m4_path = _required(
        settings.paths.artifacts / "m4" / "hybrid_selected.npz",
        "Run `python -m recsys_lab m4` first.",
    )
    m5_path = _required(
        settings.paths.artifacts / "m5" / "pairwise_ltr_selected.npz",
        "Run `python -m recsys_lab m5` first.",
    )
    m6_path = _required(
        settings.paths.artifacts / "m6" / "reranker_selected.json",
        "Run `python -m recsys_lab m6` first.",
    )
    m6_manifest_path = _required(
        settings.paths.artifacts / "m6" / "m6_manifest.json",
        "Run `python -m recsys_lab m6` first.",
    )
    train_path = _required(
        settings.paths.splits / "train.csv.gz",
        "Run `python -m recsys_lab m1` first.",
    )
    movies_path = _required(
        settings.paths.processed / "movies.csv.gz",
        "Run `python -m recsys_lab prepare` first.",
    )

    m4 = np.load(m4_path, allow_pickle=False)
    m5 = np.load(m5_path, allow_pickle=False)
    m6 = _read_json(m6_path)
    m6_manifest = _read_json(m6_manifest_path)

    train = pd.read_csv(train_path, usecols=["item_id", "is_positive"])
    positive_mask = (
        train["is_positive"]
        if train["is_positive"].dtype == bool
        else train["is_positive"].astype(str).str.lower().eq("true")
    )
    positive = train.loc[positive_mask]
    popularity = positive.groupby("item_id").size().to_dict()

    movies = pd.read_csv(movies_path, usecols=["item_id", "title", "genres"])
    movies["item_id"] = movies["item_id"].astype(int)
    movie_lookup = movies.drop_duplicates("item_id").set_index("item_id")
    all_item_ids = np.asarray(m4["all_item_ids"], dtype=np.int64)
    titles = np.asarray(
        [str(movie_lookup.at[int(i), "title"]) if int(i) in movie_lookup.index else "" for i in all_item_ids],
        dtype="U256",
    )
    genres = np.asarray(
        [str(movie_lookup.at[int(i), "genres"]) if int(i) in movie_lookup.index else "" for i in all_item_ids],
        dtype="U256",
    )
    popularity_counts = np.asarray([int(popularity.get(int(i), 0)) for i in all_item_ids], dtype=np.int64)

    gate = m6_manifest.get("deployment_gate", {})
    dev = m6_manifest.get("development_selected", {})
    deployed = m6_manifest.get("deployed", {})
    challenger_eligible = bool(gate.get("passed", False)) and (
        float(dev.get("diversity_weight", 0.0)) != 0.0
        or float(dev.get("novelty_weight", 0.0)) != 0.0
    )

    out_dir = settings.paths.artifacts / "m8"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "serving_bundle.npz"

    payload: dict[str, Any] = {name: np.asarray(m4[name]) for name in m4.files}
    payload.update(
        {
            "movie_titles": titles,
            "movie_genres": genres,
            "positive_popularity_counts": popularity_counts,
            "ltr_feature_names": np.asarray(m5["feature_names"]),
            "ltr_scaler_mean": np.asarray(m5["scaler_mean"], dtype=np.float64),
            "ltr_scaler_scale": np.asarray(m5["scaler_scale"], dtype=np.float64),
            "ltr_coefficients": np.asarray(m5["coefficients"], dtype=np.float64),
            "ltr_intercept": np.asarray(m5["intercept"], dtype=np.float64),
            "ltr_c": np.asarray(m5["c"], dtype=np.float64),
            "ltr_blend": np.asarray(m5["blend"], dtype=np.float64),
            "m6_deployed_diversity_weight": np.asarray([float(deployed.get("diversity_weight", 0.0))]),
            "m6_deployed_novelty_weight": np.asarray([float(deployed.get("novelty_weight", 0.0))]),
            "m6_development_diversity_weight": np.asarray([float(dev.get("diversity_weight", 0.0))]),
            "m6_development_novelty_weight": np.asarray([float(dev.get("novelty_weight", 0.0))]),
            "m6_pool_k": np.asarray([int(m6.get("pool_k", 50))], dtype=np.int32),
            "m6_primary_k": np.asarray([int(m6.get("primary_k", 10))], dtype=np.int32),
            "m6_max_per_genre": np.asarray([int(m6.get("max_per_genre", 4))], dtype=np.int32),
            "challenger_eligible": np.asarray([int(challenger_eligible)], dtype=np.int8),
        }
    )
    np.savez_compressed(out_path, **payload)
    fingerprint = sha256_file(out_path)

    manifest = {
        "milestone": "M8",
        "artifact": str(out_path.relative_to(settings.paths.root)),
        "artifact_sha256": fingerprint,
        "inputs": {
            "m4_sha256": sha256_file(m4_path),
            "m5_sha256": sha256_file(m5_path),
            "m6_sha256": sha256_file(m6_path),
        },
        "source_splits": ["train"],
        "validation_split_accessed": False,
        "test_split_accessed": False,
        "model_policy": {
            "control": "m6_deployed",
            "treatment": "m6_development_selected",
            "treatment_offline_eligible": challenger_eligible,
            "offline_gate_action": gate.get("action"),
        },
        "catalog_items": len(all_item_ids),
        "experiment": {
            "name": settings.experiment.default_name,
            "assignment_unit": "user",
            "sticky": True,
            "control_weight": settings.experiment.control_weight,
        },
    }
    (out_dir / "m8_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest
