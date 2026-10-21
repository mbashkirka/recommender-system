from pathlib import Path

from recsys_lab.config import DataConfig, ExperimentConfig, Paths, Settings, SplitConfig
from recsys_lab.data.ingest import prepare_processed
from recsys_lab.data.profile import build_profile


def _settings(tmp_path: Path) -> Settings:
    raw = tmp_path / "data" / "raw"
    processed = tmp_path / "data" / "processed"
    splits = tmp_path / "data" / "splits"
    reports = tmp_path / "reports"
    artifacts = tmp_path / "artifacts"
    source = raw / "ml-1m"
    source.mkdir(parents=True)
    processed.mkdir(parents=True)
    splits.mkdir(parents=True)
    reports.mkdir(parents=True)
    artifacts.mkdir(parents=True)

    (source / "ratings.dat").write_text(
        "1::10::5::978300760\n1::11::3::978302109\n2::10::4::978301968\n",
        encoding="latin-1",
    )
    (source / "movies.dat").write_text(
        "10::Movie A (2000)::Action|Comedy\n11::Movie B (1999)::Drama\n",
        encoding="latin-1",
    )
    (source / "users.dat").write_text(
        "1::F::25::1::00000\n2::M::35::2::11111\n",
        encoding="latin-1",
    )

    return Settings(
        name="test",
        seed=42,
        paths=Paths(tmp_path, raw, processed, splits, reports, artifacts),
        data=DataConfig(
            dataset="movielens-1m",
            url="https://example.invalid/ml-1m.zip",
            expected_md5="deadbeef",
            archive_name="ml-1m.zip",
            extracted_dir="ml-1m",
            implicit_positive_threshold=4.0,
        ),
        split=SplitConfig(
            train_fraction=0.8,
            validation_fraction=0.1,
            test_fraction=0.1,
            primary_eval_segment="warm_user_warm_item",
        ),
        experiment=ExperimentConfig(
            default_name="exp",
            default_salt="salt",
            control_weight=0.5,
            treatment_weight=0.5,
        ),
    )


def test_ingestion_and_profile_on_fixture(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    outputs = prepare_processed(settings)
    assert all(path.exists() for path in outputs.values())

    profile = build_profile(settings)
    assert profile["n_interactions"] == 3
    assert profile["n_users_in_ratings"] == 2
    assert profile["n_items_in_ratings"] == 2
    assert profile["positive_rate"] == 2 / 3
    assert profile["quality_checks"]["no_unknown_users"] is True
    assert profile["quality_checks"]["no_unknown_items"] is True
