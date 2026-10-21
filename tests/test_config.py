from recsys_lab.config import load_settings


def test_project_config_loads() -> None:
    settings = load_settings()
    assert settings.name == "recommender_system"
    assert settings.data.dataset == "movielens-1m"
    assert settings.data.implicit_positive_threshold == 4.0
    assert settings.experiment.control_weight + settings.experiment.treatment_weight == 1.0
    assert settings.evaluation is not None
    assert settings.evaluation.ks == (5, 10, 20)
    assert settings.evaluation.primary_k == 10
    assert settings.collaborative is not None
    assert settings.collaborative.tuning_k == 10


def test_split_config_loads() -> None:
    settings = load_settings()
    assert settings.split.train_fraction == 0.8
    assert settings.split.validation_fraction == 0.1
    assert settings.split.test_fraction == 0.1


def test_load_settings_honors_explicit_config_path(monkeypatch, tmp_path) -> None:
    import shutil
    from pathlib import Path

    source_config = Path(__file__).resolve().parents[1] / "config" / "project.toml"
    project = tmp_path / "runtime_project"
    config_dir = project / "config"
    config_dir.mkdir(parents=True)
    shutil.copy2(source_config, config_dir / "project.toml")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RECSYS_CONFIG", str(config_dir / "project.toml"))
    settings = load_settings()

    assert settings.paths.root == project.resolve()
    assert settings.paths.artifacts == project.resolve() / "artifacts"
    assert settings.paths.raw == project.resolve() / "data" / "raw"


def test_load_settings_uses_cwd_config_for_installed_layout(monkeypatch, tmp_path) -> None:
    import shutil
    from pathlib import Path

    source_config = Path(__file__).resolve().parents[1] / "config" / "project.toml"
    project = tmp_path / "app"
    config_dir = project / "config"
    config_dir.mkdir(parents=True)
    shutil.copy2(source_config, config_dir / "project.toml")

    monkeypatch.delenv("RECSYS_CONFIG", raising=False)
    monkeypatch.chdir(project)
    settings = load_settings()

    assert settings.paths.root == project.resolve()
