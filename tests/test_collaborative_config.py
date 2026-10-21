import pytest

from recsys_lab.config import CollaborativeConfig, load_settings


def test_project_collaborative_config_loads() -> None:
    settings = load_settings()
    assert settings.collaborative is not None
    assert settings.collaborative.ranks == (32, 64, 128)
    assert settings.collaborative.candidate_ks == (50, 100)


def test_collaborative_config_rejects_invalid_blend() -> None:
    cfg = CollaborativeConfig((32,), (1.0,), 10, (100,))
    with pytest.raises(ValueError):
        cfg.validate()
