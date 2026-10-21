import pytest

from recsys_lab.experiments.assignment import assign_two_arm, stable_bucket


def test_stable_bucket_is_deterministic() -> None:
    first = stable_bucket("exp", 123, "salt")
    second = stable_bucket("exp", 123, "salt")
    assert first == second
    assert 0.0 <= first < 1.0


def test_assignment_is_sticky() -> None:
    a = assign_two_arm("exp", "user-1", "salt", control_weight=0.5)
    b = assign_two_arm("exp", "user-1", "salt", control_weight=0.5)
    assert a.variant == b.variant
    assert a.bucket == b.bucket


def test_assignment_has_two_variants() -> None:
    variants = {
        assign_two_arm("exp", f"user-{i}", "salt", control_weight=0.5).variant
        for i in range(200)
    }
    assert variants == {"control", "treatment"}


def test_invalid_control_weight_rejected() -> None:
    with pytest.raises(ValueError):
        assign_two_arm("exp", "u", "salt", control_weight=1.0)
