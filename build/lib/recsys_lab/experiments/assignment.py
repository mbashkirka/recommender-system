from __future__ import annotations

import hashlib
from dataclasses import dataclass


@dataclass(frozen=True)
class Assignment:
    experiment: str
    unit_id: str
    variant: str
    bucket: float


def stable_bucket(experiment: str, unit_id: str | int, salt: str) -> float:
    """Map an experiment unit deterministically to [0, 1)."""
    key = f"{salt}:{experiment}:{unit_id}".encode()
    digest = hashlib.sha256(key).digest()
    integer = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return integer / 2**64


def assign_two_arm(
    experiment: str,
    unit_id: str | int,
    salt: str,
    control_weight: float = 0.5,
) -> Assignment:
    if not 0.0 < control_weight < 1.0:
        raise ValueError("control_weight must be strictly between 0 and 1")
    bucket = stable_bucket(experiment=experiment, unit_id=unit_id, salt=salt)
    variant = "control" if bucket < control_weight else "treatment"
    return Assignment(
        experiment=experiment,
        unit_id=str(unit_id),
        variant=variant,
        bucket=bucket,
    )
