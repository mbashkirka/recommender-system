from __future__ import annotations

import json
from pathlib import Path
from threading import Lock
from typing import Any


class JsonlEventStore:
    """Small local append-only event store for M8 development serving.

    An in-memory impression index is rebuilt from the exposure log at startup so outcome requests
    can be rejected when they reference an unknown impression or a different user.
    """

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.exposure_path = self.directory / "exposures.jsonl"
        self.outcome_path = self.directory / "outcomes.jsonl"
        self._lock = Lock()
        self._impressions: dict[str, str] = {}
        self._load_index()

    def _load_index(self) -> None:
        if not self.exposure_path.exists():
            return
        for line in self.exposure_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            impression_id = payload.get("impression_id")
            unit_id = payload.get("unit_id")
            if impression_id is not None and unit_id is not None:
                self._impressions[str(impression_id)] = str(unit_id)

    def _append(self, path: Path, payload: dict[str, Any]) -> None:
        line = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
        with self._lock, path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
            fh.flush()

    def log_exposure(self, payload: dict[str, Any]) -> None:
        self._append(self.exposure_path, payload)
        self._impressions[str(payload["impression_id"])] = str(payload["unit_id"])

    def validate_outcome_reference(self, impression_id: str, user_id: int) -> str | None:
        expected_user = self._impressions.get(str(impression_id))
        if expected_user is None:
            return "unknown_impression"
        if expected_user != str(user_id):
            return "user_mismatch"
        return None

    def log_outcome(self, payload: dict[str, Any]) -> None:
        self._append(self.outcome_path, payload)
