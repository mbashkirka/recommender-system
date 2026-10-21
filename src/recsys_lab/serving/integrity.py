from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from recsys_lab.serving.bundle import sha256_file


def verify_serving_artifact(bundle_path: Path, manifest_path: Path) -> dict[str, Any]:
    """Verify that the serving bundle is the exact artifact recorded by M8.

    The check is intentionally independent from model loading so readiness can fail closed when
    the bytes on disk no longer match the recorded fingerprint.
    """
    bundle_path = Path(bundle_path)
    manifest_path = Path(manifest_path)
    checks: dict[str, bool] = {
        "bundle_exists": bundle_path.exists(),
        "manifest_exists": manifest_path.exists(),
        "sha256_matches_manifest": False,
        "train_only_source": False,
        "validation_not_accessed": False,
        "test_not_accessed": False,
    }
    expected: str | None = None
    actual: str | None = None
    manifest: dict[str, Any] = {}

    if bundle_path.exists():
        actual = sha256_file(bundle_path)
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            manifest = {}
        expected = manifest.get("artifact_sha256")
        checks["sha256_matches_manifest"] = bool(actual and expected and actual == expected)
        checks["train_only_source"] = manifest.get("source_splits") == ["train"]
        checks["validation_not_accessed"] = manifest.get("validation_split_accessed") is False
        checks["test_not_accessed"] = manifest.get("test_split_accessed") is False

    passed = all(checks.values())
    return {
        "passed": passed,
        "checks": checks,
        "expected_sha256": expected,
        "actual_sha256": actual,
        "bundle": str(bundle_path),
        "manifest": str(manifest_path),
    }
