from __future__ import annotations

import json
from pathlib import Path

from recsys_lab.serving.bundle import sha256_file
from recsys_lab.serving.integrity import verify_serving_artifact


def write_manifest(path: Path, fingerprint: str) -> None:
    path.write_text(
        json.dumps(
            {
                "artifact_sha256": fingerprint,
                "source_splits": ["train"],
                "validation_split_accessed": False,
                "test_split_accessed": False,
            }
        ),
        encoding="utf-8",
    )


def test_integrity_passes_for_exact_bundle(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle.npz"
    bundle.write_bytes(b"frozen-serving-artifact")
    manifest = tmp_path / "manifest.json"
    write_manifest(manifest, sha256_file(bundle))
    result = verify_serving_artifact(bundle, manifest)
    assert result["passed"] is True
    assert all(result["checks"].values())


def test_integrity_fails_after_bundle_tampering(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle.npz"
    bundle.write_bytes(b"original")
    manifest = tmp_path / "manifest.json"
    write_manifest(manifest, sha256_file(bundle))
    bundle.write_bytes(b"tampered")
    result = verify_serving_artifact(bundle, manifest)
    assert result["passed"] is False
    assert result["checks"]["sha256_matches_manifest"] is False


def test_integrity_fails_when_provenance_is_not_train_only(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle.npz"
    bundle.write_bytes(b"artifact")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "artifact_sha256": sha256_file(bundle),
                "source_splits": ["train", "validation"],
                "validation_split_accessed": True,
                "test_split_accessed": False,
            }
        ),
        encoding="utf-8",
    )
    result = verify_serving_artifact(bundle, manifest)
    assert result["passed"] is False
    assert result["checks"]["train_only_source"] is False
    assert result["checks"]["validation_not_accessed"] is False
