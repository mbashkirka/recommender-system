from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient
from test_serving_engine import make_bundle

from recsys_lab.config import load_settings
from recsys_lab.serving.app import create_app
from recsys_lab.serving.engine import RecommendationEngine
from recsys_lab.serving.events import JsonlEventStore
from recsys_lab.serving.service import RecommendationService


def make_service(tmp_path: Path, eligible: bool = False) -> RecommendationService:
    settings = load_settings()
    engine = RecommendationEngine(make_bundle(tmp_path / "bundle.npz", challenger_eligible=eligible))
    return RecommendationService(settings, engine, JsonlEventStore(tmp_path / "events"))


def test_health_ready_and_model_info(tmp_path: Path) -> None:
    service = make_service(tmp_path)
    with TestClient(create_app(service)) as client:
        assert client.get("/health").json() == {"status": "ok"}
        ready = client.get("/ready")
        assert ready.status_code == 200
        assert ready.json()["status"] == "ready"
        info = client.get("/v1/model-info")
        assert info.status_code == 200
        assert info.json()["catalog_items"] == 5


def test_recommend_logs_exposure(tmp_path: Path) -> None:
    service = make_service(tmp_path, eligible=False)
    with TestClient(create_app(service)) as client:
        response = client.post("/v1/recommend", json={"user_id": 1, "k": 3})
        assert response.status_code == 200
        body = response.json()
        assert len(body["recommendations"]) == 3
        assert body["challenger_offline_eligible"] is False
        lines = (tmp_path / "events" / "exposures.jsonl").read_text().strip().splitlines()
        event = json.loads(lines[-1])
        assert event["impression_id"] == body["impression_id"]
        assert event["artifact_fingerprint"] == body["artifact_fingerprint"]


def test_offline_ineligible_treatment_is_not_served(tmp_path: Path) -> None:
    service = make_service(tmp_path, eligible=False)
    # Find a treatment-assigned user deterministically rather than assuming user 1.
    treatment_user = next(
        user for user in range(1, 1000)
        if service.recommend(user, 1)["assigned_variant"] == "treatment"
    )
    body = service.recommend(treatment_user, 2)
    assert body["assigned_variant"] == "treatment"
    assert body["served_variant"] == "control"


def test_outcome_endpoint_appends_event(tmp_path: Path) -> None:
    service = make_service(tmp_path)
    with TestClient(create_app(service)) as client:
        rec = client.post("/v1/recommend", json={"user_id": 1, "k": 2}).json()
        response = client.post(
            "/v1/outcomes",
            json={"impression_id": rec["impression_id"], "user_id": 1, "clicked": True, "converted": False, "value": 0.0},
        )
        assert response.status_code == 202
        assert (tmp_path / "events" / "outcomes.jsonl").exists()


def test_conversion_requires_click(tmp_path: Path) -> None:
    service = make_service(tmp_path)
    with TestClient(create_app(service)) as client:
        response = client.post(
            "/v1/outcomes",
            json={"impression_id": "x", "user_id": 1, "clicked": False, "converted": True, "value": 1.0},
        )
        assert response.status_code == 422


def test_recommend_k_has_upper_bound(tmp_path: Path) -> None:
    service = make_service(tmp_path)
    with TestClient(create_app(service)) as client:
        response = client.post("/v1/recommend", json={"user_id": 1, "k": 101})
        assert response.status_code == 422


def test_unknown_impression_is_rejected(tmp_path: Path) -> None:
    service = make_service(tmp_path)
    with TestClient(create_app(service)) as client:
        response = client.post(
            "/v1/outcomes",
            json={"impression_id": "missing", "user_id": 1, "clicked": True, "converted": False, "value": 0.0},
        )
        assert response.status_code == 404


def test_outcome_user_must_match_exposure(tmp_path: Path) -> None:
    service = make_service(tmp_path)
    with TestClient(create_app(service)) as client:
        rec = client.post("/v1/recommend", json={"user_id": 1, "k": 2}).json()
        response = client.post(
            "/v1/outcomes",
            json={"impression_id": rec["impression_id"], "user_id": 2, "clicked": True, "converted": False, "value": 0.0},
        )
        assert response.status_code == 409


def test_metrics_and_ops_endpoints(tmp_path: Path) -> None:
    service = make_service(tmp_path)
    with TestClient(create_app(service)) as client:
        client.post("/v1/recommend", json={"user_id": 1, "k": 2})
        metrics = client.get("/metrics")
        assert metrics.status_code == 200
        assert "recsys_recommend_requests_total 1" in metrics.text
        ops = client.get("/v1/ops")
        assert ops.status_code == 200
        assert ops.json()["ready"] is True
        assert ops.json()["metrics"]["recommend_requests_total"] == 1


def test_readiness_fails_closed_on_integrity_failure(tmp_path: Path) -> None:
    settings = load_settings()
    engine = RecommendationEngine(make_bundle(tmp_path / "bundle.npz"))
    service = RecommendationService(
        settings,
        engine,
        JsonlEventStore(tmp_path / "events"),
        integrity_report={"passed": False, "checks": {"sha256_matches_manifest": False}},
    )
    with TestClient(create_app(service)) as client:
        assert client.get("/health").status_code == 200
        ready = client.get("/ready")
        assert ready.status_code == 503
        recommend = client.post("/v1/recommend", json={"user_id": 1, "k": 2})
        assert recommend.status_code == 503
        assert "artifact_integrity" in recommend.json()["detail"]


def test_model_info_exposes_integrity_report(tmp_path: Path) -> None:
    service = make_service(tmp_path)
    with TestClient(create_app(service)) as client:
        info = client.get("/v1/model-info").json()
        assert info["model_version"] == "recsys-m9"
        assert info["artifact_integrity"]["passed"] is True
