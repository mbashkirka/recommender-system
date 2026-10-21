from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from time import perf_counter
from typing import Any
from uuid import uuid4

from recsys_lab.config import Settings
from recsys_lab.experiments.assignment import assign_two_arm
from recsys_lab.serving.engine import RecommendationEngine
from recsys_lab.serving.events import JsonlEventStore
from recsys_lab.serving.metrics import RuntimeMetrics


class RecommendationService:
    def __init__(
        self,
        settings: Settings,
        engine: RecommendationEngine,
        event_store: JsonlEventStore,
        *,
        metrics: RuntimeMetrics | None = None,
        integrity_report: dict[str, Any] | None = None,
    ) -> None:
        self.settings = settings
        self.engine = engine
        self.events = event_store
        window = settings.operations.metrics_window if settings.operations is not None else 5000
        self.metrics = metrics or RuntimeMetrics(latency_window=window)
        # Injected/unit-test services may not have an M8 manifest. Production create_app supplies the
        # strict integrity report computed from the real M8 files.
        self.integrity_report = integrity_report or {"passed": True, "mode": "injected_or_unmanaged"}
        self.model_version = "recsys-m9"
        self.experiment = settings.experiment.default_name

    @property
    def ready(self) -> bool:
        return bool(self.engine.bundle_path.exists() and self.integrity_report.get("passed", False))

    def recommend(self, user_id: int, k: int) -> dict[str, Any]:
        started = perf_counter()
        assigned = "unknown"
        effective_variant = "unknown"
        route = "unknown"
        try:
            assignment = assign_two_arm(
                experiment=self.experiment,
                unit_id=user_id,
                salt=self.settings.experiment.default_salt,
                control_weight=self.settings.experiment.control_weight,
            )
            assigned = assignment.variant
            # Offline-ineligible challengers are never served. The deterministic assignment is
            # retained for audit, but the effective variant fails closed to control.
            effective_variant = assigned if (assigned == "control" or self.engine.challenger_eligible) else "control"
            recs, routing = self.engine.recommend(user_id=user_id, k=k, policy=effective_variant)
            route = str(routing["route"])
            latency_ms = (perf_counter() - started) * 1000.0
            impression_id = str(uuid4())
            now = datetime.now(UTC).isoformat()
            exposure = {
                "event_id": str(uuid4()),
                "impression_id": impression_id,
                "experiment": self.experiment,
                "unit_id": str(user_id),
                "assigned_variant": assigned,
                "variant": effective_variant,
                "bucket": assignment.bucket,
                "timestamp": now,
                "model_version": self.model_version,
                "artifact_fingerprint": self.engine.fingerprint,
                "latency_ms": latency_ms,
                "is_error": False,
                "item_ids": [r.item_id for r in recs],
                "routing": routing,
            }
            self.events.log_exposure(exposure)
            self.metrics.observe_recommend(latency_ms, route, assigned, effective_variant, error=False)
            return {
                "impression_id": impression_id,
                "experiment": self.experiment,
                "assigned_variant": assigned,
                "served_variant": effective_variant,
                "challenger_offline_eligible": self.engine.challenger_eligible,
                "model_version": self.model_version,
                "artifact_fingerprint": self.engine.fingerprint,
                "route": route,
                "policy": routing["effective_policy"],
                "latency_ms": latency_ms,
                "recommendations": [asdict(r) for r in recs],
            }
        except Exception:
            latency_ms = (perf_counter() - started) * 1000.0
            self.metrics.observe_recommend(latency_ms, route, assigned, effective_variant, error=True)
            raise

    def outcome(self, impression_id: str, user_id: int, clicked: bool, converted: bool, value: float) -> dict[str, Any]:
        reference_error = self.events.validate_outcome_reference(impression_id, user_id)
        if reference_error is not None:
            self.metrics.observe_outcome("rejected", reference_error)
            return {"status": "rejected", "reason": reference_error}
        payload = {
            "event_id": str(uuid4()),
            "impression_id": impression_id,
            "unit_id": str(user_id),
            "timestamp": datetime.now(UTC).isoformat(),
            "clicked": bool(clicked),
            "converted": bool(converted),
            "value": float(value),
        }
        self.events.log_outcome(payload)
        self.metrics.observe_outcome("accepted")
        return {"status": "accepted", "event_id": payload["event_id"]}

    def model_info(self) -> dict[str, Any]:
        manifest_path = self.settings.paths.artifacts / "m8" / "m8_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
        return {
            "model_version": self.model_version,
            "artifact_fingerprint": self.engine.fingerprint,
            "artifact_integrity": self.integrity_report,
            "catalog_items": len(self.engine.all_item_ids),
            "candidate_k": int(self.settings.ranking.candidate_k if self.settings.ranking else 100),
            "default_top_k": int(self.engine.primary_k),
            "ltr_blend": self.engine.ltr_blend,
            "deployed_reranker": {
                "diversity_weight": self.engine.deployed_dw,
                "novelty_weight": self.engine.deployed_nw,
            },
            "challenger_offline_eligible": self.engine.challenger_eligible,
            "experiment": self.experiment,
            "manifest": manifest,
        }

    def ops_snapshot(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "model_version": self.model_version,
            "artifact_fingerprint": self.engine.fingerprint,
            "artifact_integrity": self.integrity_report,
            "challenger_offline_eligible": self.engine.challenger_eligible,
            "metrics": self.metrics.snapshot(),
        }
