from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from recsys_lab.config import load_settings
from recsys_lab.serving.engine import RecommendationEngine
from recsys_lab.serving.events import JsonlEventStore
from recsys_lab.serving.integrity import verify_serving_artifact
from recsys_lab.serving.service import RecommendationService


class RecommendRequest(BaseModel):
    user_id: int
    k: Annotated[int, Field(ge=1, le=100)] = 10


class OutcomeRequest(BaseModel):
    impression_id: str = Field(min_length=1, max_length=128)
    user_id: int
    clicked: bool = False
    converted: bool = False
    value: Annotated[float, Field(ge=0.0)] = 0.0


def create_app(service: RecommendationService | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if service is not None:
            app.state.recsys = service
        else:
            settings = load_settings()
            bundle = settings.paths.artifacts / "m8" / "serving_bundle.npz"
            manifest = settings.paths.artifacts / "m8" / "m8_manifest.json"
            engine = RecommendationEngine(bundle)
            store = JsonlEventStore(settings.paths.root / "data" / "online")
            integrity = verify_serving_artifact(bundle, manifest)
            app.state.recsys = RecommendationService(
                settings,
                engine,
                store,
                integrity_report=integrity,
            )
        yield

    app = FastAPI(
        title="Recsys Lab API",
        version="0.10.1",
        description="M9 production-hardened serving API with integrity checks and runtime metrics.",
        lifespan=lifespan,
    )

    def svc(request: Request) -> RecommendationService:
        return request.app.state.recsys

    def require_ready(request: Request) -> RecommendationService:
        service_ = svc(request)
        if not service_.ready:
            raise HTTPException(
                status_code=503,
                detail={
                    "reason": "service_not_ready",
                    "artifact_integrity": service_.integrity_report,
                },
            )
        return service_

    @app.get("/health")
    def health() -> dict[str, str]:
        # Liveness: process can answer HTTP. Readiness is intentionally stricter.
        return {"status": "ok"}

    @app.get("/ready")
    def ready(request: Request) -> dict[str, str]:
        service_ = require_ready(request)
        return {
            "status": "ready",
            "model_version": service_.model_version,
            "artifact_fingerprint": service_.engine.fingerprint,
        }

    @app.get("/metrics", response_class=PlainTextResponse)
    def metrics(request: Request) -> str:
        service_ = svc(request)
        return service_.metrics.prometheus_text(
            integrity_ok=bool(service_.integrity_report.get("passed", False)),
            challenger_eligible=service_.engine.challenger_eligible,
        )

    @app.get("/v1/ops")
    def ops(request: Request) -> dict:
        return svc(request).ops_snapshot()

    @app.get("/v1/model-info")
    def model_info(request: Request) -> dict:
        return svc(request).model_info()

    @app.post("/v1/recommend")
    def recommend(payload: RecommendRequest, request: Request) -> dict:
        return require_ready(request).recommend(user_id=payload.user_id, k=payload.k)

    @app.post("/v1/outcomes", status_code=202)
    def outcome(payload: OutcomeRequest, request: Request) -> dict:
        if payload.converted and not payload.clicked:
            raise HTTPException(status_code=422, detail="converted=true requires clicked=true")
        result = require_ready(request).outcome(
            impression_id=payload.impression_id,
            user_id=payload.user_id,
            clicked=payload.clicked,
            converted=payload.converted,
            value=payload.value,
        )
        if result.get("status") == "rejected":
            code = 404 if result.get("reason") == "unknown_impression" else 409
            raise HTTPException(status_code=code, detail=result.get("reason"))
        return result

    return app


app = create_app()
