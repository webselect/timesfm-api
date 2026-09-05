"""Sondes de disponibilite."""

from __future__ import annotations

from fastapi import APIRouter, Response, status

from app.deps import ForecasterDep

router = APIRouter(tags=["health"])


@router.get("/health", summary="Liveness")
def health() -> dict:
    """Always answers 200 while the process is alive."""
    return {"status": "ok"}


@router.get(
    "/ready",
    summary="Readiness",
    responses={503: {"description": "Model not loaded yet"}},
)
def ready(response: Response, forecaster: ForecasterDep) -> dict:
    """Answers 200 once the weights are loaded, 503 otherwise."""
    if not forecaster.loaded:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {"status": "loading", "device": forecaster.device}
    return {"status": "ready", "device": forecaster.device}
