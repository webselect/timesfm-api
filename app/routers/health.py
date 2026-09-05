"""Sondes de disponibilite."""

from __future__ import annotations

from fastapi import APIRouter, Response, status

from app.deps import ForecasterDep

router = APIRouter(tags=["health"])


@router.get("/health", summary="Liveness")
def health() -> dict:
    """Repond toujours 200 tant que le processus est vivant."""
    return {"status": "ok"}


@router.get(
    "/ready",
    summary="Readiness",
    responses={503: {"description": "Modele pas encore charge"}},
)
def ready(response: Response, forecaster: ForecasterDep) -> dict:
    """200 une fois les poids charges, 503 sinon."""
    if not forecaster.loaded:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {"status": "loading", "device": forecaster.device}
    return {"status": "ready", "device": forecaster.device}
