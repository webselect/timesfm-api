"""Dependances FastAPI partagees."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from app.config import Settings, get_settings
from app.forecaster import Forecaster


def get_forecaster(request: Request) -> Forecaster:
    """Recupere le forecaster attache a l'application (surcharge dans les tests)."""
    return request.app.state.forecaster


ForecasterDep = Annotated[Forecaster, Depends(get_forecaster)]
SettingsDep = Annotated[Settings, Depends(get_settings)]
