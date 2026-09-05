"""Point d'entree du service REST TimesFM."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.config import get_settings
from app.forecaster import TimesFMForecaster
from app.routers import forecast, health

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s | %(message)s",
)
logger = logging.getLogger("timesfm-api")

DESCRIPTION = """
Service REST autour de **TimesFM 3.0** (Google Research) pour la prevision de series
temporelles.

Toute la surface du modele est exposee : series univariees ou multivariees (avec attention
inter-variables), covariables passees et futures, z-normalisation, moyenne symetrique,
contrainte de positivite, tri des quantiles et mode univarie. Les series comportant plus de
32 variables au total sont automatiquement decoupees par le modele.

Les 9 quantiles renvoyes vont de 0.1 a 0.9 ; `point` correspond a la mediane.

> Les poids de TimesFM 3.0 sont distribues sous *TimesFM Non-Commercial License v1.0*
> (usage non commercial et hors production).
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.forecaster = TimesFMForecaster(settings)
    if settings.api_preload:
        logger.info(
            "Chargement de %s sur %s...",
            settings.timesfm_checkpoint,
            app.state.forecaster.device,
        )
        app.state.forecaster.load()
    else:
        logger.info("API_PRELOAD=false : le modele sera charge a la premiere requete.")
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="TimesFM API",
        version="0.1.0",
        description=DESCRIPTION,
        lifespan=lifespan,
        license_info={"name": "Voir README (poids sous licence non commerciale)"},
    )

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "validation_error",
                    "message": "Requete invalide.",
                    "details": [
                        {"loc": list(e.get("loc", [])), "msg": e.get("msg", "")}
                        for e in exc.errors()
                    ],
                }
            },
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": f"http_{exc.status_code}",
                    "message": str(exc.detail),
                    "details": None,
                }
            },
        )

    app.include_router(health.router)
    app.include_router(forecast.router)
    return app


app = create_app()
