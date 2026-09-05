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
REST service around **TimesFM 3.0** (Google Research) for time series forecasting.

The full surface of the model is exposed: univariate or multivariate series (with
cross-variate attention), past and future covariates, z-normalization, symmetric averaging,
positivity constraint, quantile sorting and univariate mode. Series carrying more than 32
variates in total are chunked automatically by the model.

The 9 returned quantiles run from 0.1 to 0.9; `point` is the median.

> The TimesFM 3.0 weights are distributed under the *TimesFM Non-Commercial License v1.0*
> (non-commercial, non-production use).
"""

@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.forecaster = TimesFMForecaster(settings)
    if settings.api_preload:
        logger.info(
            "Loading %s on %s...",
            settings.timesfm_checkpoint,
            app.state.forecaster.device,
        )
        app.state.forecaster.load()
    else:
        logger.info("API_PRELOAD=false: the model will load on the first request.")
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="TimesFM API",
        version="0.1.0",
        description=DESCRIPTION,
        lifespan=lifespan,
        license_info={"name": "See README (weights under a non-commercial license)"},
    )

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "validation_error",
                    "message": "Invalid request.",
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
