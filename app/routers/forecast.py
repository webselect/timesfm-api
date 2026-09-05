"""Endpoints de prevision."""

from __future__ import annotations

import logging
import math
import time
from typing import Annotated

import numpy as np
from fastapi import APIRouter, Body, HTTPException, status

from app.config import (
    INPUT_PATCH_LENGTH,
    MAX_CONTEXT_LENGTH,
    MAX_VARIATES_PER_FORWARD,
    OUTPUT_PATCH_LENGTH,
    QUANTILE_LEVELS,
)
from app.deps import ForecasterDep, SettingsDep
from app.forecaster import ModelNotReady, to_context_array
from app.schemas import (
    ForecastRequest,
    ForecastResponse,
    ModelInfo,
    SeriesForecast,
    SeriesInput,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["forecast"])


REQUEST_EXAMPLES = {
    "univarie": {
        "summary": "Univarie",
        "description": "Une serie simple ; `null` marque une valeur manquante.",
        "value": {
            "horizon": 12,
            "return_quantiles": True,
            "series": [
                {
                    "id": "ventes",
                    "values": [12.0, 13.5, 11.2, None, 14.8, 15.1, 16.0, 14.2],
                }
            ],
        },
    },
    "multivarie": {
        "summary": "Multivarie",
        "description": (
            "Une entree 2-D `(variables, longueur)` active l'attention inter-variables."
        ),
        "value": {
            "horizon": 12,
            "series": [
                {
                    "id": "capteurs",
                    "values": [
                        [1.0, 1.2, 1.4, 1.3, 1.5, 1.7, 1.6, 1.8],
                        [8.0, 7.6, 7.9, 8.2, 8.1, 8.4, 8.3, 8.6],
                    ],
                }
            ],
        },
    },
    "covariables": {
        "summary": "Avec covariables",
        "description": (
            "La covariable future couvre le contexte plus l'horizon ; `edge` la prolonge "
            "jusqu'au patch de 64."
        ),
        "value": {
            "horizon": 3,
            "series": [
                {
                    "id": "trafic",
                    "values": [100.0, 120.0, 90.0, 130.0],
                    "past_only_covariates": [[5.0, 6.0, 4.0, 7.0]],
                    "past_future_covariates": [[0, 0, 1, 1, 0, 0, 1]],
                }
            ],
            "options": {"padding_mode": "edge"},
        },
    },
    "options": {
        "summary": "Toutes les options",
        "description": "Surface complete des options d'inference de TimesFM 3.0.",
        "value": {
            "horizon": 8,
            "return_quantiles": True,
            "series": [{"id": "demande", "values": [3.0, 5.0, 4.0, 6.0, 5.5, 7.0]}],
            "options": {
                "univariate": False,
                "use_znorm": True,
                "use_symmetric_averaging": True,
                "make_positive": True,
                "sort_quantiles": True,
                "padding_mode": "none",
            },
        },
    },
}



def _covariate_array(rows: list[list[float | None]] | None) -> np.ndarray | None:
    if rows is None:
        return None
    return np.array(
        [[np.nan if v is None else v for v in row] for row in rows],
        dtype=np.float32,
    )


def _clean(values: np.ndarray) -> list:
    """Convertit un tableau numpy en listes JSON, les valeurs non finies devenant `null`."""
    if values.ndim == 1:
        return [float(v) if math.isfinite(v) else None for v in values]
    return [_clean(row) for row in values]


def _to_series_forecast(
    series: SeriesInput, point: np.ndarray, quantiles: np.ndarray | None
) -> SeriesForecast:
    """Met en forme la sortie du modele en conservant la dimension de l'entree."""
    payload: dict = {
        "id": series.id,
        "variates": series.variates,
        "point": _clean(point),
    }
    if quantiles is not None:
        # (H, 9) en univarie, (V, H, 9) en multivarie : le dernier axe porte les quantiles.
        payload["quantiles"] = {
            str(level): _clean(quantiles[..., index])
            for index, level in enumerate(QUANTILE_LEVELS)
        }
    return SeriesForecast(**payload)


@router.post(
    "/forecast",
    response_model=ForecastResponse,
    summary="Prevoir un lot de series",
    responses={503: {"description": "Modele pas encore charge"}},
)
async def forecast(
    payload: Annotated[ForecastRequest, Body(openapi_examples=REQUEST_EXAMPLES)],
    forecaster: ForecasterDep,
    settings: SettingsDep,
) -> ForecastResponse:
    """Prevoit `horizon` pas pour chaque serie du lot.

    Couvre toute la surface de TimesFM 3.0 : series univariees ou multivariees (avec
    attention inter-variables), covariables passees et futures, z-normalisation, moyenne
    symetrique, contrainte de positivite et tri des quantiles.
    """
    univariate_inputs = [s.is_univariate for s in payload.series]
    contexts = [
        to_context_array(s.rows, univariate=s.is_univariate) for s in payload.series
    ]
    past_only = [_covariate_array(s.past_only_covariates) for s in payload.series]
    past_future = [_covariate_array(s.past_future_covariates) for s in payload.series]
    padding_mode = payload.resolved_padding_mode

    started = time.perf_counter()
    try:
        predictions = await forecaster.forecast(
            contexts=contexts,
            horizon=payload.horizon,
            past_only_covariates=past_only if any(c is not None for c in past_only) else None,
            past_future_covariates=(
                past_future if any(c is not None for c in past_future) else None
            ),
            ts_ids=[s.id for s in payload.series],
            return_quantiles=payload.return_quantiles,
            univariate=payload.options.univariate,
            use_znorm=payload.options.use_znorm,
            use_symmetric_averaging=payload.options.use_symmetric_averaging,
            make_positive=payload.options.make_positive,
            sort_quantiles=payload.options.sort_quantiles,
            padding_mode=padding_mode,
        )
    except ModelNotReady as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc
    elapsed_ms = (time.perf_counter() - started) * 1000

    if len(predictions) != len(payload.series):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                f"Le modele a renvoye {len(predictions)} previsions pour "
                f"{len(payload.series)} series."
            ),
        )

    logger.info(
        "forecast series=%d horizon=%d univariate_inputs=%s duree=%.0f ms",
        len(payload.series),
        payload.horizon,
        all(univariate_inputs),
        elapsed_ms,
    )

    return ForecastResponse(
        model=settings.timesfm_checkpoint,
        device=forecaster.device,
        horizon=payload.horizon,
        elapsed_ms=round(elapsed_ms, 2),
        applied_options={
            **payload.options.model_dump(),
            "padding_mode": padding_mode,
            "return_quantiles": payload.return_quantiles,
        },
        forecasts=[
            _to_series_forecast(series, prediction.point, prediction.quantiles)
            for series, prediction in zip(payload.series, predictions, strict=True)
        ],
    )


@router.get("/model", response_model=ModelInfo, summary="Configuration effective du modele")
def model_info(forecaster: ForecasterDep, settings: SettingsDep) -> ModelInfo:
    """Expose le checkpoint, le device et toutes les options figees au chargement."""
    options = settings.model_kwargs()
    options.pop("token", None)  # ne jamais renvoyer le jeton HuggingFace
    return ModelInfo(
        model=settings.timesfm_checkpoint,
        device=forecaster.device,
        loaded=forecaster.loaded,
        input_patch_length=INPUT_PATCH_LENGTH,
        output_patch_length=OUTPUT_PATCH_LENGTH,
        max_context_length=MAX_CONTEXT_LENGTH,
        max_variates_per_forward=MAX_VARIATES_PER_FORWARD,
        model_options=options,
        api_limits={
            "max_series": settings.api_max_series,
            "max_horizon": settings.api_max_horizon,
            "max_variates": settings.api_max_variates,
        },
    )
