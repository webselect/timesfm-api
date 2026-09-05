"""Schemas Pydantic : entrees/sorties de l'API et validation.

La validation reproduit fidelement les contraintes de TimesFM 3.0 (longueur de contexte,
alignement des covariables, homogeneite du nombre de variables) afin de renvoyer un 422
explicite plutot que de laisser echouer la librairie au milieu de l'inference.

Les textes exposes (docstrings de modeles, `description`, messages d'erreur) sont en anglais :
ils constituent la documentation publique de l'API.
"""

from __future__ import annotations

import math
from typing import Annotated, Literal

from pydantic import AllowInfNan, BaseModel, ConfigDict, Field, model_validator

from app.config import (
    MAX_CONTEXT_LENGTH,
    OUTPUT_PATCH_LENGTH,
    QUANTILE_LEVELS,
    get_settings,
)

# `null` est autorise (valeur manquante -> NaN, interpole par TimesFM), mais pas inf/NaN
# litteraux, que le modele ne sait pas traiter.
Value = Annotated[float, AllowInfNan(False)] | None

Univariate = list[Value]
Multivariate = list[list[Value]]


class SeriesInput(BaseModel):
    """A series to forecast, univariate (1-D `values`) or multivariate (2-D `values`)."""

    model_config = ConfigDict(extra="forbid")

    id: str | None = Field(
        default=None,
        description="Free-form identifier, echoed back in the response.",
    )
    values: Univariate | Multivariate = Field(
        description=(
            "History. Either a list of numbers (univariate) or a list of equal-length lists "
            "(multivariate, `(variates, length)`), in which case TimesFM applies its "
            "cross-variate attention. `null` marks a missing value, linearly interpolated "
            "by the model."
        ),
    )
    past_only_covariates: Multivariate | None = Field(
        default=None,
        description="Covariates known over the past only, `(n, context_length)`.",
    )
    past_future_covariates: Multivariate | None = Field(
        default=None,
        description="Covariates known over the future too, `(n, context_length + horizon)`.",
    )

    @property
    def is_univariate(self) -> bool:
        return bool(self.values) and not isinstance(self.values[0], list)

    @property
    def rows(self) -> list[list[Value]]:
        """Vue 2-D de `values`, quelle que soit la forme d'entree."""
        if self.is_univariate:
            return [self.values]  # type: ignore[list-item]
        return self.values  # type: ignore[return-value]

    @property
    def context_length(self) -> int:
        return len(self.rows[0])

    @property
    def variates(self) -> int:
        return len(self.rows)

    @model_validator(mode="after")
    def _check_shapes(self) -> SeriesInput:
        settings = get_settings()
        label = f"series '{self.id}'" if self.id else "series"

        if not self.values:
            raise ValueError(f"{label}: `values` cannot be empty.")

        rows = self.rows
        if any(len(row) == 0 for row in rows):
            raise ValueError(f"{label}: no variate can be empty.")

        length = len(rows[0])
        if any(len(row) != length for row in rows):
            raise ValueError(
                f"{label}: every variate must have the same length (expected {length})."
            )
        if length > MAX_CONTEXT_LENGTH:
            raise ValueError(
                f"{label}: context of {length} points, maximum {MAX_CONTEXT_LENGTH} "
                "for TimesFM 3.0."
            )
        if len(rows) > settings.api_max_variates:
            raise ValueError(
                f"{label}: {len(rows)} target variates, maximum "
                f"{settings.api_max_variates} (API_MAX_VARIATES)."
            )
        if all(v is None for row in rows for v in row):
            raise ValueError(f"{label}: at least one value must be provided.")

        for name, cov in (
            ("past_only_covariates", self.past_only_covariates),
            ("past_future_covariates", self.past_future_covariates),
        ):
            if cov is None:
                continue
            if not cov:
                raise ValueError(f"{label}: `{name}` was provided but is empty.")
            if any(len(row) == 0 for row in cov):
                raise ValueError(f"{label}: `{name}` contains an empty covariate.")
            if len({len(row) for row in cov}) != 1:
                raise ValueError(
                    f"{label}: every covariate in `{name}` must have the same length."
                )

        if self.past_only_covariates is not None:
            cov_len = len(self.past_only_covariates[0])
            if cov_len != length:
                raise ValueError(
                    f"{label}: `past_only_covariates` must cover exactly the context "
                    f"({length} points), got {cov_len}."
                )

        return self


class ForecastOptions(BaseModel):
    """Inference options, passed through to `predict_batch` as-is."""

    model_config = ConfigDict(extra="forbid")

    univariate: bool = Field(
        default=False,
        description=(
            "Treat each variate of a multivariate series independently, without "
            "cross-variate attention (TimesFM3Evaluator's evaluation mode)."
        ),
    )
    use_znorm: bool = Field(
        default=False,
        description="Z-normalize each variate before inference, then denormalize the output.",
    )
    use_symmetric_averaging: bool = Field(
        default=False,
        description="Average the forecast with the one of the flipped series (reduces bias).",
    )
    make_positive: bool = Field(
        default=False,
        description="Clamp the forecast to >= 0 when the context itself is non-negative.",
    )
    sort_quantiles: bool = Field(
        default=True,
        description="Sort quantiles to avoid crossings (q10 <= q50 <= q90).",
    )
    padding_mode: Literal["none", "edge"] | None = Field(
        default=None,
        description=(
            "How future covariates are handled when the horizon is not a multiple of 64. "
            "`edge` extends the last value; leave unset for automatic selection."
        ),
    )


class ForecastRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    horizon: int = Field(ge=1, description="Number of steps to forecast.")
    return_quantiles: bool = Field(
        default=True, description="Return the 9 quantiles alongside the median forecast."
    )
    series: list[SeriesInput] = Field(min_length=1, description="Series to forecast, as a batch.")
    options: ForecastOptions = Field(default_factory=ForecastOptions)

    @property
    def global_horizon(self) -> int:
        """Horizon effectivement calcule par le modele (arrondi au patch de sortie)."""
        return math.ceil(self.horizon / OUTPUT_PATCH_LENGTH) * OUTPUT_PATCH_LENGTH

    @property
    def resolved_padding_mode(self) -> str:
        """`padding_mode` explicite, ou deduit de la presence de covariables futures.

        Sans `edge`, une covariable future fournie sur exactement `horizon` pas est rognee
        silencieusement quand l'horizon n'est pas un multiple de 64 : on bascule donc
        automatiquement dans ce cas.
        """
        if self.options.padding_mode is not None:
            return self.options.padding_mode
        has_future_cov = any(s.past_future_covariates is not None for s in self.series)
        if has_future_cov and self.horizon != self.global_horizon:
            return "edge"
        return "none"

    @model_validator(mode="after")
    def _check_batch(self) -> ForecastRequest:
        settings = get_settings()

        if self.horizon > settings.api_max_horizon:
            raise ValueError(
                f"`horizon` of {self.horizon}, maximum {settings.api_max_horizon} "
                "(API_MAX_HORIZON)."
            )
        if len(self.series) > settings.api_max_series:
            raise ValueError(
                f"{len(self.series)} series, maximum {settings.api_max_series} (API_MAX_SERIES)."
            )

        ids = [s.id for s in self.series if s.id is not None]
        if len(ids) != len(set(ids)):
            raise ValueError("Series identifiers must be unique.")

        # TimesFM exige le meme nombre de variables cibles pour tout le lot.
        variates = {s.variates for s in self.series}
        if len(variates) > 1:
            raise ValueError(
                "Every series in a batch must have the same number of target variates; "
                f"got {sorted(variates)}. Send them as separate requests."
            )

        # Les covariables futures doivent couvrir le contexte plus la partie future.
        for s in self.series:
            if s.past_future_covariates is None:
                continue
            cov_len = len(s.past_future_covariates[0])
            future_len = cov_len - s.context_length
            allowed = {self.horizon, self.global_horizon}
            if future_len not in allowed:
                label = f"series '{s.id}'" if s.id else "series"
                raise ValueError(
                    f"{label}: `past_future_covariates` must cover the context "
                    f"({s.context_length}) plus either {self.horizon} (horizon) or "
                    f"{self.global_horizon} (horizon rounded up to the 64-step patch), "
                    f"that is {s.context_length + self.horizon} or "
                    f"{s.context_length + self.global_horizon} points; got {cov_len}."
                )

        return self


class SeriesForecast(BaseModel):
    """Forecast for one series. Its shape mirrors the input."""

    id: str | None = None
    variates: int = Field(description="Number of forecast target variates.")
    point: Univariate | Multivariate = Field(
        description=(
            "Median forecast: `[horizon]` for a univariate input, `[variates][horizon]` for "
            "a multivariate one. A non-finite value returned by the model appears as `null`."
        )
    )
    quantiles: dict[str, Univariate | Multivariate] | None = Field(
        default=None,
        description="Quantiles 0.1 to 0.9, same shape as `point`. Absent when not requested.",
    )


class ForecastResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model: str
    device: str
    horizon: int
    elapsed_ms: float
    applied_options: dict = Field(
        description="Options actually passed to the model, once defaults are resolved."
    )
    forecasts: list[SeriesForecast]


class ModelInfo(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model: str
    device: str
    loaded: bool
    quantile_levels: list[float] = QUANTILE_LEVELS
    max_context_length: int = MAX_CONTEXT_LENGTH
    input_patch_length: int
    output_patch_length: int
    max_variates_per_forward: int
    model_options: dict = Field(
        description="Model construction options, frozen at load time."
    )
    api_limits: dict


class ErrorBody(BaseModel):
    code: str
    message: str
    details: list | dict | None = None


class ErrorResponse(BaseModel):
    error: ErrorBody
