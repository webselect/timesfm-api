"""Schemas Pydantic : entrees/sorties de l'API et validation.

La validation reproduit fidelement les contraintes de TimesFM 3.0 (longueur de contexte,
alignement des covariables, homogeneite du nombre de variables) afin de renvoyer un 422
explicite plutot que de laisser echouer la librairie au milieu de l'inference.
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
    """Une serie a prevoir, univariee (`values` 1-D) ou multivariee (`values` 2-D)."""

    model_config = ConfigDict(extra="forbid")

    id: str | None = Field(
        default=None,
        description="Identifiant libre, repris tel quel dans la reponse.",
    )
    values: Univariate | Multivariate = Field(
        description=(
            "Historique. Soit une liste de nombres (univarie), soit une liste de listes "
            "de meme longueur (multivarie, `(variables, longueur)`) : dans ce cas TimesFM "
            "applique son attention inter-variables. `null` marque une valeur manquante, "
            "interpolee lineairement par le modele."
        ),
    )
    past_only_covariates: Multivariate | None = Field(
        default=None,
        description=(
            "Covariables connues uniquement sur le passe, `(n, longueur_du_contexte)`."
        ),
    )
    past_future_covariates: Multivariate | None = Field(
        default=None,
        description=(
            "Covariables connues aussi sur le futur, `(n, longueur_du_contexte + horizon)`."
        ),
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
            raise ValueError(f"{label}: `values` ne peut pas etre vide.")

        rows = self.rows
        if any(len(row) == 0 for row in rows):
            raise ValueError(f"{label}: aucune variable ne peut etre vide.")

        length = len(rows[0])
        if any(len(row) != length for row in rows):
            raise ValueError(
                f"{label}: toutes les variables doivent avoir la meme longueur "
                f"(attendu {length})."
            )
        if length > MAX_CONTEXT_LENGTH:
            raise ValueError(
                f"{label}: contexte de {length} points, maximum {MAX_CONTEXT_LENGTH} "
                "pour TimesFM 3.0."
            )
        if len(rows) > settings.api_max_variates:
            raise ValueError(
                f"{label}: {len(rows)} variables cibles, maximum "
                f"{settings.api_max_variates} (API_MAX_VARIATES)."
            )
        if all(v is None for row in rows for v in row):
            raise ValueError(f"{label}: au moins une valeur doit etre renseignee.")

        for name, cov in (
            ("past_only_covariates", self.past_only_covariates),
            ("past_future_covariates", self.past_future_covariates),
        ):
            if cov is None:
                continue
            if not cov:
                raise ValueError(f"{label}: `{name}` fourni mais vide.")
            if any(len(row) == 0 for row in cov):
                raise ValueError(f"{label}: `{name}` contient une covariable vide.")
            if len({len(row) for row in cov}) != 1:
                raise ValueError(
                    f"{label}: toutes les covariables de `{name}` doivent avoir la meme longueur."
                )

        if self.past_only_covariates is not None:
            cov_len = len(self.past_only_covariates[0])
            if cov_len != length:
                raise ValueError(
                    f"{label}: `past_only_covariates` doit couvrir exactement le contexte "
                    f"({length} points), recu {cov_len}."
                )

        return self


class ForecastOptions(BaseModel):
    """Options d'inference, transmises telles quelles a `predict_batch`."""

    model_config = ConfigDict(extra="forbid")

    univariate: bool = Field(
        default=False,
        description=(
            "Traite chaque variable d'une serie multivariee independamment, sans attention "
            "inter-variables (mode d'evaluation de TimesFM3Evaluator)."
        ),
    )
    use_znorm: bool = Field(
        default=False,
        description="Z-normalise chaque variable avant l'inference, puis denormalise la sortie.",
    )
    use_symmetric_averaging: bool = Field(
        default=False,
        description="Moyenne la prevision avec celle de la serie inversee (reduit le biais).",
    )
    make_positive: bool = Field(
        default=False,
        description="Force la prevision a rester >= 0 si le contexte est lui-meme non negatif.",
    )
    sort_quantiles: bool = Field(
        default=True,
        description="Trie les quantiles pour eviter les croisements (q10 <= q50 <= q90).",
    )
    padding_mode: Literal["none", "edge"] | None = Field(
        default=None,
        description=(
            "Traitement des covariables futures quand l'horizon n'est pas un multiple de 64. "
            "`edge` prolonge la derniere valeur ; laisser vide pour un choix automatique."
        ),
    )


class ForecastRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    horizon: int = Field(ge=1, description="Nombre de pas a prevoir.")
    return_quantiles: bool = Field(
        default=True, description="Renvoyer les 9 quantiles en plus de la prevision mediane."
    )
    series: list[SeriesInput] = Field(min_length=1, description="Series a prevoir, en lot.")
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
                f"`horizon` de {self.horizon}, maximum {settings.api_max_horizon} "
                "(API_MAX_HORIZON)."
            )
        if len(self.series) > settings.api_max_series:
            raise ValueError(
                f"{len(self.series)} series, maximum {settings.api_max_series} (API_MAX_SERIES)."
            )

        ids = [s.id for s in self.series if s.id is not None]
        if len(ids) != len(set(ids)):
            raise ValueError("Les identifiants de series doivent etre uniques.")

        # TimesFM exige le meme nombre de variables cibles pour tout le lot.
        variates = {s.variates for s in self.series}
        if len(variates) > 1:
            raise ValueError(
                "Toutes les series d'un meme lot doivent avoir le meme nombre de variables "
                f"cibles ; recu {sorted(variates)}. Envoyez-les en requetes separees."
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
                    f"{label}: `past_future_covariates` doit couvrir le contexte "
                    f"({s.context_length}) plus {self.horizon} (horizon) ou "
                    f"{self.global_horizon} (horizon arrondi au patch de 64) pas, "
                    f"soit {s.context_length + self.horizon} ou "
                    f"{s.context_length + self.global_horizon} points ; recu {cov_len}."
                )

        return self


class SeriesForecast(BaseModel):
    """Prevision d'une serie. La forme reflete celle de l'entree."""

    id: str | None = None
    variates: int = Field(description="Nombre de variables cibles previstes.")
    point: Univariate | Multivariate = Field(
        description=(
            "Prevision mediane : `[horizon]` pour une entree univariee, "
            "`[variables][horizon]` pour une entree multivariee. Une valeur non finie "
            "renvoyee par le modele apparait en `null`."
        )
    )
    quantiles: dict[str, Univariate | Multivariate] | None = Field(
        default=None,
        description="Quantiles 0.1 a 0.9, meme forme que `point`. Absent si non demande.",
    )


class ForecastResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model: str
    device: str
    horizon: int
    elapsed_ms: float
    applied_options: dict = Field(
        description="Options reellement transmises au modele, apres resolution des defauts."
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
        description="Options de construction du modele, figees au chargement."
    )
    api_limits: dict


class ErrorBody(BaseModel):
    code: str
    message: str
    details: list | dict | None = None


class ErrorResponse(BaseModel):
    error: ErrorBody
