"""Chargement du modele TimesFM 3.0 et execution des previsions.

L'inference PyTorch est bloquante et monopolise le GPU : elle est deportee dans un thread
et serialisee par un semaphore, pour qu'une rafale de requetes HTTP ne sature pas le device.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Protocol

import anyio.to_thread
import numpy as np

from app.config import Settings

logger = logging.getLogger(__name__)


class ModelNotReady(RuntimeError):
    """The model is not loaded yet."""


@dataclass(frozen=True)
class Prediction:
    """Sortie pour une serie : mediane et, si demande, les 9 quantiles.

    Formes : `(H,)` / `(H, 9)` pour une entree univariee, `(V, H)` / `(V, H, 9)` pour une
    entree multivariee -- exactement ce que renvoie `TimesFM3Evaluator.predict_batch`.
    """

    point: np.ndarray
    quantiles: np.ndarray | None


def to_context_array(rows: list[list[float | None]], univariate: bool) -> np.ndarray:
    """Convertit des valeurs JSON en tableau float32, `None` devenant NaN.

    TimesFM distingue les entrees 1-D des entrees 2-D (attention inter-variables), donc la
    dimension d'origine est preservee.
    """
    array = np.array(
        [[np.nan if v is None else v for v in row] for row in rows],
        dtype=np.float32,
    )
    return array[0] if univariate else array


class Forecaster(Protocol):
    """Interface utilisee par les routes ; un double est injecte dans les tests."""

    @property
    def device(self) -> str: ...

    @property
    def loaded(self) -> bool: ...

    async def forecast(
        self,
        *,
        contexts: list[np.ndarray],
        horizon: int,
        past_only_covariates: list[np.ndarray | None] | None,
        past_future_covariates: list[np.ndarray | None] | None,
        ts_ids: list[str | None] | None,
        return_quantiles: bool,
        univariate: bool,
        use_znorm: bool,
        use_symmetric_averaging: bool,
        make_positive: bool,
        sort_quantiles: bool,
        padding_mode: str,
    ) -> list[Prediction]: ...


class TimesFMForecaster:
    """Enveloppe `TimesFM3Evaluator` : chargement, repli de device, appel en thread."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model = None
        self._device = settings.resolve_device()
        self._lock = asyncio.Semaphore(1)

    @property
    def device(self) -> str:
        return self._device

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        """Charge les poids (telechargement HuggingFace au premier appel).

        Toutes les operations PyTorch ne sont pas implementees en Metal : si l'initialisation
        sur `mps` echoue, on repli sur le CPU plutot que de laisser le service inutilisable.
        """
        if self._model is not None:
            return

        from timesfm3 import ModelConfig, TimesFM3Evaluator

        kwargs = self._settings.model_kwargs()
        started = time.perf_counter()
        try:
            self._model = TimesFM3Evaluator(ModelConfig(device=self._device, **kwargs))
        except Exception:
            if self._device == "cpu":
                raise
            logger.warning(
                "Could not load TimesFM on '%s', falling back to the CPU.",
                self._device,
                exc_info=True,
            )
            self._device = "cpu"
            self._model = TimesFM3Evaluator(ModelConfig(device="cpu", **kwargs))

        logger.info(
            "TimesFM loaded (%s) on %s in %.1f s",
            self._settings.timesfm_checkpoint,
            self._device,
            time.perf_counter() - started,
        )

    async def forecast(
        self,
        *,
        contexts: list[np.ndarray],
        horizon: int,
        past_only_covariates: list[np.ndarray | None] | None,
        past_future_covariates: list[np.ndarray | None] | None,
        ts_ids: list[str | None] | None,
        return_quantiles: bool,
        univariate: bool,
        use_znorm: bool,
        use_symmetric_averaging: bool,
        make_positive: bool,
        sort_quantiles: bool,
        padding_mode: str,
    ) -> list[Prediction]:
        if self._model is None:
            raise ModelNotReady("The TimesFM model is not loaded.")

        def run() -> list[Prediction]:
            outputs = self._model.predict_batch(
                contexts=contexts,
                horizon=horizon,
                past_only_covariates=past_only_covariates,
                past_future_covariates=past_future_covariates,
                # `ts_ids` n'est pas transmis : les identifiants sont reappliques par
                # position en sortie, ce qui evite toute ambiguite quand ils sont absents.
                ts_ids=None,
                return_quantiles=return_quantiles,
                use_symmetric_averaging=use_symmetric_averaging,
                make_positive=make_positive,
                sort_quantiles=sort_quantiles,
                use_znorm=use_znorm,
                padding_mode=padding_mode,
                univariate=univariate,
            )
            return [Prediction(point=out.forecast, quantiles=out.quantiles) for out in outputs]

        async with self._lock:
            return await anyio.to_thread.run_sync(run)
