"""Tests d'integration avec les vrais poids TimesFM 3.0.

Desactives par defaut (`addopts = -m 'not model'`) car ils telechargent ~1.2 Go au premier
lancement et prennent plusieurs secondes. A lancer via `make test-model`.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.config import Settings
from app.forecaster import TimesFMForecaster

pytestmark = pytest.mark.model


@pytest.fixture(scope="module")
def forecaster() -> TimesFMForecaster:
    instance = TimesFMForecaster(Settings())
    instance.load()
    return instance


def sine(length: int, period: float = 24.0, phase: float = 0.0) -> np.ndarray:
    return np.sin(2 * np.pi * (np.arange(length) + phase) / period).astype(np.float32)


async def test_univariate_forecast(forecaster):
    horizon = 64
    context = sine(512)
    [prediction] = await forecaster.forecast(
        contexts=[context],
        horizon=horizon,
        past_only_covariates=None,
        past_future_covariates=None,
        ts_ids=None,
        return_quantiles=True,
        univariate=False,
        use_znorm=False,
        use_symmetric_averaging=False,
        make_positive=False,
        sort_quantiles=True,
        padding_mode="none",
    )

    assert prediction.point.shape == (horizon,)
    assert prediction.quantiles.shape == (horizon, 9)
    assert np.isfinite(prediction.point).all()
    # Quantiles tries : q10 <= q50 <= q90 en tout point de l'horizon.
    assert (prediction.quantiles[:, 0] <= prediction.quantiles[:, 4] + 1e-6).all()
    assert (prediction.quantiles[:, 4] <= prediction.quantiles[:, 8] + 1e-6).all()
    # La sinusoide de periode 24 doit rester dans son amplitude d'origine.
    assert np.abs(prediction.point).max() < 2.0
    # Et la periodicite doit etre reconnue : correlation elevee avec la suite exacte.
    expected = sine(horizon, phase=512)
    correlation = np.corrcoef(prediction.point, expected)[0, 1]
    assert correlation > 0.8, f"correlation trop faible : {correlation:.2f}"


async def test_multivariate_forecast(forecaster):
    horizon = 32
    context = np.stack([sine(256), sine(256, phase=6.0), sine(256, period=48.0)])
    [prediction] = await forecaster.forecast(
        contexts=[context],
        horizon=horizon,
        past_only_covariates=None,
        past_future_covariates=None,
        ts_ids=None,
        return_quantiles=True,
        univariate=False,
        use_znorm=False,
        use_symmetric_averaging=False,
        make_positive=False,
        sort_quantiles=True,
        padding_mode="none",
    )

    assert prediction.point.shape == (3, horizon)
    assert prediction.quantiles.shape == (3, horizon, 9)
    assert np.isfinite(prediction.point).all()


async def test_future_covariates(forecaster):
    """Horizon non multiple de 64 : `edge` prolonge la covariable jusqu'au patch."""
    horizon = 30
    context_length = 256
    context = sine(context_length)
    future_covariate = np.tile([0.0, 1.0], (context_length + horizon) // 2 + 1)[
        : context_length + horizon
    ].astype(np.float32)

    [prediction] = await forecaster.forecast(
        contexts=[context],
        horizon=horizon,
        past_only_covariates=[np.atleast_2d(sine(context_length, period=12.0))],
        past_future_covariates=[np.atleast_2d(future_covariate)],
        ts_ids=None,
        return_quantiles=True,
        univariate=False,
        use_znorm=False,
        use_symmetric_averaging=False,
        make_positive=False,
        sort_quantiles=True,
        padding_mode="edge",
    )

    assert prediction.point.shape == (horizon,)
    assert np.isfinite(prediction.point).all()


async def test_variate_chunking_above_32(forecaster):
    """Au-dela de 32 variables par passe, TimesFM3Evaluator decoupe automatiquement."""
    variates = 40
    context = np.stack([sine(128, phase=float(i)) for i in range(variates)])
    [prediction] = await forecaster.forecast(
        contexts=[context],
        horizon=16,
        past_only_covariates=None,
        past_future_covariates=None,
        ts_ids=None,
        return_quantiles=False,
        univariate=False,
        use_znorm=False,
        use_symmetric_averaging=False,
        make_positive=False,
        sort_quantiles=True,
        padding_mode="none",
    )

    assert prediction.point.shape == (variates, 16)
    assert np.isfinite(prediction.point).all()


async def test_univariate_mode_unrolls_variates(forecaster):
    """`univariate=True` traite chaque variable independamment, sans attention croisee."""
    context = np.stack([sine(128), sine(128, phase=6.0)])
    [prediction] = await forecaster.forecast(
        contexts=[context],
        horizon=16,
        past_only_covariates=None,
        past_future_covariates=None,
        ts_ids=None,
        return_quantiles=True,
        univariate=True,
        use_znorm=True,
        use_symmetric_averaging=True,
        make_positive=False,
        sort_quantiles=True,
        padding_mode="none",
    )

    assert prediction.point.shape == (2, 16)
    assert prediction.quantiles.shape == (2, 16, 9)
