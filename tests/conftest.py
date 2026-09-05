"""Fixtures communes : application de test et forecaster factice.

Aucun test du coeur ne charge les vrais poids : le modele est remplace par un double
deterministe qui enregistre les arguments recus, ce qui permet de verifier que chaque option
est bien transmise a `predict_batch`.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("API_PRELOAD", "false")

from app.config import QUANTILE_LEVELS, get_settings  # noqa: E402
from app.deps import get_forecaster  # noqa: E402
from app.forecaster import Prediction  # noqa: E402
from app.main import create_app  # noqa: E402


class FakeForecaster:
    """Prolonge la derniere valeur de chaque serie et fabrique des quantiles croissants."""

    def __init__(self, *, loaded: bool = True, device: str = "cpu") -> None:
        self._loaded = loaded
        self._device = device
        self.calls: list[dict] = []

    @property
    def device(self) -> str:
        return self._device

    @property
    def loaded(self) -> bool:
        return self._loaded

    async def forecast(self, **kwargs) -> list[Prediction]:
        self.calls.append(kwargs)
        horizon = kwargs["horizon"]
        predictions: list[Prediction] = []
        for context in kwargs["contexts"]:
            array = np.atleast_2d(np.asarray(context, dtype=np.float32))
            last = np.nan_to_num(array[:, -1], nan=0.0)
            point = np.repeat(last[:, None], horizon, axis=1)  # (V, H)
            quantiles = None
            if kwargs["return_quantiles"]:
                offsets = np.array(QUANTILE_LEVELS, dtype=np.float32) - 0.5
                quantiles = point[..., None] + offsets  # (V, H, 9)
            if np.ndim(context) == 1:
                point = point[0]
                quantiles = None if quantiles is None else quantiles[0]
            predictions.append(Prediction(point=point, quantiles=quantiles))
        return predictions


@pytest.fixture
def fake_forecaster() -> FakeForecaster:
    return FakeForecaster()


@pytest.fixture
def app(fake_forecaster: FakeForecaster):
    get_settings.cache_clear()
    application = create_app()
    application.dependency_overrides[get_forecaster] = lambda: fake_forecaster
    yield application
    application.dependency_overrides.clear()
    get_settings.cache_clear()


@pytest.fixture
async def client(app):
    from httpx import ASGITransport, AsyncClient

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as async_client:
        yield async_client
