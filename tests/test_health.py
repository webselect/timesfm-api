"""Sondes de disponibilite et description du modele."""

from __future__ import annotations

from tests.conftest import FakeForecaster


async def test_health(client):
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_ready_when_loaded(client):
    response = await client.get("/ready")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"


async def test_ready_returns_503_while_loading(app, fake_forecaster: FakeForecaster):
    from httpx import ASGITransport, AsyncClient

    from app.deps import get_forecaster

    loading = FakeForecaster(loaded=False, device="mps")
    app.dependency_overrides[get_forecaster] = lambda: loading
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/ready")
    assert response.status_code == 503
    assert response.json() == {"status": "loading", "device": "mps"}


async def test_model_info(client):
    response = await client.get("/v1/model")
    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "google/timesfm-3.0-pytorch"
    assert body["max_context_length"] == 15360
    assert body["input_patch_length"] == 32
    assert body["output_patch_length"] == 64
    assert body["max_variates_per_forward"] == 32
    assert body["quantile_levels"] == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # Les options de construction du modele sont exposees, sans le jeton HuggingFace.
    assert body["model_options"]["use_variate_attention"] is True
    assert "token" not in body["model_options"]
    assert body["api_limits"]["max_horizon"] == 1024


async def test_openapi_is_served(client):
    response = await client.get("/openapi.json")
    assert response.status_code == 200
    paths = response.json()["paths"]
    assert "/v1/forecast" in paths
    assert "/v1/model" in paths
