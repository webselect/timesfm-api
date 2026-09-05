"""Prevision : formes de sortie, options et covariables."""

from __future__ import annotations

import numpy as np

CONTEXT = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]


async def test_univariate_shape(client, fake_forecaster):
    response = await client.post(
        "/v1/forecast",
        json={"horizon": 4, "series": [{"id": "ventes", "values": CONTEXT}]},
    )
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["model"] == "google/timesfm-3.0-pytorch"
    assert body["horizon"] == 4
    assert body["elapsed_ms"] >= 0

    [forecast] = body["forecasts"]
    assert forecast["id"] == "ventes"
    assert forecast["variates"] == 1
    assert forecast["point"] == [15.0, 15.0, 15.0, 15.0]
    assert sorted(forecast["quantiles"]) == [f"0.{i}" for i in range(1, 10)]
    assert forecast["quantiles"]["0.5"] == [15.0, 15.0, 15.0, 15.0]
    assert len(forecast["quantiles"]["0.1"]) == 4
    # Le contexte reste 1-D pour que le modele traite l'entree comme univariee.
    assert fake_forecaster.calls[0]["contexts"][0].ndim == 1


async def test_multivariate_shape(client, fake_forecaster):
    values = [[1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]]
    response = await client.post(
        "/v1/forecast",
        json={"horizon": 3, "series": [{"id": "capteurs", "values": values}]},
    )
    assert response.status_code == 200, response.text
    [forecast] = response.json()["forecasts"]

    assert forecast["variates"] == 2
    assert forecast["point"] == [[4.0, 4.0, 4.0], [40.0, 40.0, 40.0]]
    assert np.allclose(
        forecast["quantiles"]["0.9"], [[4.4, 4.4, 4.4], [40.4, 40.4, 40.4]]
    )
    # Une entree 2-D est transmise en 2-D : c'est ce qui declenche l'attention inter-variables.
    assert fake_forecaster.calls[0]["contexts"][0].shape == (2, 4)


async def test_batch_preserves_order_and_ids(client):
    response = await client.post(
        "/v1/forecast",
        json={
            "horizon": 2,
            "series": [
                {"id": "a", "values": [1.0, 2.0]},
                {"id": "b", "values": [5.0, 6.0]},
                {"values": [9.0, 9.0]},
            ],
        },
    )
    assert response.status_code == 200, response.text
    forecasts = response.json()["forecasts"]
    assert [f["id"] for f in forecasts] == ["a", "b", None]
    assert forecasts[1]["point"] == [6.0, 6.0]


async def test_null_values_are_accepted(client, fake_forecaster):
    response = await client.post(
        "/v1/forecast",
        json={"horizon": 2, "series": [{"values": [1.0, None, 3.0]}]},
    )
    assert response.status_code == 200, response.text
    context = fake_forecaster.calls[0]["contexts"][0]
    # `null` devient NaN : TimesFM interpole lineairement les trous internes.
    assert np.isnan(context[1])


async def test_quantiles_can_be_disabled(client, fake_forecaster):
    response = await client.post(
        "/v1/forecast",
        json={"horizon": 2, "return_quantiles": False, "series": [{"values": CONTEXT}]},
    )
    assert response.status_code == 200, response.text
    [forecast] = response.json()["forecasts"]
    assert forecast["quantiles"] is None
    assert fake_forecaster.calls[0]["return_quantiles"] is False


async def test_options_are_forwarded(client, fake_forecaster):
    options = {
        "univariate": True,
        "use_znorm": True,
        "use_symmetric_averaging": True,
        "make_positive": True,
        "sort_quantiles": False,
        "padding_mode": "edge",
    }
    response = await client.post(
        "/v1/forecast",
        json={"horizon": 2, "series": [{"values": CONTEXT}], "options": options},
    )
    assert response.status_code == 200, response.text

    call = fake_forecaster.calls[0]
    for key, value in options.items():
        assert call[key] == value, key
    assert response.json()["applied_options"]["padding_mode"] == "edge"


async def test_default_options(client, fake_forecaster):
    await client.post("/v1/forecast", json={"horizon": 2, "series": [{"values": CONTEXT}]})
    call = fake_forecaster.calls[0]
    assert call["univariate"] is False
    assert call["use_znorm"] is False
    assert call["use_symmetric_averaging"] is False
    assert call["make_positive"] is False
    assert call["sort_quantiles"] is True
    assert call["padding_mode"] == "none"


async def test_past_only_covariates_are_forwarded(client, fake_forecaster):
    response = await client.post(
        "/v1/forecast",
        json={
            "horizon": 2,
            "series": [
                {
                    "values": [1.0, 2.0, 3.0],
                    "past_only_covariates": [[0.5, 0.6, 0.7], [1.5, 1.6, 1.7]],
                }
            ],
        },
    )
    assert response.status_code == 200, response.text
    covariates = fake_forecaster.calls[0]["past_only_covariates"][0]
    assert covariates.shape == (2, 3)


async def test_past_future_covariates_are_forwarded(client, fake_forecaster):
    response = await client.post(
        "/v1/forecast",
        json={
            "horizon": 2,
            "series": [
                {"values": [1.0, 2.0, 3.0], "past_future_covariates": [[0, 0, 1, 1, 0]]}
            ],
        },
    )
    assert response.status_code == 200, response.text
    covariates = fake_forecaster.calls[0]["past_future_covariates"][0]
    assert covariates.shape == (1, 5)  # contexte (3) + horizon (2)


async def test_padding_mode_switches_to_edge_with_future_covariates(client, fake_forecaster):
    """Horizon non multiple de 64 : sans `edge`, la partie future serait rognee."""
    response = await client.post(
        "/v1/forecast",
        json={
            "horizon": 3,
            "series": [{"values": [1.0, 2.0], "past_future_covariates": [[0, 1, 0, 1, 1]]}],
        },
    )
    assert response.status_code == 200, response.text
    assert fake_forecaster.calls[0]["padding_mode"] == "edge"
    assert response.json()["applied_options"]["padding_mode"] == "edge"


async def test_padding_mode_stays_none_without_future_covariates(client, fake_forecaster):
    await client.post("/v1/forecast", json={"horizon": 3, "series": [{"values": CONTEXT}]})
    assert fake_forecaster.calls[0]["padding_mode"] == "none"


async def test_covariates_absent_are_not_sent(client, fake_forecaster):
    await client.post("/v1/forecast", json={"horizon": 2, "series": [{"values": CONTEXT}]})
    call = fake_forecaster.calls[0]
    assert call["past_only_covariates"] is None
    assert call["past_future_covariates"] is None


async def test_returns_503_when_model_not_loaded(app):
    from httpx import ASGITransport, AsyncClient

    from app.deps import get_forecaster
    from app.forecaster import ModelNotReady

    class NotReady:
        device = "mps"
        loaded = False

        async def forecast(self, **_):
            raise ModelNotReady("The TimesFM model is not loaded.")

    app.dependency_overrides[get_forecaster] = NotReady
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/v1/forecast", json={"horizon": 2, "series": [{"values": CONTEXT}]}
        )
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "http_503"
