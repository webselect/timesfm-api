"""Cas d'erreur : chaque contrainte de TimesFM est refusee en 422, avec un message explicite."""

from __future__ import annotations

import pytest

CONTEXT = [1.0, 2.0, 3.0]


def _messages(response) -> str:
    return " ".join(d["msg"] for d in response.json()["error"]["details"])


async def test_error_envelope(client):
    response = await client.post("/v1/forecast", json={"horizon": 2, "series": []})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


@pytest.mark.parametrize("horizon", [0, -5])
async def test_horizon_must_be_positive(client, horizon):
    response = await client.post(
        "/v1/forecast", json={"horizon": horizon, "series": [{"values": CONTEXT}]}
    )
    assert response.status_code == 422


async def test_horizon_above_limit(client):
    response = await client.post(
        "/v1/forecast", json={"horizon": 5000, "series": [{"values": CONTEXT}]}
    )
    assert response.status_code == 422
    assert "API_MAX_HORIZON" in _messages(response)


async def test_empty_series_list(client):
    response = await client.post("/v1/forecast", json={"horizon": 2, "series": []})
    assert response.status_code == 422


async def test_too_many_series(client):
    series = [{"id": str(i), "values": CONTEXT} for i in range(33)]
    response = await client.post("/v1/forecast", json={"horizon": 2, "series": series})
    assert response.status_code == 422
    assert "API_MAX_SERIES" in _messages(response)


async def test_context_too_long(client):
    response = await client.post(
        "/v1/forecast", json={"horizon": 2, "series": [{"values": [1.0] * 15361}]}
    )
    assert response.status_code == 422
    assert "15360" in _messages(response)


async def test_empty_values(client):
    response = await client.post("/v1/forecast", json={"horizon": 2, "series": [{"values": []}]})
    assert response.status_code == 422


async def test_all_null_values(client):
    response = await client.post(
        "/v1/forecast", json={"horizon": 2, "series": [{"values": [None, None]}]}
    )
    assert response.status_code == 422
    assert "at least one value" in _messages(response)


async def test_infinite_value_is_rejected(client):
    response = await client.post(
        "/v1/forecast",
        content=b'{"horizon": 2, "series": [{"values": [1.0, Infinity]}]}',
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 422


async def test_duplicate_ids(client):
    response = await client.post(
        "/v1/forecast",
        json={
            "horizon": 2,
            "series": [{"id": "a", "values": CONTEXT}, {"id": "a", "values": CONTEXT}],
        },
    )
    assert response.status_code == 422
    assert "must be unique" in _messages(response)


async def test_variates_of_unequal_length(client):
    response = await client.post(
        "/v1/forecast",
        json={"horizon": 2, "series": [{"values": [[1.0, 2.0, 3.0], [1.0, 2.0]]}]},
    )
    assert response.status_code == 422
    assert "same length" in _messages(response)


async def test_inconsistent_variate_count_across_batch(client):
    """TimesFM exige le meme nombre de variables cibles pour tout le lot."""
    response = await client.post(
        "/v1/forecast",
        json={
            "horizon": 2,
            "series": [
                {"id": "a", "values": [[1.0, 2.0], [3.0, 4.0]]},
                {"id": "b", "values": [[1.0, 2.0]]},
            ],
        },
    )
    assert response.status_code == 422
    assert "same number of target variates" in _messages(response)


async def test_too_many_variates(client):
    values = [[1.0, 2.0] for _ in range(65)]
    response = await client.post(
        "/v1/forecast", json={"horizon": 2, "series": [{"values": values}]}
    )
    assert response.status_code == 422
    assert "API_MAX_VARIATES" in _messages(response)


async def test_past_only_covariate_length_mismatch(client):
    response = await client.post(
        "/v1/forecast",
        json={
            "horizon": 2,
            "series": [{"values": CONTEXT, "past_only_covariates": [[1.0, 2.0]]}],
        },
    )
    assert response.status_code == 422
    assert "past_only_covariates" in _messages(response)


async def test_past_future_covariate_misaligned(client):
    """La covariable future doit couvrir le contexte plus l'horizon (ou l'horizon arrondi)."""
    response = await client.post(
        "/v1/forecast",
        json={
            "horizon": 2,
            "series": [{"values": CONTEXT, "past_future_covariates": [[1.0, 2.0, 3.0, 4.0]]}],
        },
    )
    assert response.status_code == 422
    assert "past_future_covariates" in _messages(response)


async def test_past_future_covariate_accepts_rounded_horizon(client):
    """Fournir le contexte plus l'horizon arrondi au patch de 64 est aussi valide."""
    response = await client.post(
        "/v1/forecast",
        json={
            "horizon": 2,
            "series": [{"values": CONTEXT, "past_future_covariates": [[0.0] * (3 + 64)]}],
        },
    )
    assert response.status_code == 200, response.text


async def test_unknown_field_is_rejected(client):
    response = await client.post(
        "/v1/forecast",
        json={"horizon": 2, "series": [{"values": CONTEXT}], "frequency": 0},
    )
    assert response.status_code == 422


async def test_unknown_option_is_rejected(client):
    response = await client.post(
        "/v1/forecast",
        json={"horizon": 2, "series": [{"values": CONTEXT}], "options": {"turbo": True}},
    )
    assert response.status_code == 422


async def test_invalid_padding_mode(client):
    response = await client.post(
        "/v1/forecast",
        json={"horizon": 2, "series": [{"values": CONTEXT}], "options": {"padding_mode": "zero"}},
    )
    assert response.status_code == 422
