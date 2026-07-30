"""Tests for the prediction service.

The validation tests matter more than the happy path here. The model applies a
real coefficient to whatever number it is handed, so input that is missing,
negative, or unrecognised has to be rejected at the boundary rather than turned
into a confident answer.
"""

import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def match_state(**overrides):
    """A valid minute-10 snapshot, roughly even, with an early gold lead."""
    payload = {
        "gold_diff": 1200,
        "exp_diff": 800,
        "champ_level_diff": 0.4,
        "is_first_tower": False,
        "is_first_blood": True,
        "killed_fire_drake": 1,
        "killed_water_drake": 0,
        "killed_air_drake": 0,
        "killed_earth_drake": 0,
        "lost_fire_drake": 0,
        "lost_water_drake": 0,
        "lost_air_drake": 0,
        "lost_earth_drake": 0,
        "killed_rift_herald": 1,
        "lost_rift_herald": 0,
        "destroyed_top_inner_turret": 0,
        "destroyed_mid_inner_turret": 0,
        "destroyed_bot_inner_turret": 0,
        "lost_top_inner_turret": 0,
        "lost_mid_inner_turret": 0,
        "lost_bot_inner_turret": 0,
        "destroyed_top_outer_turret": 0,
        "destroyed_mid_outer_turret": 0,
        "destroyed_bot_outer_turret": 0,
        "lost_top_outer_turret": 0,
        "lost_mid_outer_turret": 0,
        "lost_bot_outer_turret": 0,
        "kills": 5,
        "deaths": 3,
        "assists": 7,
        "wards_placed": 20,
        "wards_destroyed": 4,
        "wards_lost": 6,
    }
    payload.update(overrides)
    return payload


def test_health_reports_a_loaded_model(client):
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["n_features"] == 33


def test_predict_returns_a_probability(client):
    response = client.post("/predict", json=match_state())

    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["win_probability"] <= 1.0
    assert isinstance(body["will_win"], bool)
    assert body["will_win"] == (body["win_probability"] >= 0.5)


def test_a_gold_lead_beats_a_gold_deficit(client):
    """Sanity check that the wiring is not scrambling the feature order."""
    ahead = client.post("/predict", json=match_state(gold_diff=6000, exp_diff=4000))
    behind = client.post("/predict", json=match_state(gold_diff=-6000, exp_diff=-4000))

    assert ahead.json()["win_probability"] > behind.json()["win_probability"]


def test_missing_field_is_rejected(client):
    payload = match_state()
    del payload["gold_diff"]

    assert client.post("/predict", json=payload).status_code == 422


def test_negative_count_is_rejected(client):
    """Counts cannot be negative, and a silent -3 kills would still predict."""
    assert client.post("/predict", json=match_state(kills=-3)).status_code == 422


def test_unknown_field_is_rejected(client):
    """gameDuration is excluded by design, so it must not be quietly ignored."""
    response = client.post("/predict", json=match_state(gameDuration=1443000))

    assert response.status_code == 422


def test_wrong_type_is_rejected(client):
    assert client.post("/predict", json=match_state(kills="lots")).status_code == 422


def test_response_carries_a_request_id(client):
    response = client.get("/health")

    assert response.headers["x-request-id"]


def test_inbound_request_id_is_honoured(client):
    """A trace started upstream should survive rather than restarting here."""
    response = client.get("/health", headers={"x-request-id": "trace-abc-123"})

    assert response.headers["x-request-id"] == "trace-abc-123"


def test_unexpected_failure_returns_500_without_leaking_detail():
    """A broken model must not return internals in the response body."""

    class ExplodingModel:
        def predict_proba(self, _):
            raise RuntimeError("weights are corrupt at /secret/path")

    with TestClient(app, raise_server_exceptions=False) as broken:
        broken.app.state.model = ExplodingModel()
        response = broken.post("/predict", json=match_state())

    assert response.status_code == 500
    assert "corrupt" not in response.text
    assert "secret" not in response.text
