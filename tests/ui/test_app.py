import json

import numpy as np
import pytest
import requests

from src.api.models import PredictionRequest
from src.ui import app


@pytest.mark.parametrize(
    ("token", "expected"),
    [(None, {}), ("", {}), ("secret", {"X-API-Token": "secret"})],
)
def test_api_headers_reflects_optional_api_token(monkeypatch, token, expected):
    if token is None:
        monkeypatch.delenv("API_TOKEN", raising=False)
    else:
        monkeypatch.setenv("API_TOKEN", token)

    assert app.api_headers() == expected


def test_submit_prediction_passes_a_timeout(monkeypatch):
    seen = {}

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"task_id": "task-1"}

    def post(url, **kwargs):
        seen["url"] = url
        seen.update(kwargs)
        return Response()

    monkeypatch.setattr(app.requests, "post", post)
    request_model = PredictionRequest(
        deck_names=["a", "b"],
        matchup_matrix=[[0.5, 0.6], [0.4, 0.5]],
    )

    assert app.submit_prediction("http://api.test", request_model) == "task-1"
    assert seen["url"] == "http://api.test/predict"
    assert seen["timeout"] == 45


def test_submit_prediction_passes_api_headers(monkeypatch):
    seen = {}

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"task_id": "task-1"}

    def post(url, **kwargs):
        seen.update(kwargs)
        return Response()

    monkeypatch.setenv("API_TOKEN", "secret")
    monkeypatch.setattr(app.requests, "post", post)
    request_model = PredictionRequest(
        deck_names=["a", "b"],
        matchup_matrix=[[0.5, 0.6], [0.4, 0.5]],
    )

    app.submit_prediction("http://api.test", request_model)

    assert seen["headers"] == {"X-API-Token": "secret"}


@pytest.mark.parametrize("failure", [requests.exceptions.ConnectionError, requests.exceptions.HTTPError, json.JSONDecodeError])
def test_fetch_bdif_status_degrades_on_request_failures(monkeypatch, failure):
    def get(*args, **kwargs):
        raise failure("status unavailable", "", 0)

    monkeypatch.setattr(app.requests, "get", get)

    assert app.fetch_bdif_status("http://api.test") is None


def test_fetch_bdif_status_returns_json(monkeypatch):
    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"status": "available"}

    monkeypatch.setattr(app.requests, "get", lambda *args, **kwargs: Response())

    assert app.fetch_bdif_status("http://api.test") == {"status": "available"}


def test_resolve_auto_fill_reports_infeasible_constraints():
    filled, error = app.resolve_auto_fill(
        np.array([1.0, 0.0]),
        {"a": 0.8, "b": 0.8},
        {"a": 0, "b": 1},
    )

    assert filled is None
    assert error == "cannot satisfy minimum constraints"


@pytest.mark.unit
def test_bdif_tabs_render_status_evidence_and_provenance():
    from streamlit.testing.v1 import AppTest

    def page():
        from src.ui.app import render_bdif_tabs

        render_bdif_tabs({
            "matchup_panel": {"rows": {}, "unmatched": [], "opponents": []},
            "best60_recommendations": {
                "A": {"cards": [], "status": "missing observed skeleton", "no_signal": [], "card_evidence": {}}
            },
            "h1_report": {},
            "field_posterior": {
                "A": {
                    "expected_win_rate": 0.5,
                    "expected_win_rate_lower": 0.49,
                    "expected_win_rate_upper": 0.51,
                    "best_pick_probability": 1.0,
                }
            },
            "insufficient_data": [],
            "posterior": {"draws": 200, "interval_status": "ok"},
            "provenance": {"simulation_input": {"path": "data/input/ea_input.json", "exists": True}, "seed": 1312},
        })

    at = AppTest.from_function(page).run(timeout=60)

    assert not at.exception
    assert [tab.label for tab in at.tabs] == [
        "Field posterior",
        "BDIF matchup panel",
        "Best-60 card recommendations",
        "H1 report",
        "Provenance",
    ]
    assert any("missing observed skeleton" in warning.value for warning in at.warning)
