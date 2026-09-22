import numpy as np
import pytest

from src.api.models import PredictionRequest
from src.ui import app


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


def test_resolve_auto_fill_reports_infeasible_constraints():
    filled, error = app.resolve_auto_fill(
        np.array([1.0, 0.0]),
        {"a": 0.8, "b": 0.8},
        {"a": 0, "b": 1},
    )

    assert filled is None
    assert error == "cannot satisfy minimum constraints"
