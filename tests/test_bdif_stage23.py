import json

import pytest

from src import bdif_covariates, bdif_ingestion


@pytest.mark.unit
def test_limitless_client_sends_key_only_as_an_access_header(monkeypatch):
    seen = {}

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"events": []}

    def fake_get(url, **kwargs):
        seen.update(kwargs)
        return Response()

    monkeypatch.setattr(bdif_ingestion.requests, "get", fake_get)
    client = bdif_ingestion.LimitlessClient(api_key="secret-value")
    assert client.tournaments(game="PTCG") == {"events": []}
    assert seen["headers"]["X-Access-Key"] == "secret-value"
    assert "secret-value" not in seen.get("params", {})
    assert "secret-value" not in seen.get("url", "")


@pytest.mark.unit
def test_event_snapshot_stores_event_date_archetype_and_inclusion_rates(tmp_path):
    destination = bdif_ingestion.store_event_snapshot(
        tmp_path,
        {
            "event_id": "event-1",
            "event_date": "2026-09-20",
            "archetype": "Alakazam Dudunsparce",
            "cards": ["Misty", "Misty", "Rocky Energy"],
        },
    )
    record = json.loads(destination.read_text(encoding="utf-8"))
    assert record["inclusion_rates"] == {"Misty": 2 / 3, "Rocky Energy": 1 / 3}
    assert "secret" not in record


@pytest.mark.unit
def test_card_covariate_model_is_disabled_by_default(monkeypatch):
    monkeypatch.delenv("BDIF_CARD_COVARIATES_ENABLED", raising=False)
    model = bdif_covariates.CardCovariateModel({}, {})
    assert model.enabled is False
    with pytest.raises(RuntimeError, match="disabled"):
        model.logit_probability("a", "b", {}, {})


@pytest.mark.unit
def test_card_covariate_model_uses_inclusion_differences_when_enabled():
    model = bdif_covariates.CardCovariateModel(
        strengths={"a": 0.0, "b": 0.0},
        coefficients={"Misty": 2.0},
        enabled=True,
    )
    probability = model.logit_probability("a", "b", {"Misty": 1.0}, {})
    assert probability > 0.8
