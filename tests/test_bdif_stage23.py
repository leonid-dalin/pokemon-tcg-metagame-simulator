import sqlite3

import pytest

from src import bdif_covariates
from src.ingestion.client import LimitlessClient
from src.ingestion.model import fit_model
from src.ingestion.store import LimitlessStore
from src.ingestion.aggregate import build_artifact


@pytest.mark.unit
def test_limitless_client_sends_key_only_as_an_access_header(monkeypatch):
    seen = {}

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return []

    def fake_get(url, **kwargs):
        seen.update(kwargs)
        return Response()

    monkeypatch.setattr("src.ingestion.client.requests.get", fake_get)
    client = LimitlessClient(api_key="secret-value", min_delay=0)
    assert client.tournaments(game="PTCG") == []
    assert seen["headers"]["X-Access-Key"] == "secret-value"
    assert "secret-value" not in seen.get("params", {})
    assert "secret-value" not in seen.get("url", "")


@pytest.mark.unit

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


@pytest.mark.unit
def test_store_upserts_events_and_counts_unique_card_inclusion(tmp_path):
    store = LimitlessStore(tmp_path / "limitless.db")
    event = {"id": "event-1", "game": "PTCG", "format": "STANDARD", "name": "Test", "date": "2026-09-20", "players": 2}
    standings = [
        {"player": "p1", "placing": 1, "record": {"wins": 1}, "deck": {"id": "a"}, "decklist": {"pokemon": [{"name": "Misty", "count": 2}]}},
        {"player": "p2", "placing": 2, "record": {"wins": 0}, "deck": {"id": "b"}, "decklist": {"pokemon": [{"name": "Rocky Energy", "count": 1}]}},
        {"player": "p3", "placing": 3, "record": {"wins": 0}, "deck": {"id": "a"}, "decklist": None},
    ]
    store.upsert_tournament(event, {"decklists": True})
    store.upsert_standings("event-1", standings)
    store.upsert_standings("event-1", standings)

    assert len(list(store.iter_events())) == 1
    assert store.card_inclusion("a") == {"Misty": 1.0}
    with sqlite3.connect(tmp_path / "limitless.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM standings").fetchone()[0] == 3


@pytest.mark.unit
def test_aggregate_normalises_reversed_player_slots_into_one_pair():
    class Store:
        def matchup_rows(self):
            return iter([
                ("Crustle", "N's Zoroark", "p1", "p1", "p2"),
                ("N's Zoroark", "Crustle", "p1", "p2", "p1"),
            ])

        def card_inclusion(self, archetype):
            return {}

    artifact = build_artifact(Store())
    assert artifact["win_rate_matrix"]["Crustle"]["N's Zoroark"] == {
        "win_rate": 1.0,
        "match_count": 2,
    }
    assert artifact["win_rate_matrix"]["N's Zoroark"]["Crustle"] == {
        "win_rate": 0.0,
        "match_count": 2,
    }


@pytest.mark.unit
def test_fitted_card_model_recovers_positive_card_edge():
    observations = [("a", "b", 1)] * 80 + [("b", "a", 0)] * 20
    model = fit_model(
        observations,
        {"a": {"Misty": 1.0}, "b": {"Misty": 0.0}},
    )

    assert model.probability("a", "b") > 0.5
