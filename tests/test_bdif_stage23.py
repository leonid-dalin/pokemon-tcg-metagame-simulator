import sqlite3

import pytest

from src.ingestion.client import LimitlessClient
from src.ingestion.model import Best60Request, fit_h1_misty_variant, fit_model, h1_observations, model_artifact, recommend_best60, select_panel_decks, validate_recommendation
from src.ingestion.store import LimitlessStore
from src.ingestion.aggregate import build_artifact
from src.core.scraper import normalize_archetype


@pytest.mark.unit
def test_limitless_client_sends_key_only_as_an_access_header(monkeypatch):
    seen = {}

    class Response:
        status_code = 200
        content = b"[]"

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
def test_store_writes_missing_decklists_as_sql_null(tmp_path):
    store = LimitlessStore(tmp_path / "limitless.db")
    store.upsert_standings("event", [{"player": "p1", "deck": {"id": "a"}, "decklist": None}])

    with sqlite3.connect(tmp_path / "limitless.db") as conn:
        assert conn.execute("SELECT decklist_json FROM standings").fetchone()[0] is None


def test_store_persists_and_backfills_canonical_deck_names(tmp_path):
    store = LimitlessStore(tmp_path / "limitless.db")
    store.upsert_standings("event", [{"player": "p1", "deck": {"id": "n-zoroark"}}])
    assert store.backfill_deck_names({"n-zoroark": "N's Zoroark"}) == 1
    assert store.deck_weights() == {"N's Zoroark": 1.0}


def test_normalize_archetype_treats_n_possessive_as_canonical_name():
    assert normalize_archetype("N's Zoroark") == normalize_archetype("n zoroark")


@pytest.mark.unit
def test_limitless_client_retries_rate_limit_with_response_contract(monkeypatch):
    responses = iter([
        type("Response", (), {"status_code": 429, "headers": {"Retry-After": "0"}, "content": b"", "json": lambda self: []})(),
        type("Response", (), {"status_code": 200, "headers": {}, "content": b"[]", "json": lambda self: []})(),
    ])
    calls = []

    def fake_get(url, **kwargs):
        calls.append((url, kwargs))
        return next(responses)

    monkeypatch.setattr("src.ingestion.client.requests.get", fake_get)
    client = LimitlessClient(api_key="secret-value", min_delay=0)

    assert client.tournaments(game="PTCG") == []
    assert len(calls) == 2
    assert all(call[1]["timeout"] == client.timeout for call in calls)


@pytest.mark.unit
def test_h1_reads_opponent_mist_energy_and_alakazam_variant_flag(tmp_path):
    store = LimitlessStore(tmp_path / "limitless.db")
    store.upsert_standings("event", [
        {"player": "p1", "deck": {"id": "alakazam"}, "decklist": {"pokemon": [{"name": "Alakazam"}, {"name": "Dedenne"}], "trainer": [{"name": "Enhanced Hammer"}]}},
        {"player": "p2", "deck": {"id": "crustle"}, "decklist": {"energy": [{"name": "Mist Energy"}]}},
    ])
    store.upsert_pairings("event", [{"round": 1, "phase": 1, "player1": "p1", "player2": "p2", "winner": "p2"}])

    assert h1_observations(store.pairings_with_decklists("%alakazam%")) == [{"misty": 1, "hammer_variant": 1, "result": 0}]


@pytest.mark.unit
def test_observed_skeleton_uses_high_frequency_cards(tmp_path):
    store = LimitlessStore(tmp_path / "limitless.db")
    store.upsert_standings("event", [
        {"player": "p1", "deck": {"id": "a"}, "decklist": {"pokemon": [{"name": "Crustle", "count": 2}]}},
        {"player": "p2", "deck": {"id": "a"}, "decklist": {"pokemon": [{"name": "Crustle", "count": 2}]}},
    ])
    assert store.observed_skeleton("a") == [{"card": "Crustle", "copies": 2}]


def test_observed_skeleton_clamps_cards_and_keeps_one_ace_spec(tmp_path):
    store = LimitlessStore(tmp_path / "limitless.db")
    standings = []
    for index in range(4):
        standings.append({
            "player": f"p{index}",
            "deck": {"id": "a"},
            "decklist": {"trainer": [
                {"name": "Weird Card", "count": 9},
                {"name": "Prime Catcher", "count": 1},
                {"name": "Master Ball", "count": 1},
            ]},
        })
    store.upsert_standings("event", standings)
    skeleton = store.observed_skeleton("a")
    assert {row["card"]: row["copies"] for row in skeleton}["Weird Card"] == 4
    assert sum(row["card"] in {"Prime Catcher", "Master Ball"} for row in skeleton) == 1


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


@pytest.mark.unit
def test_best60_ranks_positive_pooled_cards_in_score_order():
    result = recommend_best60(Best60Request(
        archetype="a",
        candidates=["High", "Middle", "Low"],
        coefficients={"High": 3.0, "Middle": 2.0, "Low": 1.0},
        coefficient_intervals={"High": (1.0, 3.0), "Middle": (1.0, 2.0), "Low": (0.5, 1.5)},
        inclusion={"a": {"High": 1.0, "Middle": 1.0, "Low": 1.0}, "b": {}},
        meta_weights={"b": 1.0},
        skeleton=[{"card": "Darkness Energy", "copies": 48}],
    ))
    cards = [row["card"] for row in result["cards"] if row["card"] in {"High", "Middle", "Low"}]
    assert cards == ["High", "Middle", "Low"]


def test_best60_normalizes_skeleton_copy_and_ace_spec_rules():
    result = recommend_best60(Best60Request(
        archetype="a",
        candidates=[],
        coefficients={},
        coefficient_intervals={},
        inclusion={"a": {}, "b": {}},
        meta_weights={"b": 1.0},
        playable_cards={"Darkness Energy"},
        skeleton=[
            {"card": "Weird Card", "copies": 9},
            {"card": "Prime Catcher", "copies": 1},
            {"card": "Master Ball", "copies": 1},
            {"card": "Darkness Energy", "copies": 50},
        ],
    ))
    assert result["total_copies"] == 60
    validate_recommendation(result["cards"])
    assert sum(row["card"] in {"Prime Catcher", "Master Ball"} for row in result["cards"]) <= 1


def test_best60_benjamini_hochberg_gate_filters_weak_candidates():
    cards = [f"Card {index}" for index in range(20)]
    result = recommend_best60(Best60Request(
        archetype="a",
        candidates=cards,
        coefficients={card: 10.0 if index == 0 else 0.01 for index, card in enumerate(cards)},
        coefficient_intervals={card: (0.9, 1.1) for card in cards},
        inclusion={"a": {card: 1.0 for card in cards}, "b": {}},
        meta_weights={"b": 1.0},
        playable_cards=set(cards) | {"Darkness Energy"},
        skeleton=[{"card": "Darkness Energy", "copies": 40}],
    ))
    assert any(row["card"] != "Card 0" for row in result["no_signal"])


@pytest.mark.parametrize("skeleton_size", range(0, 61, 4))
@pytest.mark.parametrize("signal_count", [0, 1, 2, 10])
def test_best60_legality_matrix_returns_deck_or_status(skeleton_size, signal_count):
    signal_cards = [f"Signal {index}" for index in range(signal_count)]
    result = recommend_best60(Best60Request(
        archetype="a",
        candidates=signal_cards,
        coefficients={card: float(signal_count - index) for index, card in enumerate(signal_cards)},
        coefficient_intervals={card: (0.1, 0.2) for card in signal_cards},
        inclusion={"a": {card: 1.0 for card in signal_cards}, "b": {}},
        meta_weights={"b": 1.0},
        playable_cards=set(signal_cards) | {"Darkness Energy"},
        skeleton=[{"card": "Darkness Energy", "copies": skeleton_size}],
    ))
    assert "status" in result or result["total_copies"] == 60
    if "status" not in result:
        validate_recommendation(result["cards"])


@pytest.mark.unit
def test_best60_rejects_two_ace_specs():
    with pytest.raises(ValueError, match="one ACE SPEC"):
        validate_recommendation(
            [{"card": "Prime Catcher", "copies": 1}, {"card": "Master Ball", "copies": 1}, {"card": "Darkness Energy", "copies": 58}]
        )


@pytest.mark.unit
def test_best60_puts_zero_spanning_interval_in_no_signal_bucket():
    result = recommend_best60(Best60Request(
        archetype="a",
        candidates=["Uncertain"],
        coefficients={"Uncertain": 0.1},
        coefficient_intervals={"Uncertain": (-0.2, 0.2)},
        inclusion={"a": {"Uncertain": 1.0}, "b": {"Uncertain": 0.0}},
        meta_weights={"b": 1.0},
        skeleton=[{"card": "Darkness Energy", "copies": 59}],
    ))

    assert result["total_copies"] == 60
    assert {row["card"] for row in result["no_signal"]} >= {"Uncertain"}


@pytest.mark.unit
def test_best60_rejects_five_copies():
    with pytest.raises(ValueError, match="copy limit"):
        validate_recommendation([{"card": "Too Many", "copies": 5}])


@pytest.mark.unit
def test_recommendation_requires_exactly_sixty_cards():
    with pytest.raises(ValueError, match="exactly 60"):
        validate_recommendation([{"card": "Crustle", "copies": 4}])


@pytest.mark.unit
def test_panel_decks_include_every_empirical_share_above_threshold():
    assert select_panel_decks({"a": 0.031, "b": 0.03, "c": 0.029}, threshold=0.03) == ["a", "b"]


@pytest.mark.unit
def test_best60_reports_card_evidence_and_partial_status():
    result = recommend_best60(Best60Request(
        archetype="a",
        candidates=["Signal Card"],
        coefficients={"Signal Card": 2.0},
        coefficient_intervals={"Signal Card": (1.0, 3.0)},
        inclusion={"a": {"Signal Card": 1.0}, "b": {"Signal Card": 0.0}},
        meta_weights={"b": 1.0},
        playable_cards={"Signal Card"},
        skeleton=[{"card": "Signal Card", "copies": 1}],
    ))
    assert result["status"] == "insufficient legal observed cards to complete 60"
    assert result["card_evidence"]["Signal Card"] == {
        "inclusion_rate": 1.0,
        "field_inclusion_rate": 0.0,
        "inclusion_delta": 1.0,
        "coefficient": 2.0,
        "contribution": 2.0,
        "interval": (1.0, 3.0),
    }


@pytest.mark.unit
def test_basic_energy_is_not_limited_to_four_copies():
    validate_recommendation([{"card": "Darkness Energy", "copies": 60}])


@pytest.mark.unit
def test_h1_misty_report_changes_when_hammer_variant_is_controlled():
    observations = []
    observations.extend([{"misty": 1, "hammer_variant": 0, "result": 1}] * 30)
    observations.extend([{"misty": 0, "hammer_variant": 0, "result": 0}] * 5)
    observations.extend([{"misty": 1, "hammer_variant": 1, "result": 0}] * 10)
    observations.extend([{"misty": 0, "hammer_variant": 1, "result": 1}] * 10)

    report = fit_h1_misty_variant(observations)
    assert report["without_variant"]["beta"] > 0
    assert report["with_variant"]["beta"] <= 0
    assert report["without_variant"]["interval"] == pytest.approx((-0.4266468, 0.9916069), abs=1e-6)
    assert report["with_variant"]["interval"] == pytest.approx((-0.9108607, 0.7331841), abs=1e-6)
    assert report["interpretation"] == "observational association, not a causal effect"

def test_h1_standard_error_uses_weighted_fisher_information():
    observations = [{"misty": 1, "hammer_variant": 0, "result": 1}] * 30
    observations.extend([{"misty": 0, "hammer_variant": 0, "result": 0}] * 5)
    observations.extend([{"misty": 1, "hammer_variant": 1, "result": 0}] * 10)
    observations.extend([{"misty": 0, "hammer_variant": 1, "result": 1}] * 10)
    report = fit_h1_misty_variant(observations)
    assert report["without_variant"]["interval"][1] - report["without_variant"]["interval"][0] < 2.0


def test_card_model_artifact_carries_observation_counts():
    model = fit_model(
        [("a", "b", 1)] * 5 + [("b", "a", 0)] * 5,
        {"a": {"Misty": 1.0}, "b": {"Misty": 0.0}},
    )
    artifact = model_artifact(model)
    assert artifact["win_rate_matrix"]["a"]["b"]["match_count"] == 10
    assert artifact["win_rate_matrix"]["b"]["a"]["match_count"] == 10
