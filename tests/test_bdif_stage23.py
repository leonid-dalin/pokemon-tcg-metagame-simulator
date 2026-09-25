import sqlite3
from pathlib import Path

import pytest

from src.bdif import service
from src.bdif.settings import BdifSettings
from src.ingestion.client import LimitlessClient
from src.ingestion.model import Best60Request, CardModelNotIdentifiable, PlayerObservation, fit_card_model, fit_h1_misty_variant, h1_observations, model_artifact, recommend_best60, select_model_cards, select_panel_decks, validate_recommendation
from src.api.models import PredictionRequest
from src.ingestion.store import LimitlessStore, _decklist_card_names
from src.ingestion.aggregate import build_artifact
from src.ingestion.mapping import coverage_report, extract_recorded_deck_ids, resolve_archetype
from src.core.scraper import normalize_archetype
SYNTHETIC_DECK_MAPPING = {"a": "a", "alakazam": "Alakazam", "b": "b", "crustle": "Crustle"}



@pytest.mark.unit
def test_recorded_limitless_deck_ids_have_explicit_mapping_coverage():
    fixture = Path("data/Decks_ Regional Championship Prague – Limitless Labs.htm")
    observed = extract_recorded_deck_ids(fixture)
    report = coverage_report(observed)

    assert len(observed) == 59
    assert report.unmapped == ["farigiraf-milotic", "ogerpon-box", "other"]
    assert report.mapped == sorted(observed - set(report.unmapped))
    assert resolve_archetype("crustle-dri") == "Crustle"


@pytest.mark.unit
def test_baltimore_limitless_deck_ids_have_explicit_mapping_coverage():
    fixture = Path("data/input/limitless_baltimore_0072_decks.html")
    observed = extract_recorded_deck_ids(fixture)
    report = coverage_report(observed)

    assert len(observed) == 90
    assert report.unmapped == ["conkeldurr-twm", "other"]
    assert report.mapped == sorted(observed - set(report.unmapped))
    assert resolve_archetype("mega-abomasnow-ex") == "Mega Abomasnow"


@pytest.mark.unit
def test_ingestion_reports_newly_observed_unmapped_deck_ids(monkeypatch, tmp_path):
    class Client:
        @classmethod
        def from_environment(cls):
            return cls()

        def game_decks(self):
            return []

        def iter_tournaments(self, **params):
            return [{"id": "new-event"}]

        def fetch_event_bundle(self, event_id):
            return ({"decklists": True}, [{"player": "p1", "deck": {"id": "newly-observed-id"}}], [])

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("src.ingestion.client.LimitlessClient", Client)
    monkeypatch.setattr("src.ingestion.aggregate.build_artifact", lambda store: {"archetypes": []})

    result = service.run_ingestion(BdifSettings(
        use_card_model=True,
        ingestion_enabled=True,
        db_path=str(tmp_path / "limitless.db"),
        baseline_input_path="input.json",
        ingestion_input_path=str(tmp_path / "limitless_input.json"),
        model_input_path=str(tmp_path / "limitless_model_input.json"),
        panel_share_threshold=0.01,
        panel_max_decks=10,
        fallback_panel_decks=("fallback",),
        backfill_limit=10,
    ))

    assert result["unmapped_deck_ids"] == ["newly-observed-id"]


@pytest.mark.unit
def test_unknown_single_token_limitless_ids_stay_unmapped(tmp_path):
    store = LimitlessStore(tmp_path / "limitless.db", canonical_names=["Known"])

    assert store._resolve_deck_name("unknown", "Unknown") is None


@pytest.mark.unit
def test_unknown_limitless_deck_ids_are_reported_as_unmapped():
    report = coverage_report({"crustle-dri", "unknown-deck"})

    assert report.mapped == ["crustle-dri"]
    assert report.unmapped == ["unknown-deck"]


@pytest.mark.unit
def test_player_observations_include_each_players_card_presence_and_skip_invalid_pairings(tmp_path):
    store = LimitlessStore(tmp_path / "limitless.db", deck_mapping=SYNTHETIC_DECK_MAPPING)
    store.upsert_standings("event", [
        {"player": "p1", "deck": {"id": "a"}, "decklist": {"pokemon": [{"name": "A"}, {"name": "Shared"}], "trainer": [{"name": "Shared"}]}},
        {"player": "p2", "deck": {"id": "b"}, "decklist": {"energy": [{"name": "B"}]}},
        {"player": "p3", "deck": {"id": "a"}, "decklist": None},
        {"player": "p4", "deck": {"id": "b"}, "decklist": {}},
    ])
    store.upsert_pairings("event", [
        {"round": 1, "player1": "p1", "player2": "p2", "winner": "p2"},
        {"round": 2, "player1": "p1", "player2": "p2", "winner": "0"},
        {"round": 3, "player1": "p3", "player2": "p2", "winner": "p3"},
        {"round": 4, "player1": "p1", "player2": "p4", "winner": "p1"},
        {"round": 5, "player1": "p1", "player2": "missing", "winner": "p1"},
    ])

    assert store.player_observations() == [
        PlayerObservation("a", "b", frozenset({"A", "Shared"}), frozenset({"B"}), 0),
    ]


@pytest.mark.parametrize("raw", [None, "", "null", "[]", "{", "{\"pokemon\": {}}"])
def test_decklist_card_names_returns_empty_for_invalid_payloads(raw):
    assert _decklist_card_names(raw) == frozenset()


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
    store = LimitlessStore(tmp_path / "limitless.db", deck_mapping=SYNTHETIC_DECK_MAPPING)
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
    store = LimitlessStore(tmp_path / "limitless.db", deck_mapping=SYNTHETIC_DECK_MAPPING)
    store.upsert_standings("event", [{"player": "p1", "deck": {"id": "a"}, "decklist": None}])

    with sqlite3.connect(tmp_path / "limitless.db") as conn:
        assert conn.execute("SELECT decklist_json FROM standings").fetchone()[0] is None


@pytest.mark.unit
def test_store_construction_does_not_modify_existing_database(tmp_path):
    path = tmp_path / "limitless.db"
    store = LimitlessStore(path, canonical_names=["N's Zoroark"])
    store.upsert_standings("event", [{"player": "p1", "deck": {"id": "n-zoroark"}}])
    before = path.read_bytes()

    LimitlessStore(path, canonical_names=["N's Zoroark"])

    assert path.read_bytes() == before


@pytest.mark.unit
def test_readers_prepare_legacy_store_schema(tmp_path):
    path = tmp_path / "legacy.db"
    with sqlite3.connect(path) as conn:
        conn.executescript(
            "CREATE TABLE standings (tournament_id TEXT, player_id TEXT, deck_id TEXT, decklist_json TEXT);"
            "CREATE TABLE pairings (tournament_id TEXT, player1 TEXT, player2 TEXT, winner TEXT);"
        )
        conn.execute("INSERT INTO standings VALUES ('event', 'p1', 'a', '{\"pokemon\": []}')")
        conn.commit()

    store = LimitlessStore(path, canonical_names=["a"], deck_mapping=SYNTHETIC_DECK_MAPPING)
    assert list(store.matchup_rows()) == []

    with sqlite3.connect(path) as conn:
        assert "deck_name" in {row[1] for row in conn.execute("PRAGMA table_info(standings)")}
        assert conn.execute("SELECT deck_name FROM standings").fetchone()[0] == "a"


@pytest.mark.unit
def test_h1_observations_treats_legacy_json_null_as_empty_deck():
    rows = [("alakazam", "crustle", '{"pokemon": [{"name": "Alakazam"}]}', "null", "p2", "p1", "p2")]
    assert h1_observations(rows) == [{"misty": 0, "hammer_variant": 0, "result": 0}]


@pytest.mark.unit
def test_store_persists_and_backfills_canonical_deck_names(tmp_path):
    store = LimitlessStore(tmp_path / "limitless.db", canonical_names=["N's Zoroark"])
    store.upsert_standings("event", [
        {"player": "p1", "deck": {"id": "n-zoroark", "name": "N's Zoroark"}},
        {"player": "p2", "deck": {"id": "other", "name": "Other"}},
    ])

    with sqlite3.connect(tmp_path / "limitless.db") as conn:
        assert conn.execute("SELECT deck_name FROM standings").fetchone()[0] == "N's Zoroark"
        conn.execute("UPDATE standings SET deck_name=NULL")
        conn.commit()

    store.backfill_deck_names()
    assert store.deck_weights() == {"N's Zoroark": 1.0}

    with sqlite3.connect(tmp_path / "limitless.db") as conn:
        assert conn.execute("SELECT deck_name FROM standings").fetchone()[0] == "N's Zoroark"


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
def test_limitless_client_paginates_tournaments(monkeypatch):
    pages = iter([
        [{"id": "one"}, {"id": "two"}],
        [{"id": "three"}],
    ])
    calls = []

    class Response:
        status_code = 200
        headers = {}
        content = b"[]"

        def json(self):
            return next(pages)

    def fake_get(url, **kwargs):
        calls.append(kwargs["params"])
        return Response()

    monkeypatch.setattr("src.ingestion.client.requests.get", fake_get)
    client = LimitlessClient(min_delay=0)

    assert list(client.iter_tournaments(game="PTCG", format="STANDARD", limit=3, page_size=2)) == [
        {"id": "one"}, {"id": "two"}, {"id": "three"},
    ]
    assert calls == [
        {"game": "PTCG", "format": "STANDARD", "limit": 2, "page": 1},
        {"game": "PTCG", "format": "STANDARD", "limit": 2, "page": 2},
    ]


@pytest.mark.unit
def test_fetch_event_bundle_without_decklists_preserves_pairings(monkeypatch):
    client = LimitlessClient(min_delay=0)
    calls = []
    monkeypatch.setattr(client, "event_details", lambda event_id: {"id": event_id, "decklists": False})
    monkeypatch.setattr(client, "standings", lambda event_id: calls.append("standings") or [])
    monkeypatch.setattr(client, "pairings", lambda event_id: calls.append("pairings") or [{"player1": "p1", "player2": "p2"}])

    details, standings, pairings = client.fetch_event_bundle("event")

    assert details["decklists"] is False
    assert standings == []
    assert pairings == [{"player1": "p1", "player2": "p2"}]
    assert calls == ["pairings"]


@pytest.mark.unit
def test_h1_reads_opponent_mist_energy_and_alakazam_variant_flag(tmp_path):
    store = LimitlessStore(tmp_path / "limitless.db", deck_mapping=SYNTHETIC_DECK_MAPPING)
    store.upsert_standings("event", [
        {"player": "p1", "deck": {"id": "alakazam"}, "decklist": {"pokemon": [{"name": "Alakazam"}, {"name": "Dedenne"}], "trainer": [{"name": "Enhanced Hammer"}]}},
        {"player": "p2", "deck": {"id": "crustle"}, "decklist": {"energy": [{"name": "Mist Energy"}]}},
    ])
    store.upsert_pairings("event", [{"round": 1, "phase": 1, "player1": "p1", "player2": "p2", "winner": "p2"}])

    assert h1_observations(store.pairings_with_decklists("%alakazam%")) == [{"misty": 1, "hammer_variant": 1, "result": 0}]


@pytest.mark.unit
def test_observed_skeleton_uses_high_frequency_cards(tmp_path):
    store = LimitlessStore(tmp_path / "limitless.db", deck_mapping=SYNTHETIC_DECK_MAPPING)
    store.upsert_standings("event", [
        {"player": "p1", "deck": {"id": "a"}, "decklist": {"pokemon": [{"name": "Crustle", "count": 2}]}},
        {"player": "p2", "deck": {"id": "a"}, "decklist": {"pokemon": [{"name": "Crustle", "count": 2}]}},
    ])
    assert store.observed_skeleton("a") == [{"card": "Crustle", "copies": 2}]


def test_observed_skeleton_clamps_cards_and_keeps_one_ace_spec(tmp_path):
    store = LimitlessStore(tmp_path / "limitless.db", deck_mapping=SYNTHETIC_DECK_MAPPING)
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
def test_fitted_card_model_recovers_within_archetype_tech_effect():
    observations = []
    for index in range(120):
        has_tech = index % 2 == 0
        a_cards = frozenset({"Core", "Tech"} if has_tech else {"Core"})
        b_cards = frozenset({"Core"})
        result = int(index % 10 < (8 if has_tech else 2))
        result = 1 - result if index % 11 == 0 else result
        observations.append(PlayerObservation("a", "b", a_cards, b_cards, result))
        observations.append(PlayerObservation("b", "a", b_cards, a_cards, 1 - result))
    model = fit_card_model(observations, select_model_cards(observations))

    coefficients, intervals = model.coefficient_report()
    assert coefficients["Tech"] > 0
    assert intervals["Tech"][0] > 0


@pytest.mark.unit
def test_model_artifact_zero_sum_matrix_is_accepted_by_prediction_request():
    observations = []
    for index in range(120):
        has_tech = index % 2 == 0
        a_cards = frozenset({"Core", "Tech", "Filler"} if has_tech else {"Core", "Filler"})
        b_cards = frozenset({"Core", "Filler"} if index % 3 == 0 else {"Core"})
        result = int(index % 10 < (8 if has_tech else 2))
        result = 1 - result if index % 11 == 0 else result
        observations.append(PlayerObservation("a", "b", a_cards, b_cards, result))
        observations.append(PlayerObservation("b", "a", b_cards, a_cards, 1 - result))
    artifact = model_artifact(fit_card_model(observations, ["Filler", "Tech"]))

    request = PredictionRequest(**{
        "job_id": "job",
        "deck_names": artifact["archetypes"],
        "matchup_matrix": [[artifact["win_rate_matrix"][left][right]["win_rate"] for right in artifact["archetypes"]] for left in artifact["archetypes"]],
        "total_players": 4,
    })
    assert request.matchup_matrix[0][1] + request.matchup_matrix[1][0] == pytest.approx(1.0)


@pytest.mark.unit
def test_select_model_cards_drops_constant_presence_card():
    observations = [PlayerObservation("a", "b", frozenset({"Signature"}), frozenset({"Signature"}), index % 2) for index in range(20)]
    assert "Signature" not in select_model_cards(observations)


@pytest.mark.unit
def test_rank_deficient_card_raises_not_identifiable():
    observations = [PlayerObservation("a", "b", frozenset({"Tech"}), frozenset(), index % 2) for index in range(20)]
    with pytest.raises(CardModelNotIdentifiable):
        fit_card_model(observations, ["Signature", "Tech"])


@pytest.mark.unit
def test_separated_card_outcomes_raise_not_identifiable():
    observations = []
    for index in range(200):
        result = int(index < 199)
        observations.append(PlayerObservation("a", "b", frozenset({"Tech"}), frozenset(), result))
        observations.append(PlayerObservation("b", "a", frozenset({"Tech"}), frozenset(), result))
        observations.append(PlayerObservation("a", "b", frozenset(), frozenset({"Tech"}), 1 - result))
        observations.append(PlayerObservation("b", "a", frozenset(), frozenset({"Tech"}), 1 - result))
    with pytest.raises(CardModelNotIdentifiable, match="separated outcomes"):
        fit_card_model(observations, ["Tech"])


@pytest.mark.unit
def test_card_model_fit_has_no_intercept():
    observations = (
        [PlayerObservation("a", "b", frozenset({"Tech"}), frozenset(), 1)] * 30
        + [PlayerObservation("a", "b", frozenset(), frozenset({"Tech"}), 0)] * 10
        + [PlayerObservation("b", "a", frozenset({"Tech"}), frozenset(), 0)] * 5
        + [PlayerObservation("b", "a", frozenset(), frozenset({"Tech"}), 1)] * 15
    )
    model = fit_card_model(observations, ["Tech"])
    assert model.estimator.intercept_.tolist() == [0.0]
    assert model.estimator.coef_[0, 1] == pytest.approx(0.6931471805599453, abs=1e-5)


@pytest.mark.unit
def test_card_model_artifact_carries_player_observation_counts():
    observations = [PlayerObservation("a", "b", frozenset({"Core"}), frozenset({"Core"}), 1)] * 10
    observations += [PlayerObservation("b", "a", frozenset({"Core"}), frozenset({"Core"}), 0)] * 10
    artifact = model_artifact(fit_card_model(observations, ["Tech"]))
    assert artifact["win_rate_matrix"]["a"]["b"]["match_count"] == 10
    assert artifact["win_rate_matrix"]["b"]["a"]["match_count"] == 10


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


def test_best60_fill_target_preserves_exact_sixty_card_contract():
    result = recommend_best60(Best60Request(
        archetype="a",
        candidates=[],
        coefficients={},
        coefficient_intervals={},
        inclusion={"a": {}, "b": {}},
        meta_weights={"b": 1.0},
        playable_cards={"Darkness Energy"},
        skeleton=[{"card": "Darkness Energy", "copies": 1}],
    ))
    assert result["total_copies"] == 60
    validate_recommendation(result["cards"])


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
def test_panel_decks_cap_at_ten_after_share_threshold():
    shares = {f"deck-{index}": 0.04 - index / 1000 for index in range(12)}
    assert len(select_panel_decks(shares, threshold=0.03)) == 10


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
        "q_value": pytest.approx(8.854896862420247e-05),
        "bucket": "signal",
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


def test_card_model_artifact_carries_player_observation_counts(monkeypatch):
    monkeypatch.setattr("src.ingestion.model.BDIF_CARD_MIN_PLAYERS", 1)
    observations = []
    for index in range(10):
        has_tech = index % 2 == 0
        a_cards = frozenset({"Core", "Tech", "Filler"} if has_tech else {"Core", "Filler"})
        b_cards = frozenset({"Core"})
        result = 1 if index % 4 in {0, 1} else 0
        observations.append(PlayerObservation("a", "b", a_cards, b_cards, result))
        observations.append(PlayerObservation("b", "a", b_cards, a_cards, 1 - result))
    artifact = model_artifact(fit_card_model(observations, ["Tech"]))
    assert artifact["win_rate_matrix"]["a"]["b"]["match_count"] == 10
    assert artifact["win_rate_matrix"]["b"]["a"]["match_count"] == 10
