import json
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from src.ingestion.labs import DECK_SIZE, LabsEvent, _RateLimiter, collect_event, decklist_size, import_bundles, validate_bundle

MAPPING = {"dragapult-ex": "Dragapult"}


def _decklist(size: int = DECK_SIZE):
    return {"pokemon": [{"name": "Mega Kangaskhan ex", "count": size}]}


def _bundle(*, labs_id: str = "0075", public_id: str = "580", city: str = "Recife", list_size: int = DECK_SIZE):
    standings = [
        {
            "player_id": 100,
            "tp_id": 1,
            "placement": 1,
            "wins": 14,
            "losses": 2,
            "ties": 1,
            "drop_round": None,
            "decklist": 1,
            "deck_id": "dragapult-ex",
            "deck_name": "Dragapult",
        },
        {
            "player_id": 200,
            "tp_id": 2,
            "placement": 2,
            "wins": 12,
            "losses": 4,
            "ties": 1,
            "drop_round": 3,
            "decklist": 1,
            "deck_id": "unknown-id",
            "deck_name": "Mystery Box",
        },
    ]
    pairings = [
        {"round": 1, "phase": 1, "p1_id": 100, "p2_id": 200, "winner": 2},
        {"round": 2, "phase": 1, "p1_id": 200, "p2_id": 100, "winner": 0},
        {"round": 3, "phase": 1, "p1_id": 100, "p2_id": 200, "winner": -1},
    ]
    return {
        "labs_id": labs_id,
        "public_id": public_id,
        "city": city,
        "date": "2026-10-03T00:00:00.000Z",
        "metadata": {"players": 1074, "round": 3, "city": city, "type": "regional"},
        "standings": standings,
        "pairings": pairings,
        "decklists": {"100": _decklist(list_size), "200": _decklist(list_size)},
    }


@pytest.mark.unit
def test_event_spec_parses_a_valid_spec():
    event = LabsEvent.parse("0075:580:Recife:2026-10-03T00:00:00.000Z")

    assert event.labs_id == "0075"
    assert event.public_id == "580"
    assert event.name == "Regional Recife"
    assert event.source_url == "https://limitlesstcg.com/tournaments/580"


@pytest.mark.unit
def test_event_spec_stores_a_bare_date_in_the_form_the_online_path_writes():
    bare = LabsEvent.parse("0077:582:Nice:2026-10-17")
    full = LabsEvent.parse("0077:582:Nice:2026-10-17T00:00:00.000Z")

    assert bare.date == "2026-10-17T00:00:00.000Z"
    assert bare.date == full.date


@pytest.mark.unit
@pytest.mark.parametrize("spec", ["0075:580:Recife", "0075:580::2026-10-03T00:00:00.000Z", "", "a:b:c:d:e"])
def test_event_spec_rejects_a_malformed_spec(spec):
    with pytest.raises(ValueError):
        LabsEvent.parse(spec)


@pytest.mark.unit
def test_decklist_size_counts_cards_and_rejects_malformed_payloads():
    assert decklist_size(_decklist(60)) == 60
    assert decklist_size(_decklist(59)) == 59
    assert decklist_size({"pokemon": [{"name": "Card", "count": 0}]}) is None
    assert decklist_size({"pokemon": [{"name": "", "count": 1}]}) is None
    assert decklist_size({"pokemon": "not-a-list"}) is None
    assert decklist_size(None) is None


@pytest.mark.unit
def test_validate_bundle_accepts_a_complete_bundle():
    report = validate_bundle(_bundle())

    assert report["ok"] is True
    assert report["problems"] == []
    assert report["players"] == 2
    assert report["pairings"] == 3
    assert report["lists_valid"] == 2


@pytest.mark.unit
def test_validate_bundle_flags_a_short_list_and_an_unknown_participant():
    bundle = _bundle(list_size=59)
    bundle["pairings"][0]["p2_id"] = 999

    report = validate_bundle(bundle)

    assert report["ok"] is False
    assert any("not 60 cards" in problem for problem in report["problems"])
    assert any("absent from standings" in problem for problem in report["problems"])


@pytest.mark.unit
def test_validate_bundle_flags_a_duplicate_pairing_identity():
    bundle = _bundle()
    bundle["pairings"][1] = {**bundle["pairings"][0]}

    report = validate_bundle(bundle)

    assert report["ok"] is False
    assert any("duplicate pairing identities" in problem for problem in report["problems"])


@pytest.mark.unit
def test_validate_bundle_fails_when_pairings_do_not_cover_every_round():
    bundle = _bundle()
    bundle["metadata"] = {**bundle["metadata"], "round": 4}

    report = validate_bundle(bundle)

    assert report["ok"] is False
    assert any("missing rounds" in problem and "[4]" in problem for problem in report["problems"])


@pytest.mark.unit
def test_collect_event_resumes_by_fetching_only_the_missing_rounds(tmp_path):
    directory = tmp_path / "0099"
    directory.mkdir()
    metadata = {"players": 4, "round": 4, "city": "Testville", "type": "regional"}
    (directory / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    standings = [
        {
            "player_id": 1,
            "tp_id": 1,
            "placement": 1,
            "wins": 2,
            "losses": 0,
            "ties": 0,
            "drop_round": None,
            "decklist": 0,
            "deck_id": "x",
            "deck_name": "X",
        }
    ]
    (directory / "standings.json").write_text(json.dumps(standings), encoding="utf-8")
    existing_pairings = [
        {"round": 1, "phase": 1, "p1_id": 1, "p2_id": None, "winner": 0},
        {"round": 2, "phase": 1, "p1_id": 1, "p2_id": None, "winner": 0},
    ]
    (directory / "pairings.json").write_text(json.dumps(existing_pairings), encoding="utf-8")

    class _FakeClient:
        workers = 1

        def __init__(self):
            self.pairing_calls = []

        def tournament(self, labs_id):
            raise AssertionError("metadata is already on disk")

        def standings(self, labs_id):
            raise AssertionError("standings are already on disk")

        def pairings(self, labs_id, round_number):
            self.pairing_calls.append(round_number)
            return [{"p1_id": 1, "p2_id": None, "winner": 0}]

        def decklist(self, labs_id, tp_id):
            return None

    client = _FakeClient()
    event = LabsEvent(labs_id="0099", public_id="900", city="Testville", date="2026-10-03T00:00:00.000Z")

    bundle = collect_event(client, event, directory)

    assert client.pairing_calls == [3, 4]
    assert {row["round"] for row in bundle["pairings"]} == {1, 2, 3, 4}


@pytest.mark.unit
def test_import_bundles_records_labs_provenance_and_resolves_deck_names(tmp_path):
    database = tmp_path / "limitless.db"

    counts = import_bundles(database, [_bundle()], deck_mapping=MAPPING)

    assert counts == {"tournaments": 1, "standings": 2, "pairings": 3, "decklists": 2}
    with sqlite3.connect(database) as conn:
        row = conn.execute("SELECT name, date, players, details_json FROM tournaments WHERE id='580'").fetchone()
        names = dict(conn.execute("SELECT player_id, deck_name FROM standings WHERE tournament_id='580'"))
    assert row[0] == "Regional Recife"
    assert row[1] == "2026-10-03T00:00:00.000Z"
    assert row[2] == 1074
    details = json.loads(row[3])
    assert details["source"] == "Limitless Labs"
    assert details["labs_id"] == "0075"
    assert details["source_url"] == "https://limitlesstcg.com/tournaments/580"
    assert details["labs_metadata"]["players"] == 1074
    assert names == {"100": "Dragapult", "200": "Mystery Box"}


@pytest.mark.unit
def test_import_bundles_maps_pairing_winners_to_stable_player_ids(tmp_path):
    database = tmp_path / "limitless.db"

    import_bundles(database, [_bundle()], deck_mapping=MAPPING)

    with sqlite3.connect(database) as conn:
        winners = [
            tuple(row)
            for row in conn.execute(
                "SELECT round, match, player1, player2, winner FROM pairings WHERE tournament_id='580' ORDER BY round"
            )
        ]
    assert winners == [(1, "", "100", "200", "200"), (2, "", "200", "100", "0"), (3, "", "100", "200", "-1")]


@pytest.mark.unit
def test_import_bundles_refuses_an_event_id_already_present(tmp_path):
    database = tmp_path / "limitless.db"
    import_bundles(database, [_bundle()], deck_mapping=MAPPING)

    with pytest.raises(RuntimeError, match="already exist"):
        import_bundles(database, [_bundle()], deck_mapping=MAPPING)


@pytest.mark.unit
def test_import_bundles_refuses_a_bundle_that_fails_validation(tmp_path):
    database = tmp_path / "limitless.db"

    with pytest.raises(RuntimeError, match="failed validation"):
        import_bundles(database, [_bundle(list_size=59)], deck_mapping=MAPPING)


@pytest.mark.unit
def test_rate_limiter_holds_one_shared_budget_across_threads():
    delay = 0.1
    waits = 12
    limiter = _RateLimiter(delay)
    started = time.monotonic()

    def consume() -> None:
        for _ in range(3):
            limiter.wait()

    with ThreadPoolExecutor(max_workers=4) as pool:
        [future.result() for future in [pool.submit(consume) for _ in range(4)]]
    elapsed = time.monotonic() - started

    assert elapsed >= (waits - 1) * delay * 0.9
