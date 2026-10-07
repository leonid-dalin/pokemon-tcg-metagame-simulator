import sqlite3

import pytest

from src.ingestion.aggregate import build_artifact
from src.ingestion.store import LimitlessStore


def _is_open(connection: sqlite3.Connection) -> bool:
    try:
        connection.execute("SELECT 1")
    except sqlite3.ProgrammingError:
        return False
    return True


@pytest.mark.unit
def test_store_operations_close_every_connection(monkeypatch, tmp_path):
    opened = []
    real_connect = sqlite3.connect

    def tracking_connect(*args, **kwargs):
        connection = real_connect(*args, **kwargs)
        opened.append(connection)
        return connection

    monkeypatch.setattr(sqlite3, "connect", tracking_connect)
    store = LimitlessStore(tmp_path / "limitless.db", deck_mapping={"a": "A", "b": "B"})
    decklist = {"pokemon": [{"name": "Mon", "count": 4}], "energy": [{"name": "Grass Energy", "count": 56}]}
    store.upsert_tournament({"id": "e", "date": "2026-09-01"}, {})
    store.upsert_standings("e", [
        {"player": "p1", "deck": {"id": "a"}, "decklist": decklist},
        {"player": "p2", "deck": {"id": "b"}, "decklist": decklist},
    ])
    store.upsert_pairings("e", [{"round": 1, "player1": "p1", "player2": "p2", "winner": "p1"}])
    store.backfill_deck_names({"a": "A"})
    store.player_observations()
    store.deck_weights()
    store.unmapped_deck_ids()
    store.archetype_lists("A")
    store.player_records()
    store.existing_tournament_ids()
    build_artifact(store)

    assert opened
    assert [index for index, connection in enumerate(opened) if _is_open(connection)] == []


def test_store_reads_games_with_opponent_decks_and_records_before_a_date(tmp_path):
    from src.ingestion.best60_games import Game

    store = LimitlessStore(tmp_path / "limitless.db", deck_mapping={"a": "A", "b": "B"})
    decklist = {"pokemon": [{"name": "Mon", "count": 4}], "energy": [{"name": "Grass Energy", "count": 56}]}
    for event, date in (("e1", "2026-09-01"), ("e2", "2026-09-30")):
        store.upsert_tournament({"id": event, "date": date}, {})
        store.upsert_standings(event, [
            {"player": "p1", "deck": {"id": "a"}, "decklist": decklist, "record": {"wins": 1, "losses": 0}},
            {"player": "p2", "deck": {"id": "b"}, "decklist": decklist, "record": {"wins": 0, "losses": 1}},
        ])
    store.upsert_pairings("e1", [{"round": 1, "player1": "p1", "player2": "p2", "winner": "p1"}])
    store.upsert_pairings("e2", [{"round": 1, "player1": "p2", "player2": "p1", "winner": "0"}])

    assert store.archetype_games("A") == [Game("e1", "p1", "p2", "B", 1, "2026-09-01")]
    assert store.archetype_games("B") == [Game("e1", "p2", "p1", "A", 0, "2026-09-01")]
    assert set(store.player_records(before="2026-09-15")) == {("e1", "p1"), ("e1", "p2")}
    assert set(store.player_records()) == {("e1", "p1"), ("e1", "p2"), ("e2", "p1"), ("e2", "p2")}
