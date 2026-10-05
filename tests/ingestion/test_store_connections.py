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
