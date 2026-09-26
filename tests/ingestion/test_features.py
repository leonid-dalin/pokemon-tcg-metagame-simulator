from __future__ import annotations

from datetime import datetime, timedelta, timezone

from src.ingestion.features import inclusion_rates
from src.ingestion.store import LimitlessStore


def _add_event(store, event_id: str, date: str, standings: list[dict]) -> None:
    store.upsert_tournament(
        {
            "id": event_id,
            "game": "PTCG",
            "format": "STANDARD",
            "name": event_id,
            "date": date,
            "players": len(standings),
        },
        {"decklists": True},
    )
    store.upsert_standings(event_id, standings)


def _standing(player: str, cards: list[dict], deck: str = "test-deck") -> dict:
    return {
        "player": player,
        "deck": {"id": deck, "name": "Test Archetype"},
        "decklist": {"pokemon": [], "trainer": cards, "energy": []},
    }


def test_inclusion_rates_counts_decks_containing_card_once(tmp_path):
    store = LimitlessStore(tmp_path / "limitless.db")
    date = datetime.now(timezone.utc).isoformat()
    standings = [
        _standing(
            f"player-{index}",
            (
                [{"name": "Misty Energy", "count": 2}, {"name": "Misty Energy", "count": 2}]
                if index < 4
                else [{"name": "Basic Energy", "count": 8}]
            ),
        )
        for index in range(10)
    ]
    _add_event(store, "event", date, standings)

    rates = inclusion_rates(store, "Test Archetype")

    assert rates["Misty Energy"] == 0.4
    assert rates["Basic Energy"] == 0.6
    assert store.card_inclusion("Test Archetype")["Misty Energy"] == 0.4


def test_inclusion_rates_applies_window_to_tournament_date(tmp_path):
    store = LimitlessStore(tmp_path / "limitless.db")
    now = datetime.now(timezone.utc)
    _add_event(
        store,
        "recent",
        (now - timedelta(days=2)).isoformat(),
        [_standing("recent-player", [{"name": "Recent Card", "count": 1}])],
    )
    _add_event(
        store,
        "old",
        (now - timedelta(days=20)).isoformat(),
        [_standing("old-player", [{"name": "Old Card", "count": 1}])],
    )

    assert inclusion_rates(store, "Test Archetype", window_days=7) == {"Recent Card": 1.0}


def test_inclusion_rates_returns_empty_for_empty_window(tmp_path):
    store = LimitlessStore(tmp_path / "limitless.db")
    old_date = (datetime.now(timezone.utc) - timedelta(days=20)).isoformat()
    _add_event(
        store,
        "old",
        old_date,
        [_standing("old-player", [{"name": "Old Card", "count": 1}])],
    )

    assert inclusion_rates(store, "Test Archetype", window_days=7) == {}


def test_inclusion_rates_excludes_standings_without_decklists(tmp_path):
    store = LimitlessStore(tmp_path / "limitless.db")
    date = datetime.now(timezone.utc).isoformat()
    _add_event(
        store,
        "event",
        date,
        [
            _standing("with-list", [{"name": "Misty Energy", "count": 1}]),
            {"player": "without-list", "deck": {"id": "test-deck", "name": "Test Archetype"}, "decklist": None},
        ],
    )

    assert inclusion_rates(store, "Test Archetype", window_days=7) == {"Misty Energy": 1.0}