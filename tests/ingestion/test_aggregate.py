from __future__ import annotations

from src.ingestion.store import LimitlessStore
from src.ingestion.aggregate import build_artifact


class Store:
    def aggregate_rows(self):
        return [
            {
                "event_id": "event-a",
                "round": 1,
                "deck1": "Crustle",
                "deck2": "N's Zoroark",
                "winner": "0",
                "player1": "p1",
                "player2": "p2",
                "drop1": None,
                "drop2": None,
            },
            {
                "event_id": "event-a",
                "round": 4,
                "deck1": "Crustle",
                "deck2": "N's Zoroark",
                "winner": "p1",
                "player1": "p1",
                "player2": "p2",
                "drop1": 3,
                "drop2": None,
            },
        ]

    def card_inclusion(self, archetype, event_id=None):
        return {"Core": 1.0} if event_id is None else {"Core": 1.0}


def test_build_artifact_counts_ties_and_reports_post_drop_exclusions():
    artifact = build_artifact(Store())

    assert artifact["win_rate_matrix"]["Crustle"]["N's Zoroark"] == {
        "win_rate": 0.5,
        "match_count": 1,
    }
    assert artifact["coverage"] == {
        "combined": {
            "included_matches": 1,
            "excluded_matches": 1,
            "exclusions": {"after_drop": 1},
        },
        "events": {
            "event-a": {
                "included_matches": 1,
                "excluded_matches": 1,
                "exclusions": {"after_drop": 1},
            },
        },
    }
    assert artifact["card_inclusion_by_event"] == {
        "event-a": {
            "Crustle": {"Core": 1.0},
            "N's Zoroark": {"Core": 1.0},
        },
    }


def test_sqlite_aggregation_preserves_exclusions_and_event_card_rates(tmp_path):
    store = LimitlessStore(tmp_path / "limitless.db", deck_mapping={"a": "A", "b": "B"})
    store.upsert_standings("event-a", [
        {"player": "a1", "deck": {"id": "a"}, "decklist": {"pokemon": [{"name": "Core"}]}},
        {"player": "a2", "deck": {"id": "b"}, "decklist": {"pokemon": [{"name": "Core"}]}},
    ])
    store.upsert_standings("event-b", [
        {"player": "b1", "deck": {"id": "a"}, "decklist": {"pokemon": [{"name": "Side"}]}},
        {"player": "b2", "deck": {"id": "b"}, "decklist": {"pokemon": [{"name": "Side"}]}},
    ])
    store.upsert_pairings("event-a", [
        {"round": 1, "player1": "a1", "player2": "a2", "winner": "0"},
        {"round": 2, "player1": "a1", "player2": "", "winner": "a1"},
        {"round": 3, "player1": "a1", "player2": "a2", "winner": "-1"},
    ])
    store.upsert_pairings("event-b", [
        {"round": 1, "player1": "b1", "player2": "b2", "winner": "b1"},
    ])

    artifact = build_artifact(store)

    assert artifact["coverage"]["combined"] == {
        "included_matches": 2,
        "excluded_matches": 2,
        "exclusions": {"bye": 1, "double_loss": 1},
    }
    assert artifact["coverage"]["events"]["event-a"]["exclusions"] == {
        "bye": 1,
        "double_loss": 1,
    }
    assert artifact["win_rate_matrix"]["A"]["B"] == {
        "win_rate": 0.75,
        "match_count": 2,
    }
    assert artifact["card_inclusion_by_event"] == {
        "event-a": {"A": {"Core": 1.0}, "B": {"Core": 1.0}},
        "event-b": {"A": {"Side": 1.0}, "B": {"Side": 1.0}},
    }