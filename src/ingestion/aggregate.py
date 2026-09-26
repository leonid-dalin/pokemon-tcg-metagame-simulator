from __future__ import annotations

from collections import defaultdict
from typing import Any

from src.core.config import BDIF_PAIR_MIN_GAMES
from .store import LimitlessStore


def _record_winner(wins: dict[tuple[str, str], float], key: tuple[str, str], winner: str | None, first: str, second: str) -> None:
    if winner == first:
        wins[key] += 1.0
    elif winner is None:
        wins[key] += 0.5


def _row_exclusion(row: dict[str, Any]) -> str | None:
    if not row.get("player2"):
        return "bye"
    if str(row.get("winner")) == "-1":
        return "double_loss"
    round_number = int(row.get("round") or 0)
    if (row.get("drop1") is not None and round_number > int(row["drop1"])) or (
        row.get("drop2") is not None and round_number > int(row["drop2"])
    ):
        return "after_drop"
    if not row.get("deck1") or not row.get("deck2"):
        return "missing_deck"
    if row["deck1"] == row["deck2"]:
        return "mirror"
    if str(row.get("winner")) not in {"0", str(row.get("player1")), str(row.get("player2"))}:
        return "invalid_winner"
    return None


def _card_inclusion(store: LimitlessStore, archetype: str, event_id: str | None = None) -> dict[str, float]:
    if event_id is None:
        return store.card_inclusion(archetype)
    return store.card_inclusion(archetype, event_id)


def build_artifact(store: LimitlessStore) -> dict[str, Any]:
    totals: dict[tuple[str, str], int] = defaultdict(int)
    wins: dict[tuple[str, str], float] = defaultdict(float)
    coverage: dict[str, dict[str, Any]] = {}
    rows = store.aggregate_rows() if hasattr(store, "aggregate_rows") else (
        {
            "event_id": "combined",
            "round": 0,
            "deck1": deck1,
            "deck2": deck2,
            "winner": winner,
            "player1": player1,
            "player2": player2,
            "drop1": None,
            "drop2": None,
        }
        for deck1, deck2, winner, player1, player2 in store.matchup_rows()
    )
    for row in rows:
        event_id = str(row.get("event_id") or "combined")
        event_coverage = coverage.setdefault(event_id, {"included_matches": 0, "excluded_matches": 0, "exclusions": defaultdict(int)})
        exclusion = _row_exclusion(row)
        if exclusion:
            event_coverage["excluded_matches"] += 1
            event_coverage["exclusions"][exclusion] += 1
            continue
        event_coverage["included_matches"] += 1
        deck1, deck2 = str(row["deck1"]), str(row["deck2"])
        first, second = sorted((deck1, deck2))
        key = (first, second)
        totals[key] += 1
        winner_deck = deck1 if str(row["winner"]) == str(row["player1"]) else first if str(row["winner"]) == "0" and first == deck1 else second if str(row["winner"]) == "0" else deck2
        _record_winner(wins, key, None if str(row["winner"]) == "0" else winner_deck, first, second)
    archetypes = sorted({deck for pair in totals for deck in pair})
    matrix: dict[str, dict[str, dict[str, float | int]]] = {}
    for deck in archetypes:
        matrix[deck] = {}
        for opponent in archetypes:
            if deck == opponent:
                matrix[deck][opponent] = {"win_rate": 0.5, "match_count": 0}
                continue
            first, second = sorted((deck, opponent))
            count = totals.get((first, second), 0)
            first_rate = wins[(first, second)] / count if count else 0.5
            rate = first_rate if deck == first else 1.0 - first_rate
            matrix[deck][opponent] = {"win_rate": rate, "match_count": count}
    combined = {"included_matches": 0, "excluded_matches": 0, "exclusions": defaultdict(int)}
    for event in coverage.values():
        combined["included_matches"] += event["included_matches"]
        combined["excluded_matches"] += event["excluded_matches"]
        for reason, count in event["exclusions"].items():
            combined["exclusions"][reason] += count
    normalised_coverage = {
        "combined": {
            "included_matches": combined["included_matches"],
            "excluded_matches": combined["excluded_matches"],
            "exclusions": dict(sorted(combined["exclusions"].items())),
        },
        "events": {
            event_id: {
                "included_matches": event["included_matches"],
                "excluded_matches": event["excluded_matches"],
                "exclusions": dict(sorted(event["exclusions"].items())),
            }
            for event_id, event in sorted(coverage.items())
        },
    }
    event_cards = {
        event_id: {
            deck: _card_inclusion(store, deck, event_id)
            for deck in archetypes
        }
        for event_id in sorted(coverage)
    } if hasattr(store, "aggregate_rows") else {}
    return {
        "archetypes": archetypes,
        "win_rate_matrix": matrix,
        "card_inclusion": {deck: _card_inclusion(store, deck) for deck in archetypes},
        "card_inclusion_by_event": event_cards,
        "coverage": normalised_coverage,
        "min_pair_matches": BDIF_PAIR_MIN_GAMES,
    }
