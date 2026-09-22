from __future__ import annotations

from collections import defaultdict
from typing import Any

from src.core.config import BDIF_PAIR_MIN_GAMES
from .store import LimitlessStore


def _record_winner(wins: dict[tuple[str, str], float], totals: dict[tuple[str, str], int], key: tuple[str, str], winner: str | None, first: str, second: str) -> None:
    if winner == first:
        wins[key] += 1.0
    elif winner == second:
        return
    else:
        totals[key] -= 1


def build_artifact(store: LimitlessStore) -> dict[str, Any]:
    totals: dict[tuple[str, str], int] = defaultdict(int)
    wins: dict[tuple[str, str], float] = defaultdict(float)
    for deck1, deck2, winner, player1, player2 in store.matchup_rows():
        if not deck1 or not deck2 or deck1 == deck2:
            continue
        deck1, deck2 = str(deck1), str(deck2)
        first, second = sorted((deck1, deck2))
        key = (first, second)
        totals[key] += 1
        winner_deck = deck1 if str(winner) == str(player1) else deck2 if str(winner) == str(player2) else None
        _record_winner(wins, totals, key, winner_deck, first, second)
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
    return {
        "archetypes": archetypes,
        "win_rate_matrix": matrix,
        "card_inclusion": {deck: store.card_inclusion(deck) for deck in archetypes},
        "min_pair_matches": BDIF_PAIR_MIN_GAMES,
    }
