from __future__ import annotations

from collections import defaultdict
from typing import Any

from src.core.config import BDIF_PAIR_MIN_GAMES
from .store import LimitlessStore


def build_artifact(store: LimitlessStore, min_matches: int = BDIF_PAIR_MIN_GAMES) -> dict[str, Any]:
    totals: dict[tuple[str, str], int] = defaultdict(int)
    wins: dict[tuple[str, str], float] = defaultdict(float)
    for deck1, deck2, winner, player1, player2 in store.matchup_rows():
        if not deck1 or not deck2 or deck1 == deck2:
            continue
        key = (str(deck1), str(deck2))
        totals[key] += 1
        if str(winner) == str(player1):
            wins[key] += 1.0
        elif str(winner) == str(player2):
            wins[key] += 0.0
        else:
            totals[key] -= 1
    archetypes = sorted({deck for pair in totals for deck in pair})
    matrix: dict[str, dict[str, dict[str, float | int]]] = {}
    for deck in archetypes:
        matrix[deck] = {}
        for opponent in archetypes:
            if deck == opponent:
                matrix[deck][opponent] = {"win_rate": 0.5, "match_count": 0}
                continue
            count = totals.get((deck, opponent), 0)
            rate = wins[(deck, opponent)] / count if count else 0.5
            matrix[deck][opponent] = {"win_rate": rate, "match_count": count}
    return {
        "archetypes": archetypes,
        "win_rate_matrix": matrix,
        "card_inclusion": {deck: store.card_inclusion(deck) for deck in archetypes},
        "min_pair_matches": min_matches,
    }
