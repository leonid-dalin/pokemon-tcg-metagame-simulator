from collections import Counter

import numpy as np
import pytest

from src.ingestion import best60
from src.ingestion.best60 import improve, parse_list, prerequisites


def _row(event, player, cards, wins=3, losses=2):
    groups = {"pokemon": [], "trainer": [], "energy": []}
    for card, count, group in cards:
        groups[group].append({"name": card, "count": count})
    return {
        "event": event, "player": player, "deck_id": "x", "placing": None, "wins": wins, "losses": losses,
        "decklist": groups, "players": 16, "date": "2026-09-01T00:00:00Z",
    }


def _lists(mix, tech_in=0):
    """Lists built from (Munkidori, Darkness Energy, count) groups, one companion copy each."""
    rows = []
    for munkidori, darkness, count in mix:
        for _ in range(count):
            index = len(rows)
            cards = [
                ("Munkidori", munkidori, "pokemon"),
                ("Darkness Energy", darkness, "energy"),
                ("Companion", 1, "trainer"),
                ("Grass Energy", 60 - munkidori - darkness - 1, "energy"),
            ]
            if index < tech_in:
                cards = [card for card in cards if card[0] != "Grass Energy"]
                cards.append(("Tech", 1, "trainer"))
                cards.append(("Grass Energy", 60 - munkidori - darkness - 1 - 1, "energy"))
            rows.append(parse_list(_row(f"e{index}", f"p{index}", cards)))
    return rows


MIX = ((2, 2, 30), (1, 2, 8), (2, 3, 2))


@pytest.mark.unit
def test_prerequisites_ask_for_a_companion_copy_per_copy():
    needs = prerequisites(_lists(MIX))

    assert needs["Munkidori"]["Darkness Energy"] == 1


@pytest.mark.unit
def test_prerequisites_ask_for_one_copy_when_the_companion_does_not_scale():
    needs = prerequisites(_lists(MIX))

    assert needs["Munkidori"]["Companion"] == 0


@pytest.mark.unit
def test_prerequisites_are_silent_when_the_counts_run_both_ways():
    needs = prerequisites(_lists(((2, 2, 40),)))

    assert needs["Munkidori"]["Darkness Energy"] == 0


@pytest.mark.unit
def test_prerequisites_drop_a_companion_the_lists_do_not_share():
    needs = prerequisites(_lists(MIX, tech_in=30))

    assert needs["Munkidori"]["Darkness Energy"] == 1
    assert "Tech" not in needs["Munkidori"]


@pytest.mark.unit
def test_improve_refuses_a_move_that_starves_a_companion():
    deck = Counter({"Munkidori": 2, "Darkness Energy": 2, "Filler": 56})
    groups = {"Munkidori": "pokemon", "Darkness Energy": "energy", "Filler": "trainer"}
    model = best60.SlotModel(
        slots=[("Darkness Energy", 2), ("Watchtower", 1)],
        beta=np.array([-1.0, 1.0]),
        covariance=np.eye(2) * 1e-4,
        prior_sd=0.1,
    )
    needs = {"Munkidori": {"Darkness Energy": 1}}

    starved, applied, _ = improve(deck, model, groups, needs)

    assert starved["Darkness Energy"] == 2
    assert applied == []

    free, applied_free, _ = improve(deck, model, groups, {})

    assert free["Darkness Energy"] == 1
    assert [swap["add"] for swap in applied_free] == ["Watchtower"]


@pytest.mark.unit
def test_improve_still_allows_a_move_that_keeps_the_companion_counts():
    deck = Counter({"Munkidori": 2, "Darkness Energy": 2, "Filler": 56})
    groups = {"Munkidori": "pokemon", "Darkness Energy": "energy", "Filler": "trainer"}
    model = best60.SlotModel(
        slots=[("Filler", 56), ("Watchtower", 1)],
        beta=np.array([-1.0, 1.0]),
        covariance=np.eye(2) * 1e-4,
        prior_sd=0.1,
    )
    needs = {"Munkidori": {"Darkness Energy": 1}}

    improved, applied, _ = improve(deck, model, groups, needs)

    assert improved["Filler"] == 55
    assert [swap["add"] for swap in applied] == ["Watchtower"]
