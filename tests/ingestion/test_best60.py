from collections import Counter

import numpy as np
import pytest

from src.ingestion import best60
from src.ingestion.best60 import (
    ArchetypeList,
    build_best60,
    consensus_sixty,
    event_folds,
    fit_slot_model,
    improve,
    parse_list,
    player_strength,
    prerequisites,
    trends,
    card_stats,
)


def _row(event, player, cards, wins=3, losses=2, placing=None, players=16, date="2026-09-01T00:00:00Z", deck_id="x"):
    groups = {"pokemon": [], "trainer": [], "energy": []}
    for card, count, group in cards:
        groups[group].append({"name": card, "count": count})
    return {"event": event, "player": player, "deck_id": deck_id, "placing": placing, "wins": wins, "losses": losses,
            "decklist": groups, "players": players, "date": date}


def _base(tech_a=0, tech_b=0, line=True):
    filler = 60 - 8 - 4 - tech_a - tech_b
    cards = [("Mon", 4, "pokemon"), ("Mon ex", 4 if line else 0, "pokemon"), ("Ultra Ball", 4, "trainer")]
    if not line:
        filler += 4
    cards = [card for card in cards if card[1]]
    if tech_a:
        cards.append(("Tech A", tech_a, "trainer"))
    if tech_b:
        cards.append(("Tech B", tech_b, "trainer"))
    cards.append(("Grass Energy", filler, "energy"))
    return cards


def _synthetic(effect_a=0.0, effect_b=0.0, strong_play_b=False, lists=600, seed=7, other_games=40):
    rng = np.random.default_rng(seed)
    rows, records = [], {}
    for index in range(lists):
        event, player = f"e{index % 60}", f"p{index}"
        skill = rng.normal(0.0, 1.0)
        tech_a = int(rng.integers(0, 3))
        tech_b = int(skill > 0.5) if strong_play_b else int(rng.integers(0, 2))
        logit = 0.6 * skill + effect_a * (tech_a >= 2) + effect_b * tech_b
        games = 9
        wins = int(rng.binomial(games, 1.0 / (1.0 + np.exp(-logit))))
        rows.append(_row(event, player, _base(tech_a, tech_b), wins=wins, losses=games - wins))
        records[(event, player)] = (wins, games - wins)
        other_wins = int(rng.binomial(other_games, 1.0 / (1.0 + np.exp(-0.6 * skill))))
        records[(f"other{index}", player)] = (other_wins, other_games - other_wins)
    return rows, records


@pytest.mark.unit
def test_parse_list_rejects_illegal_sixties_and_sums_printings():
    legal = _row("e", "p", [("Mon", 2, "pokemon"), ("Mon", 2, "pokemon"), ("Grass Energy", 56, "energy")])
    assert parse_list(legal).counts == {"Mon": 4, "Grass Energy": 56}
    assert parse_list(_row("e", "p", [("Mon", 5, "pokemon"), ("Grass Energy", 55, "energy")])) is None
    assert parse_list(_row("e", "p", [("Mon", 4, "pokemon"), ("Grass Energy", 55, "energy")])) is None
    assert parse_list(_row("e", "p", [("Prime Catcher", 1, "trainer"), ("Master Ball", 1, "trainer"), ("Grass Energy", 58, "energy")])) is None
    assert parse_list({**legal, "decklist": "not json"}) is None


@pytest.mark.unit
def test_consensus_is_sixty_nested_slots_with_one_ace_spec():
    rows = [
        _row("e1", "p1", [("Mon", 4, "pokemon"), ("Prime Catcher", 1, "trainer"), ("Grass Energy", 55, "energy")]),
        _row("e1", "p2", [("Mon", 4, "pokemon"), ("Master Ball", 1, "trainer"), ("Grass Energy", 55, "energy")]),
        _row("e1", "p3", [("Mon", 3, "pokemon"), ("Master Ball", 1, "trainer"), ("Grass Energy", 56, "energy")]),
    ]
    deck = consensus_sixty([parse_list(row) for row in rows])
    assert deck == Counter({"Grass Energy": 55, "Mon": 4, "Master Ball": 1})


@pytest.mark.unit
def test_consensus_skips_a_second_ace_spec_even_when_it_outranks_the_next_slot():
    common = [("Mon", 4, "pokemon"), ("Grass Energy", 54, "energy")]
    rows = [
        _row("e1", f"a{index}", common + [("Amulet of Hope", 1, "trainer"), ("Zeta A", 1, "trainer")]) for index in range(3)
    ] + [
        _row("e1", f"b{index}", common + [("Awakening Drum", 1, "trainer"), ("Zeta B", 1, "trainer")]) for index in range(3)
    ]
    deck = consensus_sixty([parse_list(row) for row in rows])
    assert deck == Counter({"Grass Energy": 54, "Mon": 4, "Amulet of Hope": 1, "Zeta A": 1})


@pytest.mark.unit
def test_player_strength_leaves_the_scored_event_out():
    strength = player_strength({("e1", "p"): (9, 0), ("e2", "p"): (0, 9)})
    assert strength[("e1", "p")] == pytest.approx(np.log(10 / 19))
    assert strength[("e2", "p")] == pytest.approx(np.log(19 / 10))


@pytest.mark.unit
def test_held_out_folds_never_split_an_event():
    lists = [parse_list(_row(f"e{index % 7}", f"p{index}", _base())) for index in range(70)]
    folds = event_folds(lists)
    by_event = {}
    for entry, fold in zip(lists, folds):
        by_event.setdefault(entry.event, set()).add(int(fold))
    assert all(len(found) == 1 for found in by_event.values())
    assert set(folds.tolist()) == set(range(best60.FOLDS))


@pytest.mark.unit
def test_a_planted_second_copy_effect_is_applied_and_holds_up_on_held_out_events():
    rows, records = _synthetic(effect_a=0.8)
    result = build_best60("X", rows, player_strength(records))

    assert result["status"] == "complete"
    assert result["model"]["held_out_gain"] > 0
    assert {(swap["add"], swap["add_copy"]) for swap in result["swaps"]} >= {("Tech A", 2)}
    assert {row["card"]: row["copies"] for row in result["cards"]}["Tech A"] == 2
    assert result["total_copies"] == 60


@pytest.mark.unit
def test_without_any_card_effect_the_consensus_is_kept():
    rows, records = _synthetic()
    result = build_best60("X", rows, player_strength(records))

    consensus = consensus_sixty([parse_list(row) for row in rows])
    assert result["status"].startswith("consensus")
    assert {row["card"]: row["copies"] for row in result["cards"]} == dict(consensus)
    assert result["swaps"] == []


@pytest.mark.unit
def test_player_strength_absorbs_a_card_that_only_strong_players_run():
    rows, records = _synthetic(strong_play_b=True, other_games=400)
    lists = [parse_list(row) for row in rows]
    controlled = fit_slot_model(lists, player_strength(records), 0.5)
    ignored = fit_slot_model(lists, {}, 0.5)
    index = controlled.slots.index(("Tech B", 1))
    assert abs(controlled.beta[index]) < 0.5 * abs(ignored.beta[ignored.slots.index(("Tech B", 1))])


@pytest.mark.unit
def test_swaps_never_add_an_evolution_without_its_basic_or_drop_the_last_pokemon():
    lists = [parse_list(_row("e", f"p{i}", _base(line=i % 2 == 0))) for i in range(10)]
    deck = Counter({"Mon": 4, "Ultra Ball": 4, "Grass Energy": 52})
    groups = {"Mon": "pokemon", "Mon ex": "pokemon", "Ultra Ball": "trainer", "Grass Energy": "energy"}
    model = best60.SlotModel(
        slots=[("Mon", 1), ("Mon ex", 1), ("Ultra Ball", 4)],
        beta=np.array([-1.0, 1.0, -1.0]),
        covariance=np.eye(3) * 1e-4,
        prior_sd=0.1,
    )
    needs = prerequisites(lists)
    assert "Mon" in needs["Mon ex"]

    improved, applied, _ = improve(deck, model, groups, needs)

    assert improved["Mon"] == 4
    assert [(swap["remove"], swap["add"]) for swap in applied] == [("Ultra Ball", "Mon ex")]
    improved, applied, _ = improve(Counter({"Ultra Ball": 4, "Grass Energy": 56}), model, groups, {"Mon ex": {"Mon"}})
    assert all(swap["add"] != "Mon ex" for swap in applied)

    single = best60.SlotModel(slots=[("Mon", 1), ("Ultra Ball", 4)], beta=np.array([-1.0, 1.0]), covariance=np.eye(2) * 1e-4, prior_sd=0.1)
    improved, applied, _ = improve(Counter({"Mon": 1, "Ultra Ball": 3, "Grass Energy": 56}), single, groups, {})
    assert improved["Mon"] == 1
    assert applied == []


@pytest.mark.unit
def test_too_few_lists_return_the_consensus_with_a_status():
    rows = [_row("e", f"p{i}", _base()) for i in range(5)]
    result = build_best60("X", rows, {})
    assert result["status"] == "consensus only: too few lists to score cards"
    assert result["total_copies"] == 60
    assert result["deck_ids"] == ["x"]


@pytest.mark.unit
def test_no_legal_lists_is_a_status_not_an_error():
    result = build_best60("X", [_row("e", "p", [("Mon", 4, "pokemon")])], {})
    assert result["status"] == "no legal lists"
    assert result["cards"] == []


@pytest.mark.unit
def test_finish_tiers_count_only_events_large_enough_to_earn_them():
    rows = [
        _row("big", "p1", _base(tech_a=1), placing=1, players=16),
        _row("big", "p2", _base(), placing=9, players=16),
        _row("small", "p3", _base(tech_a=1), placing=1, players=6),
    ]
    stats = card_stats([parse_list(row) for row in rows])
    assert stats["Tech A"]["top25"] == {"with_card": 1.0, "archetype": 0.5, "lists": 1}
    assert stats["Tech A"]["top50"]["lists"] == 1
    assert stats["Tech A"]["winner"]["lists"] == 2


@pytest.mark.unit
def test_a_rising_card_is_a_breakthrough_only_when_the_model_does_not_rate_it_negative():
    rows = []
    for index in range(40):
        rows.append(_row(f"early{index}", f"a{index}", _base(), placing=8, players=16, date="2026-07-01T00:00:00Z"))
        rows.append(_row(f"late{index}", f"b{index}", _base(tech_a=1), placing=1, players=16, date="2026-09-01T00:00:00Z"))
    lists = [parse_list(row) for row in rows]
    stats = card_stats(lists)

    plain = trends(lists, stats)
    assert [row["card"] for row in plain["rising"]] == ["Tech A"]
    assert plain["breakthrough"] == ["Tech A"]

    negative = best60.SlotModel(slots=[("Tech A", 1)], beta=np.array([-0.2]), covariance=np.eye(1), prior_sd=0.1)
    assert trends(lists, stats, negative)["breakthrough"] == []


@pytest.mark.unit
def test_an_unlikely_swap_is_listed_as_leaning_not_applied():
    deck = Counter({"Ultra Ball": 4, "Grass Energy": 56})
    model = best60.SlotModel(
        slots=[("Ultra Ball", 4), ("Tech A", 1)],
        beta=np.array([0.0, 0.01]),
        covariance=np.eye(2) * 0.01,
        prior_sd=0.1,
    )

    improved, applied, leaning = improve(deck, model, {"Ultra Ball": "trainer", "Tech A": "trainer"}, {})

    assert applied == []
    assert improved == deck
    assert [(swap["remove"], swap["add"]) for swap in leaning] == [("Ultra Ball", "Tech A")]
    assert 0.5 <= leaning[0]["probability"] < 0.7


@pytest.mark.unit
def test_swaps_that_do_not_hold_up_on_held_out_events_are_proposed_not_applied(monkeypatch):
    rows, records = _synthetic(effect_a=0.8)
    monkeypatch.setattr(best60, "held_out_gain", lambda *args: -0.01)

    result = build_best60("X", rows, player_strength(records))

    assert result["status"] == "consensus kept: swaps did not hold up on held-out events"
    assert result["swaps"] == []
    assert {(swap["add"], swap["add_copy"]) for swap in result["proposed_swaps"]} >= {("Tech A", 2)}
    assert {row["card"]: row["copies"] for row in result["cards"]} == dict(consensus_sixty([parse_list(row) for row in rows]))
