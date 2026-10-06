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
from src.ingestion.model import validate_recommendation



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
    result = build_best60("X", rows, player_strength(records), records)

    assert result["status"] == "complete"
    assert result["model"]["held_out_gain"] - result["model"]["held_out_gain_se"] > 0
    assert {(swap["add"], swap["add_copy"]) for swap in result["swaps"]} >= {("Tech A", 2)}
    assert {row["card"]: row["copies"] for row in result["cards"]}["Tech A"] == 2
    assert result["total_copies"] == 60


@pytest.mark.unit
def test_a_weaker_planted_effect_is_still_applied():
    rows, records = _synthetic(effect_a=0.4)
    result = build_best60("X", rows, player_strength(records), records)

    assert result["status"] == "complete"
    assert {(swap["add"], swap["add_copy"]) for swap in result["swaps"]} >= {("Tech A", 2)}


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
def test_one_event_lists_keep_consensus_without_cross_validation():
    rows = [_row("e", f"p{i}", _base()) for i in range(100)]
    result = build_best60("X", rows, {})
    assert result["status"] == "consensus only: card counts did not predict held-out results"
    assert result["total_copies"] == 60


@pytest.mark.unit
def test_fold_strength_uses_training_records_only():
    records = {
        ("train", "p"): (9, 0),
        ("held-out", "p"): (0, 9),
    }
    strength = best60._fold_strength(records, {"train"}, {"held-out"})
    assert strength[("held-out", "p")] == pytest.approx(np.log(19 / 10))


@pytest.mark.unit
def test_held_out_gain_chooses_on_training_events_and_prices_on_held_out_events(monkeypatch):
    rows, records = _synthetic(effect_a=0.8, lists=600)
    lists = [parse_list(row) for row in rows]
    calls = []
    original = best60.fit_slot_model

    def capture_model(fitted, strength, prior_sd):
        calls.append({entry.event for entry in fitted})
        return original(fitted, strength, prior_sd)

    monkeypatch.setattr(best60, "fit_slot_model", capture_model)
    mean, se = best60.held_out_gain(lists, player_strength(records), 0.5, consensus_sixty(lists), {"Tech A": "trainer"}, records)

    folds = best60.event_folds(lists)
    expected = []
    for index in range(best60.FOLDS):
        expected.append({entry.event for entry, fold in zip(lists, folds) if fold != index})
        expected.append({entry.event for entry, fold in zip(lists, folds) if fold == index})
    assert calls == expected
    assert mean > 0
    assert se >= 0


@pytest.mark.unit
def test_no_legal_lists_is_a_status_not_an_error():
    result = build_best60("X", [_row("e", "p", [("Mon", 4, "pokemon")])], {})
    assert result["status"] == "no legal lists"
    assert result["cards"] == []


@pytest.mark.parametrize("copies", range(0, 61, 4))
@pytest.mark.parametrize("extra_ace_specs", [0, 1, 2])
def test_best60_legality_matrix_returns_legal_sixty_or_status(copies, extra_ace_specs):
    cards = [("Mon", 4, "pokemon"), ("Grass Energy", 56 - copies, "energy")]
    if copies:
        cards.append(("Tech A", copies, "trainer"))
    if extra_ace_specs:
        ace_names = ("Prime Catcher", "Master Ball")
        cards.extend((ace_names[index], 1, "trainer") for index in range(extra_ace_specs))
    result = build_best60("X", [_row("e", "p", cards)], {})
    assert result["status"] == "no legal lists" or result["total_copies"] == 60
    if result["status"] != "no legal lists":
        validate_recommendation(result["cards"])


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
    monkeypatch.setattr(best60, "held_out_gain", lambda *args: (-0.01, 0.0))

    result = build_best60("X", rows, player_strength(records))

    assert result["status"] == "consensus kept: swaps did not hold up on held-out events"
    assert result["swaps"] == []
    assert {(swap["add"], swap["add_copy"]) for swap in result["proposed_swaps"]} >= {("Tech A", 2)}
    assert {row["card"]: row["copies"] for row in result["cards"]} == dict(consensus_sixty([parse_list(row) for row in rows]))


@pytest.mark.unit
@pytest.mark.parametrize(("gain", "se", "applied"), [(0.05, 0.01, True), (0.01, 0.02, False)])
def test_swaps_need_the_held_out_gain_to_clear_one_standard_error(monkeypatch, gain, se, applied):
    rows, records = _synthetic(effect_a=0.8)
    monkeypatch.setattr(best60, "held_out_gain", lambda *args: (gain, se))

    result = build_best60("X", rows, player_strength(records), records)

    assert bool(result["swaps"]) is applied
    assert result["status"] == ("complete" if applied else "consensus kept: swaps did not hold up on held-out events")
    assert result["model"]["held_out_gain_se"] == se


def _tome_synthetic(effect=-0.8, lists=600, seed=11):
    """Lists that run 'Tome' at 0 or 4 and almost never in between; running it costs `effect` log-odds."""
    rng = np.random.default_rng(seed)
    rows, records = [], {}
    for index in range(lists):
        event, player = f"e{index % 61}", f"p{index}"
        skill = rng.normal(0.0, 1.0)
        tome = 4 if rng.random() < 0.6 else 0
        tech = int(rng.integers(0, 3))
        cards = [("Mon", 4, "pokemon"), ("Ultra Ball", 4, "trainer")]
        if tome:
            cards.append(("Tome", tome, "trainer"))
        if tech:
            cards.append(("Tech C", tech, "trainer"))
        cards.append(("Grass Energy", 60 - 8 - tome - tech, "energy"))
        logit = 0.6 * skill + effect * (tome > 0)
        wins = int(rng.binomial(9, 1.0 / (1.0 + np.exp(-logit))))
        rows.append(_row(event, player, cards, wins=wins, losses=9 - wins))
        records[(event, player)] = (wins, 9 - wins)
        other = int(rng.binomial(40, 1.0 / (1.0 + np.exp(-0.6 * skill))))
        records[(f"other{index}", player)] = (other, 40 - other)
    return rows, records


@pytest.mark.unit
def test_count_levels_keep_only_counts_enough_lists_play():
    rows = (
        [_row("e", f"a{i}", [("Mon", 4, "pokemon"), ("Tome", 4, "trainer"), ("Grass Energy", 52, "energy")]) for i in range(60)]
        + [_row("e", f"b{i}", [("Mon", 4, "pokemon"), ("Grass Energy", 56, "energy")]) for i in range(40)]
        + [_row("e", f"c{i}", [("Mon", 4, "pokemon"), ("Tome", 2, "trainer"), ("Grass Energy", 54, "energy")]) for i in range(5)]
    )
    lists = [parse_list(row) for row in rows]
    levels = best60.count_levels(lists, consensus_sixty(lists))
    assert levels["Tome"] == {0, 4}
    assert levels["Mon"] == {4}


@pytest.mark.unit
def test_a_card_played_at_zero_or_four_leaves_in_one_move():
    rows, records = _tome_synthetic()
    result = build_best60("X", rows, player_strength(records), records)

    assert result["status"] == "complete"
    tome_moves = [move for move in result["swaps"] if any(card == "Tome" for card, _ in move["removed"])]
    assert len(tome_moves) == 1
    assert sorted(copy for card, copy in tome_moves[0]["removed"] if card == "Tome") == [1, 2, 3, 4]
    assert "Tome" not in {row["card"] for row in result["cards"]}
    assert result["joint_probability"] > 0.9


@pytest.mark.unit
def test_moves_never_take_back_an_earlier_move():
    for rows, records in (_tome_synthetic(), _synthetic(effect_a=0.8)):
        result = build_best60("X", rows, player_strength(records), records)
        added = {card for move in result["swaps"] for card, _ in move["added"]}
        removed = {card for move in result["swaps"] for card, _ in move["removed"]}
        assert not added & removed


@pytest.mark.unit
def test_a_level_move_that_would_undo_an_earlier_move_is_not_taken():
    # Adding all 4 "Hammer" first frees the 4 weak "Bad" copies (gain 6). Undoing it and filling with
    # 4 "Good" would gain 6 again, but would take back the first move.
    slots = [("Bad", copy) for copy in range(1, 5)] + [("Hammer", copy) for copy in range(1, 5)] + [("Good", copy) for copy in range(1, 5)]
    beta = np.array([-1.0] * 4 + [0.5] * 4 + [2.0] * 4)
    model = best60.SlotModel(slots=slots, beta=beta, covariance=np.eye(len(slots)) * 1e-4, prior_sd=0.1)
    levels = {"Bad": {0, 1, 2, 3, 4}, "Hammer": {0, 4}, "Good": {0, 1, 2, 3, 4}}
    groups = {card: "trainer" for card in ("Bad", "Hammer", "Good")}

    improved, applied, _ = improve(Counter({"Bad": 4, "Grass Energy": 56}), model, groups, {}, levels)

    assert [sorted({card for card, _ in move["added"]}) for move in applied] == [["Hammer"]]
    assert improved["Hammer"] == 4


@pytest.mark.unit
def test_observed_mode_never_leaves_lists_people_played():
    rows, records = _tome_synthetic()
    observed = build_best60("X", rows, player_strength(records), records, "observed")

    assert observed["list_mode"] == "observed"
    assert observed["support"]["recommended"][best60.BDIF_BEST60_SUPPORT_CHANGES] >= best60.BDIF_BEST60_SUPPORT_LISTS
    deck = Counter({"Ultra Ball": 4, "Grass Energy": 56})
    model = best60.SlotModel(slots=[("Ultra Ball", 4), ("Tech", 1)], beta=np.array([-1.0, 1.0]), covariance=np.eye(2) * 1e-4, prior_sd=0.1)
    _, applied, _ = improve(deck, model, {"Ultra Ball": "trainer", "Tech": "trainer"}, {}, None, lambda candidate: False)
    assert applied == []


@pytest.mark.unit
def test_observed_mode_drops_moves_no_played_lists_support(monkeypatch):
    rows, records = _tome_synthetic()
    monkeypatch.setattr(best60, "BDIF_BEST60_SUPPORT_LISTS", len(rows) + 1)

    novel = build_best60("X", rows, player_strength(records), records, "novel")
    observed = build_best60("X", rows, player_strength(records), records, "observed")

    assert novel["swaps"]
    assert observed["swaps"] == []


@pytest.mark.unit
def test_novel_mode_reports_support_without_limiting_it():
    rows, records = _tome_synthetic()
    novel = build_best60("X", rows, player_strength(records), records, "novel")
    assert novel["list_mode"] == "novel"
    assert set(novel["support"]["recommended"]) == {2, 4, 6}
    assert novel["support"]["consensus"][6] >= novel["support"]["consensus"][4] >= novel["support"]["consensus"][2]


@pytest.mark.unit
def test_an_unknown_list_mode_is_rejected():
    with pytest.raises(ValueError, match="list mode"):
        build_best60("X", [_row("e", "p", _base())], {}, None, "theoretical")


@pytest.mark.unit
def test_list_support_counts_lists_within_a_number_of_changes():
    lists = [
        parse_list(_row("e", "p1", [("Mon", 4, "pokemon"), ("Grass Energy", 56, "energy")])),
        parse_list(_row("e", "p2", [("Mon", 4, "pokemon"), ("Tech", 2, "trainer"), ("Grass Energy", 54, "energy")])),
        parse_list(_row("e", "p3", [("Mon", 4, "pokemon"), ("Tech", 4, "trainer"), ("New", 2, "trainer"), ("Grass Energy", 50, "energy")])),
    ]
    support = best60.ListSupport(lists)
    deck = {"Mon": 4, "Grass Energy": 56}
    assert [support.within(deck, changes) for changes in (0, 2, 4, 6)] == [1, 2, 2, 3]
    assert support.within({"Mon": 4, "Unseen": 1, "Grass Energy": 55}, 1) == 1


@pytest.mark.unit
def test_moves_and_the_whole_list_are_priced_with_the_full_covariance():
    model = best60.SlotModel(
        slots=[("A", 1), ("B", 1)],
        beta=np.array([0.0, 0.1]),
        covariance=np.array([[0.01, 0.008], [0.008, 0.01]]),
        prior_sd=0.1,
    )
    expected = float(best60.norm.cdf(0.1 / np.sqrt(0.004)))
    move = best60._priced_move([("A", 1)], [("B", 1)], {("A", 1): 0, ("B", 1): 1}, model)
    assert move["probability"] == pytest.approx(expected)
    assert best60.joint_probability({"A": 1, "Grass Energy": 59}, {"B": 1, "Grass Energy": 59}, model) == pytest.approx(expected)


@pytest.mark.unit
def test_a_card_is_never_left_at_a_count_nobody_plays():
    # Cutting the 4th "Tome" alone looks good (+1), but 3 copies is not a played count; cutting all 4 costs 5.
    slots = [("Tome", copy) for copy in range(1, 5)] + [("Tech", copy) for copy in range(1, 5)]
    beta = np.array([2.0, 2.0, 2.0, -1.0] + [0.0] * 4)
    model = best60.SlotModel(slots=slots, beta=beta, covariance=np.eye(len(slots)) * 1e-4, prior_sd=0.1)
    levels = {"Tome": {0, 4}, "Tech": {0, 1, 2, 3, 4}}

    improved, applied, _ = improve(Counter({"Tome": 4, "Grass Energy": 56}), model, {"Tome": "trainer", "Tech": "trainer"}, {}, levels)

    assert applied == []
    assert improved["Tome"] == 4
