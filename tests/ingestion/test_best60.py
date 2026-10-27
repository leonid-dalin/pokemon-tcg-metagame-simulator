import types
from collections import Counter

import numpy as np
import pytest

from src.ingestion import best60
from src.ingestion.best60_games import Game, GameModel, ModelSpec
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
def test_player_strength_counts_only_earlier_events():
    records = {("july", "p"): (3, 3), ("october", "p"): (9, 0)}
    dates = {"july": "2026-07-01T00:00:00Z", "october": "2026-10-01T00:00:00Z"}

    earlier = player_strength(records, dates)

    assert earlier[("july", "p")] == pytest.approx(0.0)
    assert earlier[("october", "p")] == pytest.approx(0.0)
    assert player_strength(records)[("july", "p")] == pytest.approx(np.log(19 / 10))


@pytest.mark.unit
def test_forward_folds_put_every_validation_event_after_its_training_events():
    rows = []
    for index in range(10):
        for player in range(3):
            rows.append(_row(f"e{index}", f"p{index}-{player}", _base(), date=f"2026-09-{index + 1:02d}T00:00:00Z"))
    lists = [parse_list(row) for row in rows]
    folds = event_folds(lists)
    dates = {entry.event: entry.date for entry in lists}
    by_event = {entry.event: int(fold) for entry, fold in zip(lists, folds)}

    assert len(set(by_event.values())) == best60.FOLDS
    for event, fold in by_event.items():
        earlier = [other for other, other_fold in by_event.items() if other_fold < fold]
        later = [other for other, other_fold in by_event.items() if other_fold > fold]
        if fold:
            assert earlier and max(dates[name] for name in earlier) < dates[event]
        if fold < best60.FOLDS - 1:
            assert later and min(dates[name] for name in later) > dates[event]


@pytest.mark.unit
def test_lists_without_pairings_stay_out_of_the_loss_when_the_archetype_has_pairings():
    from src.ingestion.best60_games import Game, ModelSpec, build_rows

    lists = [parse_list(_row("e1", "p1", _base())), parse_list(_row("e1", "p2", _base()))]
    games = (Game("e1", "p1", "opponent", "Opp", 1, "2026-09-01T00:00:00Z"),)

    with_games = build_rows(lists, {}, ModelSpec(prior_sd=1.0, games=games))
    without_games = build_rows(lists, {}, ModelSpec(prior_sd=1.0))

    assert list(with_games.list_index) == [0]
    assert list(without_games.list_index) == [0, 1]


@pytest.mark.unit
def test_a_modelled_card_reports_the_probability_lost_when_a_copy_is_removed():
    from src.ingestion.best60_games import GameModel, ModelSpec

    slots = [("Tech A", 1), ("Tech B", 1)]
    model = best60.SlotModel(
        slots=slots,
        beta=np.array([0.5, 0.0]),
        covariance=np.eye(2) * 0.01,
        prior_sd=0.1,
        game=GameModel(
            slots=slots,
            slot_means=np.array([0.0, 0.0]),
            opponents=["O1"],
            specific=[],
            theta=np.array([0.0, 0.0, 0.5, 0.0]),
            covariance=np.eye(4) * 0.01,
            spec=ModelSpec(prior_sd=0.1),
        ),
    )
    deck = {"Tech A": 1, "Tech B": 1, "Grass Energy": 58}

    values = {row["card"]: row for row in best60.card_values(deck, model, {"O1": 1.0})}

    assert values["Tech A"]["status"] == "estimated"
    assert values["Tech A"]["value"] == pytest.approx(1 / (1 + np.exp(-0.5)) - 0.5)
    assert values["Tech A"]["interval"][0] < values["Tech A"]["value"] < values["Tech A"]["interval"][1]
    assert values["Tech B"]["value"] == pytest.approx(0.0)
    assert values["Grass Energy"] == {"card": "Grass Energy", "copies": 58, "value": None, "interval": None, "status": "not estimable"}


@pytest.mark.unit
def test_the_report_says_decided_match_not_game():
    from src.ingestion.best60_games import Game

    rows, records = _synthetic(effect_a=0.8)
    lists = [parse_list(row) for row in rows]
    games = tuple(Game(entry.event, entry.player, "opponent", "Opp", 1, entry.date) for entry in lists)

    result = build_best60("X", rows, player_strength(records), records, games=games)

    assert "decided match" in result["model"]["outcome"]
    assert "each game" not in result["model"]["outcome"]


@pytest.mark.unit
def test_player_strength_scores_simultaneous_events_from_the_totals_before_them():
    records = {("a", "p"): (9, 0), ("b", "p"): (0, 9)}
    dates = {"a": "2026-07-01T00:00:00Z", "b": "2026-07-01T00:00:00Z"}

    strength = player_strength(records, dates)

    assert strength[("a", "p")] == pytest.approx(0.0)
    assert strength[("b", "p")] == pytest.approx(0.0)


@pytest.mark.unit
def test_fold_strength_counts_only_earlier_training_events():
    records = {("early", "p"): (3, 3), ("late", "p"): (9, 0)}
    dates = {"early": "2026-07-01T00:00:00Z", "late": "2026-10-01T00:00:00Z"}

    strength = best60._fold_strength(records, {"early", "late"}, {"validation"}, dates)

    assert strength[("early", "p")] == pytest.approx(0.0)
    assert strength[("late", "p")] == pytest.approx(0.0)


@pytest.mark.unit
def test_cross_validation_trains_only_on_earlier_folds(monkeypatch):
    rows = []
    for index in range(10):
        for player in range(3):
            rows.append(_row(f"e{index}", f"p{index}-{player}", _base(), date=f"2026-09-{index + 1:02d}T00:00:00Z"))
    lists = [parse_list(row) for row in rows]
    folds = event_folds(lists)
    captured = []
    real = best60.fit_game

    def capture(train, strength, spec, *args, **kwargs):
        captured.append(frozenset(entry.event for entry in train))
        return real(train, strength, spec, *args, **kwargs)

    monkeypatch.setattr(best60, "fit_game", capture)
    best60._cross_validated(lists, {}, None, best60.ModelSpec(prior_sd=0.5), False)

    expected = {
        frozenset(entry.event for entry, fold in zip(lists, folds) if fold < block)
        for block in range(1, best60.FOLDS)
    }
    assert set(captured) == expected


@pytest.mark.unit
def test_held_out_gain_trains_only_on_earlier_folds(monkeypatch):
    rows = []
    for index in range(10):
        for player in range(3):
            rows.append(_row(f"e{index}", f"p{index}-{player}", _base(), date=f"2026-09-{index + 1:02d}T00:00:00Z"))
    lists = [parse_list(row) for row in rows]
    folds = event_folds(lists)
    captured = []
    trivial = best60.SlotModel(slots=[], beta=np.array([]), covariance=np.zeros((0, 0)), prior_sd=0.5)

    def capture(train, *_args, **_kwargs):
        captured.append(frozenset(entry.event for entry in train))
        return trivial

    monkeypatch.setattr(best60, "fit_slot_model", capture)
    best60.held_out_gain(lists, {}, 0.5, {}, list_mode="novel")

    expected = [
        frozenset(entry.event for entry, fold in zip(lists, folds) if fold < block)
        for block in range(1, best60.FOLDS)
    ]
    assert captured[0::2] == expected


@pytest.mark.unit
def test_build_model_addons_reads_the_store_event_dates(monkeypatch, tmp_path):
    from src.ingestion import model_cache
    from src.ingestion.model import CardModelNotIdentifiable

    class Store:
        def __init__(self):
            self.read_dates = False

        def player_records(self):
            return {("e1", "p"): (3, 2)}

        def event_dates(self):
            self.read_dates = True
            return {"e1": "2026-09-01T00:00:00Z"}

        def deck_weights(self):
            return {"a": 1.0}

        def archetype_lists(self, deck):
            return []

        def pairings_with_decklists(self, pattern):
            return []

    store = Store()
    monkeypatch.setattr(model_cache, "load_or_fit_card_model",
                        lambda *args: (_ for _ in ()).throw(CardModelNotIdentifiable("rank")))

    model_cache.build_model_addons([], {"a": 1.0}, ["a"], "sha", tmp_path / "cache.json", store)

    assert store.read_dates is True


@pytest.mark.unit
@pytest.mark.parametrize(
    ("slot_mean", "specific"),
    [(0.0, []), (0.5, []), (0.0, ["O1"])],
)
def test_card_value_interval_matches_a_finite_difference_delta_method(slot_mean, specific):
    from dataclasses import replace

    from scipy.stats import norm as scipy_norm

    from src.ingestion.best60_games import GameModel, ModelSpec

    slots = [("Tech A", 1)]
    theta = np.array([0.0, 0.0, 0.5] + ([0.5] if specific else []))
    covariance = np.eye(len(theta)) * 0.01
    game = GameModel(
        slots=slots,
        slot_means=np.array([slot_mean]),
        opponents=["O1"],
        specific=list(specific),
        theta=theta,
        covariance=covariance,
        spec=ModelSpec(prior_sd=0.1),
    )
    model = best60.SlotModel(slots=slots, beta=np.array([0.5]), covariance=np.eye(1) * 0.01, prior_sd=0.1, game=game)
    deck = {"Tech A": 1, "Grass Energy": 59}

    def value_at(perturbed):
        perturbed_model = replace(model, game=replace(game, theta=perturbed))
        return next(row["value"] for row in best60.card_values(deck, perturbed_model, {"O1": 1.0}) if row["card"] == "Tech A")

    row = next(item for item in best60.card_values(deck, model, {"O1": 1.0}) if item["card"] == "Tech A")
    half_width = (row["interval"][1] - row["interval"][0]) / 2
    step = 1e-6
    gradient = np.array([
        (value_at(theta + step * unit) - value_at(theta - step * unit)) / (2 * step)
        for unit in np.eye(len(theta))
    ])
    expected = float(scipy_norm.ppf(0.975)) * np.sqrt(float(gradient @ covariance @ gradient))

    assert half_width == pytest.approx(expected, rel=0.01)


@pytest.mark.unit
def test_best_field_list_needs_the_configured_minimum_players():
    from src.ingestion.best60_games import GameModel, ModelSpec

    rows = [_row("e1", "solo", _base(tech_a=2))] + [_row("e1", f"p{index}", _base(tech_a=1)) for index in range(3)]
    lists = [parse_list(row) for row in rows]
    model = GameModel(
        slots=[("Tech A", 2)],
        slot_means=np.array([0.0]),
        opponents=["O1"],
        specific=[],
        theta=np.array([0.0, 0.0, 2.0]),
        covariance=np.eye(3),
        spec=ModelSpec(prior_sd=0.1),
    )

    picked, players, _ = best60.best_field_list(lists, model, {"O1": 1.0})
    with_solo, solo_players, _ = best60.best_field_list(lists, model, {"O1": 1.0}, minimum=1)

    assert (picked["Tech A"], players) == (1, 3)
    assert (with_solo["Tech A"], solo_players) == (2, 1)


@pytest.mark.unit
def test_the_pick_ships_only_when_its_held_out_gain_clears_the_gate(monkeypatch):
    rows, records = _synthetic(effect_a=0.8)
    lists = [parse_list(row) for row in rows]
    played, played_players = best60.most_played_list(lists)
    pick = Counter({"Mon": 4, "Mon ex": 4, "Ultra Ball": 4, "Tech A": 2, "Air Balloon": 1, "Grass Energy": 45})
    assert pick != played
    monkeypatch.setattr(best60, "best_field_list", lambda *args, **kwargs: (pick, 7, 0.6))
    monkeypatch.setattr(best60, "held_out_gain", lambda *args, **kwargs: (-0.01, 0.0))
    monkeypatch.setattr(best60, "tuned_spec", lambda *args: best60.ModelSpec(prior_sd=0.5))

    monkeypatch.setattr(best60, "seed_gain", lambda *args, **kwargs: (-0.01, 0.0))
    rejected = build_best60("X", rows, player_strength(records), records)
    assert {row["card"]: row["copies"] for row in rejected["cards"]} == dict(played)
    assert rejected["shipped_list"]["players"] == played_players
    assert rejected["model"]["seed_source"] == "most played"

    monkeypatch.setattr(best60, "seed_gain", lambda *args, **kwargs: (0.05, 0.01))
    accepted = build_best60("X", rows, player_strength(records), records)
    assert {row["card"]: row["copies"] for row in accepted["cards"]} == dict(pick)
    assert accepted["shipped_list"]["seed_players"] == 7
    assert accepted["model"]["seed_source"] == "validated pick"


@pytest.mark.unit
def test_held_out_gain_re_picks_inside_each_fold_from_training_blocks(monkeypatch):
    rows = []
    for index in range(10):
        for player in range(3):
            rows.append(_row(f"e{index}", f"p{index}-{player}", _base(), date=f"2026-09-{index + 1:02d}T00:00:00Z"))
    lists = [parse_list(row) for row in rows]
    folds = event_folds(lists)
    seen = []
    trivial = best60.SlotModel(slots=[], beta=np.array([]), covariance=np.zeros((0, 0)), prior_sd=0.5)

    def capture(train, *_args, **_kwargs):
        seen.append(frozenset(entry.event for entry in train))
        return best60.most_played_list(train)[0], 0, "most played", 0.0, 0.0

    monkeypatch.setattr(best60, "fit_slot_model", lambda *args, **kwargs: trivial)
    monkeypatch.setattr(best60, "seed_list", capture)
    monkeypatch.setattr(best60, "improve", lambda deck, *_args, **_kwargs: (Counter(deck), [], []))

    best60.held_out_gain(lists, {}, 0.5, {}, list_mode="novel")

    expected = [
        frozenset(entry.event for entry, fold in zip(lists, folds) if fold < block)
        for block in range(1, best60.FOLDS)
    ]
    assert seen == expected


@pytest.mark.unit
@pytest.mark.unit
def test_a_planted_second_copy_effect_is_applied_and_holds_up_on_held_out_events(monkeypatch):
    monkeypatch.setattr(best60, "seed_gain", lambda *args, **kwargs: (0.05, 0.01))
    rows, records = _synthetic(effect_a=0.8)
    result = build_best60("X", rows, player_strength(records), records)

    assert result["status"] == "complete"
    assert result["model"]["held_out_gain"] - result["model"]["held_out_gain_se"] > 0
    assert {(swap["add"], swap["add_copy"]) for swap in result["swaps"]} >= {("Tech A", 2)}
    assert {row["card"]: row["copies"] for row in result["cards"]}["Tech A"] == 2
    assert result["total_copies"] == 60


@pytest.mark.unit
def test_a_weaker_planted_effect_is_still_applied(monkeypatch):
    monkeypatch.setattr(best60, "seed_gain", lambda *args, **kwargs: (0.05, 0.01))
    rows, records = _synthetic(effect_a=0.4)
    result = build_best60("X", rows, player_strength(records), records)

    assert result["status"] == "complete"
    assert {(swap["add"], swap["add_copy"]) for swap in result["swaps"]} >= {("Tech A", 2)}


@pytest.mark.unit
def test_without_any_card_effect_the_most_played_observed_list_is_kept(monkeypatch):
    rows, records = _synthetic()
    monkeypatch.setattr(best60, "tuned_spec", lambda *args: None)
    result = build_best60("X", rows, player_strength(records))

    played, players = best60.most_played_list([parse_list(row) for row in rows])
    assert result["status"].startswith("observed list kept")
    assert {row["card"]: row["copies"] for row in result["cards"]} == dict(played)
    assert result["shipped_list"]["players"] == players
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
    improved, applied, _ = improve(Counter({"Ultra Ball": 4, "Grass Energy": 56}), model, groups, {"Mon ex": {"Mon": 0}})
    assert all(swap["add"] != "Mon ex" for swap in applied)

    single = best60.SlotModel(slots=[("Mon", 1), ("Ultra Ball", 4)], beta=np.array([-1.0, 1.0]), covariance=np.eye(2) * 1e-4, prior_sd=0.1)
    improved, applied, _ = improve(Counter({"Mon": 1, "Ultra Ball": 3, "Grass Energy": 56}), single, groups, {})
    assert improved["Mon"] == 1
    assert applied == []


@pytest.mark.unit
def test_too_few_lists_return_the_consensus_with_a_status():
    rows = [_row("e", f"p{i}", _base()) for i in range(5)]
    result = build_best60("X", rows, {})
    assert result["status"] == "observed list kept: too few lists to score cards"
    assert result["total_copies"] == 60
    assert result["deck_ids"] == ["x"]


@pytest.mark.unit
def test_one_event_lists_keep_consensus_without_cross_validation():
    rows = [_row("e", f"p{i}", _base()) for i in range(100)]
    result = build_best60("X", rows, {})
    assert result["status"] == "observed list kept: card counts did not predict held-out results"
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
def test_seed_gain_prices_the_pick_on_the_held_out_block(monkeypatch):
    rows, records = _synthetic(lists=120)
    lists = [parse_list(row) for row in rows]
    folds = event_folds(lists)
    blocks = [
        frozenset(entry.event for entry, fold in zip(lists, folds) if fold == block)
        for block in range(1, best60.FOLDS)
    ]
    captured = []
    original = best60.fit_slot_model

    def capture(train, strength, prior_sd):
        captured.append(frozenset(entry.event for entry in train))
        return original(train, strength, prior_sd)

    monkeypatch.setattr(best60, "fit_slot_model", capture)
    best60.seed_gain(lists, {}, 0.5, records)

    assert all(block in captured for block in blocks)


@pytest.mark.unit
def test_seed_list_keeps_the_most_played_list_when_the_gain_is_inside_one_standard_error(monkeypatch):
    rows, _ = _synthetic(lists=60)
    lists = [parse_list(row) for row in rows]
    pick = Counter({"Mon": 4, "Mon ex": 4, "Ultra Ball": 4, "Tech A": 2, "Grass Energy": 46})
    stub = types.SimpleNamespace(game=object())
    monkeypatch.setattr(best60, "best_field_list", lambda *args, **kwargs: (pick, 1, 0.9))
    monkeypatch.setattr(best60, "seed_gain", lambda *args, **kwargs: (0.02, 0.05))

    deck, players, source, gain, se = best60.seed_list(lists, {}, 0.5, None, model=stub)

    assert source == "most played"
    assert deck == Counter(best60.most_played_list(lists)[0])


@pytest.mark.unit
def test_seed_list_scores_the_pick_against_the_specified_field(monkeypatch):
    rows = [_row("e1", f"a{index}", _base(tech_a=2)) for index in range(3)]
    rows += [_row("e1", f"b{index}", _base()) for index in range(3)]
    lists = [parse_list(row) for row in rows]
    game = GameModel(
        slots=[("Tech A", 2)], slot_means=np.array([0.0]), opponents=["Deck X", "Deck Y"],
        specific=["Deck X", "Deck Y"], theta=np.array([0.0, 0.0, 0.0, 0.0, 2.0, -2.0]),
        covariance=np.zeros((6, 6)), spec=ModelSpec(prior_sd=0.5),
    )
    stub = types.SimpleNamespace(game=game)
    monkeypatch.setattr(best60, "seed_gain", lambda *args, **kwargs: (0.05, 0.01))

    first, _, first_source, _, _ = best60.seed_list(lists, {}, 0.5, None, model=stub, field={"Deck X": 1.0})
    second, _, second_source, _, _ = best60.seed_list(lists, {}, 0.5, None, model=stub, field={"Deck Y": 1.0})

    assert first_source == second_source == "validated pick"
    assert first["Tech A"] == 2
    assert second.get("Tech A", 0) == 0


@pytest.mark.unit
def test_seed_gain_uses_the_specified_field_in_every_fold(monkeypatch):
    rows, records = _synthetic(lists=120)
    lists = [parse_list(row) for row in rows]
    game = GameModel(
        slots=[], slot_means=np.array([]), opponents=["Deck X"], specific=[],
        theta=np.array([0.0]), covariance=np.zeros((1, 1)), spec=ModelSpec(prior_sd=0.5),
    )
    stub = types.SimpleNamespace(game=game)
    seen = []
    original = best60._list_score

    def capture(game_arg, deck, field):
        seen.append(dict(field))
        return original(game_arg, deck, field)

    monkeypatch.setattr(best60, "fit_slot_model", lambda *args, **kwargs: stub)
    monkeypatch.setattr(best60, "_list_score", capture)
    best60.seed_gain(lists, {}, 0.5, records, field={"Deck X": 1.0})

    assert seen and all(field == {"Deck X": 1.0} for field in seen)


@pytest.mark.unit
def test_recency_losses_passes_event_dates_to_the_fold_strength(monkeypatch):
    rows = [
        _row(f"e{index}", f"p{index}", _base(), date=f"2026-09-{1 + index % 25:02d}T00:00:00Z")
        for index in range(40)
    ]
    lists = [parse_list(row) for row in rows]
    records = {(entry.event, entry.player): (3, 2) for entry in lists}
    seen = []
    original = best60._fold_strength

    def capture(records_arg, train_events, test_events, dates=None):
        seen.append(dates)
        return original(records_arg, train_events, test_events, dates)

    monkeypatch.setattr(best60, "_fold_strength", capture)
    best60.recency_losses(lists, {}, records, ModelSpec(prior_sd=0.5, as_of="2026-09-30T00:00:00Z"))

    assert seen
    assert seen[0] is not None
    assert set(seen[0]) >= {entry.event for entry in lists[:5]}


@pytest.mark.unit
def test_seed_gain_estimates_the_field_from_training_games_only(monkeypatch):
    rows, records = _synthetic(lists=120)
    lists = [parse_list(row) for row in rows]
    folds = event_folds(lists)
    games = [
        Game(event=entry.event, player=entry.player, opponent="other", opponent_deck="Other", won=1)
        for entry in lists
    ]
    seen = []
    original = best60.observed_field

    def capture(entries, games_arg, *args, **kwargs):
        seen.append(frozenset(entry.event for entry in entries))
        return original(entries, games_arg, *args, **kwargs)

    monkeypatch.setattr(best60, "observed_field", capture)
    best60.seed_gain(lists, {}, 0.5, records, games=games)

    prefixes = {
        frozenset(entry.event for entry, fold in zip(lists, folds) if fold < block)
        for block in range(1, best60.FOLDS)
    }
    blocks = {
        frozenset(entry.event for entry, fold in zip(lists, folds) if fold == block)
        for block in range(1, best60.FOLDS)
    }
    assert seen and set(seen) <= prefixes | blocks
    assert prefixes & set(seen)


@pytest.mark.unit
def test_seed_gain_measures_the_pick_against_the_most_played_list(monkeypatch):
    rows, records = _synthetic(lists=60)
    lists = [parse_list(row) for row in rows]
    pick = Counter({"Mon": 4, "Mon ex": 4, "Ultra Ball": 4, "Tech A": 2, "Grass Energy": 46})
    baseline = Counter({"Mon": 4, "Mon ex": 4, "Ultra Ball": 4, "Tech B": 2, "Grass Energy": 46})
    stub = types.SimpleNamespace(game=object())
    monkeypatch.setattr(best60, "fit_slot_model", lambda *args, **kwargs: stub)
    monkeypatch.setattr(best60, "best_field_list", lambda *args, **kwargs: (pick, 1, 0.9))
    monkeypatch.setattr(best60, "most_played_list", lambda *args, **kwargs: (baseline, 1))
    monkeypatch.setattr(best60, "consensus_sixty", lambda *args, **kwargs: pick)
    monkeypatch.setattr(best60, "_list_score", lambda game, deck, field: 1.0 if deck == pick else 0.0)

    mean, se = best60.seed_gain(lists, {}, 0.5, records)

    assert mean == pytest.approx(1.0)
    assert se == pytest.approx(0.0)


@pytest.mark.unit
def test_held_out_gain_prices_each_fold_on_its_held_out_block(monkeypatch):
    rows, records = _synthetic(effect_a=0.8, lists=600)
    lists = [parse_list(row) for row in rows]
    folds = best60.event_folds(lists)
    blocks = [
        frozenset(entry.event for entry, fold in zip(lists, folds) if fold == block)
        for block in range(1, best60.FOLDS)
    ]
    captured = []
    original = best60.fit_slot_model

    def capture(train, strength, prior_sd):
        captured.append(frozenset(entry.event for entry in train))
        return original(train, strength, prior_sd)

    monkeypatch.setattr(best60, "fit_slot_model", capture)
    mean, se = best60.held_out_gain(lists, player_strength(records), 0.5, {"Tech A": "trainer"}, records)

    assert all(captured.count(block) == 1 for block in blocks)
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
def test_a_failed_card_model_ships_the_most_played_observed_list():
    base = [("Mon", 4, "pokemon"), ("Mon ex", 4, "pokemon"), ("Ultra Ball", 4, "trainer")]
    rows = []
    for index in range(40):
        rows.append(_row("e", f"a{index}", base + [("Tech A", 2, "trainer"), ("Grass Energy", 46, "energy")]))
    for index in range(35):
        rows.append(_row("e", f"b{index}", base + [("Tech B", 2, "trainer"), ("Grass Energy", 46, "energy")]))
    for index in range(25):
        rows.append(_row("e", f"c{index}", base + [("Tech A", 1, "trainer"), ("Tech B", 1, "trainer"), ("Grass Energy", 46, "energy")]))
    lists = [parse_list(row) for row in rows]
    vote = consensus_sixty(lists)

    result = build_best60("X", rows, {})

    shipped = {row["card"]: row["copies"] for row in result["cards"]}
    assert result["status"] == "observed list kept: card counts did not predict held-out results"
    assert shipped["Tech A"] == 2 and shipped.get("Tech B", 0) == 0
    assert shipped != dict(vote)
    assert result["shipped_list"] == {"source": "observed", "players": 40, "seed_players": 40, "seed_field_score": None}


@pytest.mark.unit
def test_a_tied_most_played_list_goes_to_the_one_nearest_the_other_lists():
    base = [("Mon", 4, "pokemon"), ("Mon ex", 4, "pokemon"), ("Ultra Ball", 4, "trainer")]
    rows = []
    for index in range(30):
        rows.append(_row("e", f"b{index}", base + [("Tech B", 2, "trainer"), ("Grass Energy", 46, "energy")]))
    for index in range(30):
        rows.append(_row("e", f"z{index}", base + [("Tech Z", 2, "trainer"), ("Grass Energy", 46, "energy")]))
    for index in range(20):
        rows.append(_row("e", f"c{index}", base + [("Tech Z", 1, "trainer"), ("Grass Energy", 47, "energy")]))
    lists = [parse_list(row) for row in rows]

    deck, players = best60.most_played_list(lists)

    assert players == 30
    assert deck == Counter({**dict((card, count) for card, count, _ in base), "Tech Z": 2, "Grass Energy": 46})


@pytest.mark.unit
def test_build_best60_forwards_the_target_field_and_half_life(monkeypatch):
    rows, records = _synthetic(lists=120)
    strength = player_strength(records)
    deck = Counter({card: count for card, count, _ in _base()})
    stub = types.SimpleNamespace(slots=[], game=None)
    spec = ModelSpec(prior_sd=0.5, half_life=42.0)
    seen = {}

    monkeypatch.setattr(best60, "held_out_losses", lambda *args, **kwargs: {"model": 0.6})
    monkeypatch.setattr(best60, "tuned_spec", lambda *args, **kwargs: spec)
    monkeypatch.setattr(best60, "recency_losses", lambda *args, **kwargs: {})
    monkeypatch.setattr(best60, "fit_slot_model", lambda *args, **kwargs: stub)
    monkeypatch.setattr(best60, "improve", lambda deck, *args, **kwargs: (Counter(deck), [], []))

    def seed_spy(lists, strength, prior_sd, records=None, model=None, **kwargs):
        seen["seed"] = kwargs
        return Counter(deck), 0, "most played", 0.0, 0.0

    def gain_spy(lists, strength, prior_sd, groups, records=None, list_mode="novel", **kwargs):
        seen["gain"] = kwargs
        return 0.0, 0.0

    monkeypatch.setattr(best60, "seed_list", seed_spy)
    monkeypatch.setattr(best60, "held_out_gain", gain_spy)
    best60.build_best60("X", rows, strength, records, field={"Deck X": 1.0}, as_of="2026-09-30T00:00:00Z")

    assert seen["seed"]["field"] == {"Deck X": 1.0}
    assert seen["seed"]["half_life"] == 42.0
    assert seen["gain"]["field"] == {"Deck X": 1.0}
    assert seen["gain"]["half_life"] == 42.0


@pytest.mark.unit
def test_seed_list_forwards_the_field_and_half_life_to_seed_gain(monkeypatch):
    rows, records = _synthetic(lists=60)
    lists = [parse_list(row) for row in rows]
    stub = types.SimpleNamespace(game=object())
    seen = {}

    def gain_spy(lists, strength, prior_sd, records=None, **kwargs):
        seen.update(kwargs)
        return 0.05, 0.01

    monkeypatch.setattr(best60, "seed_gain", gain_spy)
    monkeypatch.setattr(best60, "best_field_list", lambda *args, **kwargs: (Counter(), 0, 0.0))
    best60.seed_list(lists, {}, 0.5, records, model=stub, field={"Deck X": 1.0}, as_of="2026-09-30", half_life=42.0)

    assert seen["field"] == {"Deck X": 1.0}
    assert seen["as_of"] == "2026-09-30"
    assert seen["half_life"] == 42.0


@pytest.mark.unit
def test_held_out_gain_forwards_the_field_to_each_fold(monkeypatch):
    rows, records = _synthetic(lists=120)
    lists = [parse_list(row) for row in rows]
    stub = types.SimpleNamespace(slots=[], beta=np.array([]), game=None)
    calls = []

    def seed_spy(*args, **kwargs):
        calls.append(kwargs)
        return Counter({card: count for card, count, _ in _base()}), 0, "most played", 0.0, 0.0

    monkeypatch.setattr(best60, "fit_slot_model", lambda *args, **kwargs: stub)
    monkeypatch.setattr(best60, "seed_list", seed_spy)
    monkeypatch.setattr(best60, "improve", lambda deck, *args, **kwargs: (Counter(deck), [], []))
    monkeypatch.setattr(best60, "_list_score", lambda *args, **kwargs: 0.0)
    best60.held_out_gain(
        lists, {}, 0.5, {}, records, "novel", field={"Deck X": 1.0}, as_of="2026-09-30", half_life=42.0,
    )

    assert calls
    assert all(call["field"] == {"Deck X": 1.0} for call in calls)
    assert all(call["half_life"] == 42.0 for call in calls)


@pytest.mark.unit
def test_seed_gain_estimates_the_observed_field_with_the_report_settings(monkeypatch):
    rows, records = _synthetic(lists=120)
    lists = [parse_list(row) for row in rows]
    stub = types.SimpleNamespace(game=object())
    seen = []

    def field_spy(entries, games, as_of=None, half_life=None):
        seen.append((as_of, half_life))
        return {"Deck X": 1.0}

    monkeypatch.setattr(best60, "fit_slot_model", lambda *args, **kwargs: stub)
    monkeypatch.setattr(best60, "observed_field", field_spy)
    monkeypatch.setattr(best60, "best_field_list", lambda *args, **kwargs: (Counter(), 0, 0.0))
    monkeypatch.setattr(best60, "_list_score", lambda *args, **kwargs: 0.0)
    best60.seed_gain(lists, {}, 0.5, records, games=[object()], as_of="2026-09-30", half_life=42.0)

    assert seen and all(as_of == "2026-09-30" and half_life == 42.0 for as_of, half_life in seen)


@pytest.mark.unit
def test_seed_gain_scores_strength_from_event_dates(monkeypatch):
    rows, records = _synthetic(lists=120)
    lists = [parse_list(row) for row in rows]
    stub = types.SimpleNamespace(game=object())
    seen = []
    original = best60._fold_strength

    def capture(records_arg, train_events, test_events, dates=None):
        seen.append(dates)
        return original(records_arg, train_events, test_events, dates)

    monkeypatch.setattr(best60, "_fold_strength", capture)
    monkeypatch.setattr(best60, "fit_slot_model", lambda *args, **kwargs: stub)
    monkeypatch.setattr(best60, "best_field_list", lambda *args, **kwargs: (Counter(), 0, 0.0))
    monkeypatch.setattr(best60, "_list_score", lambda *args, **kwargs: 0.0)
    best60.seed_gain(lists, {}, 0.5, records)

    assert seen and all(dates for dates in seen)


@pytest.mark.unit
def test_the_swap_search_starts_from_the_field_scored_observed_list(monkeypatch):
    monkeypatch.setattr(best60, "seed_gain", lambda *args, **kwargs: (0.05, 0.01))
    rows, records = _synthetic(effect_a=0.8)
    chosen = Counter({"Mon": 4, "Mon ex": 4, "Ultra Ball": 4, "Tech A": 2, "Grass Energy": 46})
    seen = {}
    monkeypatch.setattr(best60, "best_field_list", lambda *args: (chosen, 7, 0.51))

    def capture(deck, *_args, **_kwargs):
        seen["deck"] = Counter(deck)
        return Counter(deck), [], []

    monkeypatch.setattr(best60, "improve", capture)
    monkeypatch.setattr(best60, "held_out_gain", lambda *args, **kwargs: (-0.01, 0.0))

    result = build_best60("X", rows, player_strength(records), records)

    assert seen["deck"] == chosen
    assert result["shipped_list"]["source"] == "observed"
    assert result["shipped_list"]["seed_players"] == 7
    assert result["shipped_list"]["seed_field_score"] is not None
    assert result["status"] == "observed list kept: swaps did not hold up on held-out events"


@pytest.mark.unit
def test_field_list_scores_averages_each_opponent_probability_not_its_logit():
    from src.ingestion.best60_games import GameModel, ModelSpec
    from src.ingestion.best60_research.selection import field_scores

    model = GameModel(
        slots=[("Tech A", 1)],
        slot_means=np.array([0.5]),
        opponents=["O1", "O2"],
        specific=[],
        theta=np.array([0.0, 0.5, 0.0, 1.0]),
        covariance=np.eye(4),
        spec=ModelSpec(prior_sd=1.0),
    )
    rows = [(Counter({"Tech A": 1}), 3), (Counter({"Tech A": 0}), 5)]
    field = {"O1": 0.5, "O2": 0.5}

    scores = best60.field_list_scores(model, rows, field)

    logits = model.logits(Counter({"Tech A": 1}), ["O1", "O2"])
    assert scores[0] == pytest.approx(float(field_scores(logits, [0.5, 0.5])))
    assert scores[0] > scores[1]
    assert logits == pytest.approx([0.5, 1.0])


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
    monkeypatch.setattr(best60, "seed_gain", lambda *args, **kwargs: (0.05, 0.01))
    rows, records = _synthetic(effect_a=0.8)
    chosen = Counter({"Mon": 4, "Mon ex": 4, "Ultra Ball": 4, "Tech A": 1, "Grass Energy": 47})
    monkeypatch.setattr(best60, "best_field_list", lambda *args: (chosen, 7, 0.51))
    monkeypatch.setattr(best60, "held_out_gain", lambda *args, **kwargs: (-0.01, 0.0))

    result = build_best60("X", rows, player_strength(records))

    assert result["status"] == "observed list kept: swaps did not hold up on held-out events"
    assert result["swaps"] == []
    assert {(swap["add"], swap["add_copy"]) for swap in result["proposed_swaps"]} >= {("Tech A", 2)}
    shipped = {row["card"]: row["copies"] for row in result["cards"]}
    assert shipped == {card: copies for card, copies in chosen.items()}
    assert result["shipped_list"]["source"] == "observed"
    assert result["shipped_list"]["players"] > 0


@pytest.mark.unit
@pytest.mark.parametrize(("gain", "se", "applied"), [(0.05, 0.01, True), (0.01, 0.02, False)])
def test_swaps_need_the_held_out_gain_to_clear_one_standard_error(monkeypatch, gain, se, applied):
    monkeypatch.setattr(best60, "seed_gain", lambda *args, **kwargs: (0.05, 0.01))
    rows, records = _synthetic(effect_a=0.8)
    monkeypatch.setattr(best60, "held_out_gain", lambda *args, **kwargs: (gain, se))

    result = build_best60("X", rows, player_strength(records), records)

    assert bool(result["swaps"]) is applied
    assert result["status"] == ("complete" if applied else "observed list kept: swaps did not hold up on held-out events")
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
def test_a_card_played_at_zero_or_four_leaves_in_one_move(monkeypatch):
    rows, records = _tome_synthetic()
    seed = Counter({"Mon": 4, "Ultra Ball": 4, "Tome": 4, "Grass Energy": 48})
    monkeypatch.setattr(best60, "best_field_list", lambda *args: (seed, 12, 0.5))
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
    lists = [parse_list(row) for row in rows]
    seed = consensus_sixty(lists)
    monkeypatch.setattr(best60, "best_field_list", lambda *args: (seed, 0, 0.5))
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


def test_a_planted_change_is_stable_across_event_resamples():
    rows, records = _synthetic(effect_a=0.8)
    lists = [parse_list(row) for row in rows]
    consensus = consensus_sixty(lists)
    recommended = dict(consensus, **{"Tech A": 2})
    recommended["Grass Energy"] = consensus["Grass Energy"] - (2 - consensus.get("Tech A", 0))
    groups = {"Mon": "pokemon", "Mon ex": "pokemon", "Ultra Ball": "trainer", "Tech A": "trainer", "Tech B": "trainer", "Grass Energy": "energy"}

    stable = best60.stability(lists, player_strength(records), 0.2, consensus, recommended, groups, "novel", draws=10)

    assert stable["Tech A"]["same_direction"] >= 0.9
    assert stable["Tech A"]["draws"] == 10


@pytest.mark.unit
def test_stability_resamples_whole_events(monkeypatch):
    rows, records = _synthetic(effect_a=0.8, lists=120)
    lists = [parse_list(row) for row in rows]
    sizes = Counter(entry.event for entry in lists)
    samples = []
    original = best60.fit_slot_model

    def capture(sample, strength, prior_sd):
        samples.append(Counter(entry.event for entry in sample))
        return original(sample, strength, prior_sd)

    monkeypatch.setattr(best60, "fit_slot_model", capture)
    consensus = consensus_sixty(lists)
    best60.stability(lists, player_strength(records), 0.2, consensus, dict(consensus, **{"Tech A": 3}), {}, "novel", draws=3)

    assert len(samples) == 3
    assert all(count % sizes[event] == 0 for sample in samples for event, count in sample.items())


@pytest.mark.unit
def test_a_resample_that_keeps_the_consensus_counts_as_neither_same_count_nor_same_direction(monkeypatch):
    rows, records = _synthetic(lists=120)
    lists = [parse_list(row) for row in rows]
    consensus = consensus_sixty(lists)
    recommended = dict(consensus)
    recommended["Ultra Ball"] = consensus["Ultra Ball"] - 1
    recommended["Grass Energy"] = consensus["Grass Energy"] + 1
    monkeypatch.setattr(best60, "improve", lambda deck, *args: (Counter(deck), [], []))

    stable = best60.stability(lists, player_strength(records), 0.2, consensus, recommended, {}, "novel", draws=2)

    assert {card: (row["same_count"], row["same_direction"]) for card, row in stable.items()} == {
        "Grass Energy": (0.0, 0.0), "Ultra Ball": (0.0, 0.0),
    }


@pytest.mark.unit
def test_stability_is_skipped_without_changes_or_draws():
    rows, records = _synthetic(lists=120)
    lists = [parse_list(row) for row in rows]
    consensus = consensus_sixty(lists)
    assert best60.stability(lists, player_strength(records), 0.2, consensus, dict(consensus), {}, "novel", draws=5) == {}
    assert best60.stability(lists, player_strength(records), 0.2, consensus, dict(consensus, **{"Tech A": 2}), {}, "novel", draws=0) == {}
