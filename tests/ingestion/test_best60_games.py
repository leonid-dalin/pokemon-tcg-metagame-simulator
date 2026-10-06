from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from src.ingestion import best60, best60_games
from src.ingestion.best60 import build_best60, fit_slot_model, parse_list, player_strength
from src.ingestion.best60_games import UNKNOWN, Game, ModelSpec, build_rows, fit_game_model, normalise_field, observed_field, recency_weight

START = datetime(2026, 7, 1, tzinfo=timezone.utc)


def _cards(tech):
    cards = [("Mon", 4, "pokemon"), ("Ultra Ball", 4, "trainer")]
    if tech:
        cards.append(("Tech A", tech, "trainer"))
    cards.append(("Grass Energy", 52 - tech, "energy"))
    return cards


def _row(event, player, cards, wins, losses, date):
    groups = {"pokemon": [], "trainer": [], "energy": []}
    for card, count, group in cards:
        groups[group].append({"name": card, "count": count})
    return {"event": event, "player": player, "deck_id": "x", "placing": None, "wins": wins, "losses": losses,
            "decklist": groups, "players": 32, "date": date}


def _games(effect_vs_b=1.0, effect_vs_c=0.0, lists=600, seed=5, late_only=False):
    """Lists play six games each against decks B and C; the 2nd Tech A copy helps only against the given decks."""
    rng = np.random.default_rng(seed)
    rows, games, records = [], [], {}
    for index in range(lists):
        event = f"e{index % 61}"
        day = (index % 61)
        date = (START + timedelta(days=day)).isoformat()
        player = f"p{index}"
        skill = rng.normal(0.0, 1.0)
        tech = int(rng.integers(0, 3))
        wins = losses = 0
        for game_number in range(6):
            deck = "B" if rng.random() < 0.5 else "C"
            opponent = f"o{index}-{game_number}"
            opponent_skill = rng.normal(0.0, 1.0)
            active = (not late_only) or day >= 40
            effect = (effect_vs_b if deck == "B" else effect_vs_c) * (tech >= 2) * active
            logit = 0.6 * (skill - opponent_skill) + (0.2 if deck == "B" else -0.2) + effect
            won = int(rng.random() < 1.0 / (1.0 + np.exp(-logit)))
            wins, losses = wins + won, losses + 1 - won
            games.append(Game(event, player, opponent, deck, won, date))
            other = int(rng.binomial(40, 1.0 / (1.0 + np.exp(-0.6 * opponent_skill))))
            records[(event, opponent)] = (1 - won, won)
            records[(f"other-{opponent}", opponent)] = (other, 40 - other)
        rows.append(_row(event, player, _cards(tech), wins, losses, date))
        records[(event, player)] = (wins, losses)
        other = int(rng.binomial(40, 1.0 / (1.0 + np.exp(-0.6 * skill))))
        records[(f"other-{player}", player)] = (other, 40 - other)
    return rows, games, records


@pytest.mark.unit
def test_game_rows_use_both_players_strength_and_lists_without_pairings_use_their_record():
    rows = [_row("e", "p1", _cards(0), 3, 2, "2026-07-01"), _row("e", "p2", _cards(0), 4, 1, "2026-07-01")]
    lists = [parse_list(row) for row in rows]
    strength = {("e", "p1"): 0.5, ("e", "p2"): 0.1, ("e", "q"): 0.2}
    games = (Game("e", "p1", "q", "B", 1, "2026-07-01"),)

    built = build_rows(lists, strength, ModelSpec(prior_sd=0.1, games=games))

    assert built.opponent == ["B", UNKNOWN]
    assert built.strength.tolist() == pytest.approx([0.5 - 0.2, 0.1])
    assert built.games.tolist() == [1, 5]
    assert built.wins.tolist() == [1, 4]


@pytest.mark.unit
def test_recency_halves_a_game_every_half_life():
    reference = datetime(2026, 9, 1, tzinfo=timezone.utc)
    assert recency_weight("2026-09-01T00:00:00Z", reference, 21.0) == pytest.approx(1.0)
    assert recency_weight("2026-08-11T00:00:00Z", reference, 21.0) == pytest.approx(0.5)
    assert recency_weight("2026-08-11T00:00:00Z", reference, None) == 1.0


@pytest.mark.unit
def test_fields_are_normalised_and_the_observed_field_follows_recency():
    assert normalise_field({"B": 2, "C": 2, "D": 0}) == {"B": 0.5, "C": 0.5}
    rows = [_row("e1", "p", _cards(0), 1, 1, "2026-07-01T00:00:00Z"), _row("e2", "p", _cards(0), 1, 1, "2026-08-12T00:00:00Z")]
    lists = [parse_list(row) for row in rows]
    games = [Game("e1", "p", "q", "B", 1, "2026-07-01T00:00:00Z"), Game("e2", "p", "r", "C", 0, "2026-08-12T00:00:00Z")]
    assert observed_field(lists, games) == {"B": 0.5, "C": 0.5}
    weighted = observed_field(lists, games, half_life=42.0)
    assert weighted["C"] == pytest.approx(2 / 3)


@pytest.mark.unit
def test_an_opponent_specific_effect_shows_against_its_deck_only():
    rows, games, records = _games(effect_vs_b=1.0, effect_vs_c=0.0)
    lists = [parse_list(row) for row in rows]
    spec = ModelSpec(prior_sd=0.5, opponent_sd=0.5, games=tuple(games))
    model = fit_game_model(lists, player_strength(records), [("Tech A", 2)], spec)

    against_b, _ = model.against("B")
    against_c, _ = model.against("C")

    assert model.specific == ["B", "C"]
    assert against_b[0] > 0.5
    assert abs(against_c[0]) < 0.25


@pytest.mark.unit
def test_the_field_decides_whether_a_matchup_tech_is_worth_its_slot():
    # The 2nd Tech A wins games against B and loses them against C: worth a slot only into a B field.
    rows, games, records = _games(effect_vs_b=1.0, effect_vs_c=-1.0)
    lists = [parse_list(row) for row in rows]
    strength = player_strength(records)
    spec = ModelSpec(prior_sd=0.5, opponent_sd=0.5, games=tuple(games))

    into_b = fit_slot_model(lists, strength, ModelSpec(**{**spec.__dict__, "field": {"B": 1.0}}))
    into_c = fit_slot_model(lists, strength, ModelSpec(**{**spec.__dict__, "field": {"C": 1.0}}))
    slot = into_b.slots.index(("Tech A", 2))

    assert into_b.beta[slot] > 0.25 > -0.25 > into_c.beta[slot]

    report_b = build_best60("X", rows, strength, records, games=games, field={"B": 1.0})
    report_c = build_best60("X", rows, strength, records, games=games, field={"C": 1.0})
    assert {row["card"]: row["copies"] for row in report_b["cards"]}.get("Tech A") == 2
    assert {row["card"]: row["copies"] for row in report_c["cards"]}.get("Tech A", 0) < 2
    assert report_b["field"] == {"B": 1.0}
    assert report_b["model"]["field_source"] == "specified"


@pytest.mark.unit
def test_the_tuning_keeps_opponent_effects_only_when_they_predict_better():
    better = {"strength only": 0.70, "prior sd 0.05": 0.68, "opponent sd 0.05": 0.67}
    worse = {"strength only": 0.70, "prior sd 0.05": 0.68, "opponent sd 0.05": 0.69}
    assert best60.tuned_spec(better, None).opponent_sd == 0.05
    assert best60.tuned_spec(worse, None).opponent_sd == 0.0
    assert best60.tuned_spec({"strength only": 0.60, "prior sd 0.05": 0.68}, None) is None


@pytest.mark.unit
def test_the_report_uses_the_half_life_that_predicted_recent_games_best(monkeypatch):
    rows, games, records = _games()
    monkeypatch.setattr(best60, "recency_losses", lambda *args: {"no recency weighting": 0.70, "half-life 21 days": 0.69})

    report = build_best60("X", rows, player_strength(records), records, games=games)

    assert report["model"]["half_life_days"] == 21.0
    assert report["model"]["recency_loss"] == {"no recency weighting": 0.70, "half-life 21 days": 0.69}


@pytest.mark.unit
def test_recency_losses_score_the_last_days_with_a_model_fitted_on_the_days_before():
    rows, games, records = _games(late_only=True, effect_vs_c=1.0)
    lists = [parse_list(row) for row in rows]
    losses = best60.recency_losses(lists, player_strength(records), records, ModelSpec(prior_sd=0.5, games=tuple(games)))

    assert set(losses) == {"no recency weighting", "half-life 42 days", "half-life 21 days", "half-life 14 days"}
    assert min(losses, key=losses.get) != "no recency weighting"


@pytest.mark.unit
def test_as_of_ignores_every_list_and_game_on_or_after_the_date():
    rows, games, records = _games()
    cutoff = (START + timedelta(days=40)).isoformat()
    before_rows = [row for row in rows if row["date"] < cutoff]
    before_games = [game for game in games if game.date < cutoff]
    strength = player_strength(records)

    pinned = build_best60("X", rows, strength, records, games=games, as_of=cutoff)
    truncated = build_best60("X", before_rows, strength, records, games=before_games, as_of=cutoff)

    assert pinned["legal_list_count"] == len(before_rows)
    assert pinned["cards"] == truncated["cards"]
    assert pinned["model"]["held_out_loss"] == truncated["model"]["held_out_loss"]
    assert pinned["model"]["as_of"] == cutoff


@pytest.mark.unit
def test_an_unreadable_as_of_date_is_rejected():
    with pytest.raises(ValueError, match="as-of"):
        build_best60("X", [], {}, None, as_of="last Tuesday")
