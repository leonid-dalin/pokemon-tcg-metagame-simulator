import sqlite3
from contextlib import closing

import numpy as np
import pytest

from src.ingestion.model import PlayerObservation, fit_card_model, select_model_cards
from src.ingestion.store import LimitlessStore
from src.tournament import monte_carlo


def _stored_names(store: LimitlessStore) -> dict[str, str | None]:
    with closing(sqlite3.connect(store.path)) as connection:
        return dict(connection.execute("SELECT player_id, deck_name FROM standings"))


def _sixty(pokemon: list[tuple[str, int]]) -> dict[str, list[dict[str, object]]]:
    energy = 60 - 4 - sum(count for _, count in pokemon)
    return {
        "pokemon": [{"name": name, "count": count} for name, count in pokemon],
        "trainer": [{"name": "Ultra Ball", "count": 4}],
        "energy": [{"name": "Grass Energy", "count": energy}],
    }


@pytest.mark.unit
def test_field_posterior_interval_is_the_central_95_percent_of_draws():
    alpha = np.array([[1.0, 30.0], [20.0, 1.0]])
    beta = np.array([[1.0, 20.0], [30.0, 1.0]])
    metrics = monte_carlo.posterior_field_metrics(["a", "b"], np.array([0.0, 1.0]), alpha, beta, match_format="BO1", draws=4_000, seed=7)
    samples = np.random.default_rng(7).beta(alpha[0, 1], beta[0, 1], size=(4_000, 1))[:, 0]

    assert metrics["a"]["expected_win_rate_lower"] == pytest.approx(float(np.quantile(samples, 0.025)))
    assert metrics["a"]["expected_win_rate_upper"] == pytest.approx(float(np.quantile(samples, 0.975)))


@pytest.mark.unit
def test_card_intervals_span_1_96_standard_errors():
    observations = []
    for index in range(240):
        has_tech = index % 2 == 0
        observations.append(PlayerObservation(
            "a", "b", frozenset({"Core", "Tech"} if has_tech else {"Core"}),
            frozenset({"Core", "Tech"} if index % 3 == 0 else {"Core"}), int(index % 10 < (7 if has_tech else 4)),
        ))
    model = fit_card_model(observations, ["Tech"])
    coefficients, intervals = model.coefficient_report()
    low, high = intervals["Tech"]

    assert (high - low) / 2 == pytest.approx(1.959964 * model.standard_errors["Tech"], rel=1e-6)
    assert (low + high) / 2 == pytest.approx(coefficients["Tech"])


@pytest.mark.unit
@pytest.mark.parametrize(("players", "selected"), [(49, []), (50, ["Tech"])])
def test_card_selection_needs_the_minimum_players_in_one_deck(players, selected):
    observations = [
        PlayerObservation("a", "b", frozenset({"Tech"} if index % 2 else set()), frozenset(), index % 2)
        for index in range(players)
    ]

    assert select_model_cards(observations, min_players=50) == selected


@pytest.mark.unit
def test_catalogue_backfill_fills_missing_names_and_keeps_stored_ones(tmp_path):
    store = LimitlessStore(tmp_path / "limitless.db", deck_mapping={})
    store.upsert_standings("e", [
        {"player": "stored", "deck": {"id": "zoroark", "name": "Zoroark Box"}},
        {"player": "missing", "deck": {"id": "zoroark"}},
    ])
    with closing(sqlite3.connect(store.path)) as connection, connection:
        connection.execute("INSERT INTO standings (tournament_id, player_id, deck_id, deck_name) VALUES ('e', 'slug', 'slugged', 'slugged')")

    updated = store.backfill_deck_names({"zoroark": "N's Zoroark", "slugged": ""})

    assert updated == 1
    assert _stored_names(store) == {"stored": "Zoroark Box", "missing": "N's Zoroark", "slug": "slugged"}


@pytest.mark.unit
def test_explicitly_unmapped_id_ignores_the_provider_name(tmp_path):
    store = LimitlessStore(tmp_path / "limitless.db", deck_mapping={"retired": None, "kept": "Kept"})
    store.upsert_standings("e", [
        {"player": "p1", "deck": {"id": "retired", "name": "Provider Retired"}},
        {"player": "p2", "deck": {"id": "kept", "name": "Provider Kept"}},
        {"player": "p3", "deck": {"id": "unlisted", "name": "Provider Unlisted"}},
    ])

    assert _stored_names(store) == {"p1": None, "p2": "Kept", "p3": "Provider Unlisted"}
    assert store.unmapped_deck_ids() == ["retired"]


@pytest.mark.unit
@pytest.mark.parametrize(("modal_lists", "expected"), [
    (3, [{"card": "Mon", "copies": 4}, {"card": "Grass Energy", "copies": 52}]),
    (2, [{"card": "Mon", "copies": 4}, {"card": "Grass Energy", "copies": 52}]),
])
def test_skeleton_uses_modal_legal_core_and_reports_support(tmp_path, modal_lists, expected):
    store = LimitlessStore(tmp_path / "limitless.db", deck_mapping={"x": "X"})
    lists = [_sixty([("Mon", 4)])] * modal_lists + [_sixty([("Mon", 3), ("Alt", 1)])] * (4 - modal_lists)
    store.upsert_standings("e", [
        {"player": f"p{index}", "deck": {"id": "x"}, "decklist": decklist}
        for index, decklist in enumerate(lists)
    ])

    result = store.observed_skeleton_details("X")

    assert result["cards"] == expected
    assert result["legal_list_count"] == 4
    assert result["core_support"] == modal_lists
    assert result["core_share"] == pytest.approx(modal_lists / 4)
    assert result["core_count"] == 2
