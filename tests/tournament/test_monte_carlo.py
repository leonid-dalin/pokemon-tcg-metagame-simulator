import numpy as np
import pytest
import json
import itertools
from pathlib import Path

from src.api.models import PrecisionTier, TIER_MAPPING
from src.core.config import BDIF_MIN_INTERVAL_DRAWS, BDIF_MIN_ITERATIONS_PER_DRAW, BDIF_POSTERIOR_DRAWS
from src.tournament import monte_carlo


@pytest.mark.unit
def test_chunk_seeds_come_from_the_supplied_seed(monkeypatch):
    seeds = []

    monkeypatch.setattr(monte_carlo.tcg_engine, "initialize_rayon", lambda cores: None)

    def run_parallel(iterations, players, meta, matrix, d1, cut, d2, top, seed, *args):
        seeds.append(seed)
        return [1], [1], [1], [1]

    monkeypatch.setattr(monte_carlo.tcg_engine, "run_parallel_monte_carlo", run_parallel)

    kwargs = {
        "deck_names": ["a"],
        "win_matrix": np.array([[0.5]]),
        "meta_distribution": {"a": 1.0},
        "d1_rounds": 1,
        "cut_points": 1,
        "d2_rounds": 1,
        "top_cut": 1,
        "iterations": 20_000,
        "seed": 1234,
    }
    monte_carlo.run_monte_carlo_analytics(**kwargs)
    monte_carlo.run_monte_carlo_analytics(**kwargs)

    assert seeds == [1234, 11234, 1234, 11234]


@pytest.mark.unit
def test_empty_meta_distribution_uses_a_uniform_field(monkeypatch):
    distributions = []

    monkeypatch.setattr(monte_carlo.tcg_engine, "initialize_rayon", lambda cores: None)

    def run_parallel(iterations, players, meta, matrix, d1, cut, d2, top, seed, *args):
        distributions.append(meta)
        return [1, 1], [1, 1], [1, 1], [1, 1]

    monkeypatch.setattr(monte_carlo.tcg_engine, "run_parallel_monte_carlo", run_parallel)

    monte_carlo.run_monte_carlo_analytics(
        deck_names=["a", "b"],
        win_matrix=np.array([[0.5, 0.6], [0.4, 0.5]]),
        meta_distribution={},
        d1_rounds=1,
        cut_points=1,
        d2_rounds=1,
        top_cut=1,
        iterations=1,
    )

    assert distributions == [[0.5, 0.5]]


TWO_DECK_DETAILS = {
    ("a", "b"): {"win_rate": 0.6, "match_count": 2_000},
    ("b", "a"): {"win_rate": 0.4, "match_count": 2_000},
}

EXPECTED_DRAWS_BY_TIER = {
    PrecisionTier.BULLET: 10,
    PrecisionTier.BLITZ: 100,
    PrecisionTier.STANDARD: 200,
    PrecisionTier.EXHAUSTIVE: 200,
    PrecisionTier.MAXIMUM: 200,
}


@pytest.mark.parametrize("tier", list(PrecisionTier))
@pytest.mark.unit
def test_posterior_draw_budget_follows_the_precision_tier(monkeypatch, tier):
    calls = []
    monkeypatch.setattr(monte_carlo.tcg_engine, "initialize_rayon", lambda cores: None)
    monkeypatch.setattr(
        monte_carlo.tcg_engine,
        "run_parallel_monte_carlo",
        lambda n, *args: (calls.append(n) or ([n, n], [n // 2, n // 2], [1, 1], [1, 1])),
    )

    result = monte_carlo.run_monte_carlo_analytics(
        deck_names=["a", "b"],
        win_matrix=np.array([[0.5, 0.6], [0.4, 0.5]]),
        meta_distribution={"a": 0.5, "b": 0.5},
        matchup_details=TWO_DECK_DETAILS,
        d1_rounds=1,
        cut_points=1,
        d2_rounds=1,
        top_cut=1,
        iterations=TIER_MAPPING[tier],
    )

    expected_draws = min(
        BDIF_POSTERIOR_DRAWS,
        max(1, TIER_MAPPING[tier] // BDIF_MIN_ITERATIONS_PER_DRAW),
    )
    assert len(calls) == expected_draws
    assert sum(calls) == TIER_MAPPING[tier]
    assert result["posterior"]["draws"] == expected_draws
    assert result["posterior"]["interval_status"] == (
        "ok" if expected_draws >= BDIF_MIN_INTERVAL_DRAWS else "too few posterior draws"
    )
    assert ("win_probability_lower" in result["metrics"]["a"]) is (
        expected_draws >= BDIF_MIN_INTERVAL_DRAWS
    )


@pytest.mark.unit
def test_metrics_carry_binomial_monte_carlo_standard_errors(monkeypatch):
    monkeypatch.setattr(monte_carlo.tcg_engine, "initialize_rayon", lambda cores: None)
    monkeypatch.setattr(
        monte_carlo.tcg_engine,
        "run_parallel_monte_carlo",
        lambda *args: ([400, 400], [100, 300], [40, 40], [4, 36]),
    )

    result = monte_carlo.run_monte_carlo_analytics(
        deck_names=["a", "b"],
        win_matrix=np.array([[0.5, 0.6], [0.4, 0.5]]),
        meta_distribution={"a": 0.5, "b": 0.5},
        d1_rounds=1,
        cut_points=1,
        d2_rounds=1,
        top_cut=1,
        iterations=10,
    )

    deck = result["metrics"]["a"]
    assert deck["day2_conversion"] == pytest.approx(0.25)
    assert deck["day2_conversion_mc_se"] == pytest.approx((0.25 * 0.75 / 400) ** 0.5)
    assert deck["top_cut_conversion_mc_se"] == pytest.approx((0.1 * 0.9 / 400) ** 0.5)
    assert deck["win_probability_mc_se"] == pytest.approx((0.01 * 0.99 / 400) ** 0.5)
    assert result["posterior"] == {"draws": 0, "interval_status": "posterior disabled"}


@pytest.mark.unit
def test_empty_deck_result_keeps_the_engine_report_shape():
    result = monte_carlo.run_monte_carlo_analytics(
        deck_names=[],
        win_matrix=np.empty((0, 0)),
        meta_distribution={},
        d1_rounds=1,
        cut_points=1,
        d2_rounds=1,
        top_cut=1,
        panel_decks=["Unknown deck"],
    )

    assert result == {
        "metrics": {},
        "ranked_metrics": {},
        "insufficient_data": [],
        "posterior": {"draws": 0, "interval_status": "posterior disabled"},
        "field_posterior": {},
        "matchup_panel": {
            "rows": {},
            "unmatched": ["Unknown deck"],
            "opponents": [],
        },
    }


@pytest.mark.unit
def test_hierarchical_posterior_shrinks_a_six_match_pair_toward_field_prior():
    alpha, beta, _ = monte_carlo.build_hierarchical_beta_posteriors(
        ["a", "b", "c"],
        np.full((3, 3), 0.5),
        {
            ("a", "a"): {"win_rate": 0.5, "match_count": 0},
            ("b", "b"): {"win_rate": 0.5, "match_count": 0},
            ("c", "c"): {"win_rate": 0.5, "match_count": 0},
            ("a", "b"): {"win_rate": 1.0, "match_count": 6},
            ("b", "a"): {"win_rate": 0.0, "match_count": 6},
            ("a", "c"): {"win_rate": 0.5, "match_count": 1_000},
            ("c", "a"): {"win_rate": 0.5, "match_count": 1_000},
        },
    )

    posterior_mean = alpha[0, 1] / (alpha[0, 1] + beta[0, 1])
    reverse_mean = alpha[1, 0] / (alpha[1, 0] + beta[1, 0])
    assert posterior_mean == pytest.approx(11 / 16)
    assert posterior_mean + reverse_mean == pytest.approx(1.0)


@pytest.mark.unit
def test_winless_deck_pair_gets_positive_posterior_pseudocounts(monkeypatch):
    monkeypatch.setattr(monte_carlo.tcg_engine, "initialize_rayon", lambda cores: None)
    monkeypatch.setattr(
        monte_carlo.tcg_engine,
        "run_parallel_monte_carlo",
        lambda *args: ([1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]),
    )

    deck_names = ["winless-a", "winless-b", "winner"]
    win_matrix = np.array([
        [0.5, 0.0, 0.0],
        [1.0, 0.5, 0.0],
        [1.0, 1.0, 0.5],
    ])
    matchup_details = {
        ("winless-a", "winner"): {"win_rate": 0.0, "match_count": 100},
        ("winless-b", "winner"): {"win_rate": 0.0, "match_count": 100},
        ("winner", "winless-a"): {"win_rate": 1.0, "match_count": 100},
        ("winner", "winless-b"): {"win_rate": 1.0, "match_count": 100},
    }
    alpha, beta, _ = monte_carlo.build_hierarchical_beta_posteriors(
        deck_names, win_matrix, matchup_details,
    )
    pair_indices = np.triu_indices(len(deck_names), k=1)
    assert np.all(alpha[pair_indices] > 0)
    assert np.all(beta[pair_indices] > 0)

    result = monte_carlo.run_monte_carlo_analytics(
        deck_names=deck_names,
        win_matrix=win_matrix,
        meta_distribution={"winless-a": 0.4, "winless-b": 0.4, "winner": 0.2},
        matchup_details=matchup_details,
        d1_rounds=1,
        cut_points=1,
        d2_rounds=1,
        top_cut=1,
        iterations=1,
        posterior_draws=1,
    )

    assert set(result) == {"metrics", "ranked_metrics", "insufficient_data", "matchup_panel", "posterior", "field_posterior"}


@pytest.mark.unit
def test_hierarchical_posterior_large_sample_barely_moves_from_observation():
    alpha, beta, _ = monte_carlo.build_hierarchical_beta_posteriors(
        ["a", "b"],
        np.array([[0.5, 0.7], [0.3, 0.5]]),
        {
            ("a", "a"): {"win_rate": 0.5, "match_count": 0},
            ("b", "b"): {"win_rate": 0.5, "match_count": 0},
            ("a", "b"): {"win_rate": 0.7, "match_count": 18_000},
            ("b", "a"): {"win_rate": 0.3, "match_count": 18_000},
        },
    )

    posterior_mean = alpha[0, 1] / (alpha[0, 1] + beta[0, 1])
    assert posterior_mean == pytest.approx(0.7, abs=0.001)


@pytest.mark.unit
def test_data_sufficiency_filter_lists_a_low_coverage_deck():
    _, _, insufficient = monte_carlo.build_hierarchical_beta_posteriors(
        ["a", "b", "c", "d", "e"],
        np.full((5, 5), 0.5),
        {
            ("a", "a"): {"win_rate": 0.5, "match_count": 0},
            ("b", "b"): {"win_rate": 0.5, "match_count": 0},
            ("c", "c"): {"win_rate": 0.5, "match_count": 0},
            ("d", "d"): {"win_rate": 0.5, "match_count": 0},
            ("e", "e"): {"win_rate": 0.5, "match_count": 0},
            ("a", "b"): {"win_rate": 0.6, "match_count": 400},
            ("b", "a"): {"win_rate": 0.4, "match_count": 400},
            ("a", "c"): {"win_rate": 0.6, "match_count": 200},
            ("c", "a"): {"win_rate": 0.4, "match_count": 200},
            ("a", "d"): {"win_rate": 0.6, "match_count": 200},
            ("d", "a"): {"win_rate": 0.4, "match_count": 200},
            ("a", "e"): {"win_rate": 0.6, "match_count": 200},
            ("e", "a"): {"win_rate": 0.4, "match_count": 200},
        },
    )

    assert "a" in insufficient


def test_data_sufficiency_filter_separates_qualified_and_insufficient_decks():
    details = {
        ("a", "a"): {"win_rate": 0.5, "match_count": 0},
        ("b", "b"): {"win_rate": 0.5, "match_count": 0},
        ("a", "b"): {"win_rate": 0.6, "match_count": 1_000},
        ("b", "a"): {"win_rate": 0.4, "match_count": 1_000},
    }
    _, _, insufficient = monte_carlo.build_hierarchical_beta_posteriors(
        ["a", "b"], np.full((2, 2), 0.5), details,
    )
    assert insufficient == []
    details[("a", "b")]["match_count"] = 300
    details[("b", "a")]["match_count"] = 300
    _, _, insufficient = monte_carlo.build_hierarchical_beta_posteriors(
        ["a", "b"], np.full((2, 2), 0.5), details,
    )
    assert insufficient == ["a", "b"]


@pytest.mark.unit
def test_data_sufficiency_filter_rejects_low_total_matches_with_full_coverage():
    details = {
        ("a", "a"): {"win_rate": 0.5, "match_count": 0},
        ("b", "b"): {"win_rate": 0.5, "match_count": 0},
        ("a", "b"): {"win_rate": 0.6, "match_count": 100},
        ("b", "a"): {"win_rate": 0.4, "match_count": 100},
    }
    _, _, insufficient = monte_carlo.build_hierarchical_beta_posteriors(
        ["a", "b"], np.full((2, 2), 0.5), details,
    )
    assert insufficient == ["a", "b"]


@pytest.mark.unit
def test_posterior_report_contains_intervals_and_ranked_split(monkeypatch):
    monkeypatch.setattr(monte_carlo.tcg_engine, "initialize_rayon", lambda cores: None)
    monkeypatch.setattr(
        monte_carlo.tcg_engine,
        "run_parallel_monte_carlo",
        lambda *args: ([10, 10], [5, 5], [2, 2], [1, 1]),
    )

    result = monte_carlo.run_monte_carlo_analytics(
        deck_names=["a", "b"],
        win_matrix=np.array([[0.5, 0.6], [0.4, 0.5]]),
        meta_distribution={"a": 0.5, "b": 0.5},
        matchup_details={
            ("a", "a"): {"win_rate": 0.5, "match_count": 0},
            ("b", "b"): {"win_rate": 0.5, "match_count": 0},
            ("a", "b"): {"win_rate": 0.6, "match_count": 2_000},
            ("b", "a"): {"win_rate": 0.4, "match_count": 2_000},
        },
        d1_rounds=1,
        cut_points=1,
        d2_rounds=1,
        top_cut=1,
        iterations=5_000,
        posterior_draws=50,
    )

    assert set(result) == {"metrics", "ranked_metrics", "insufficient_data", "matchup_panel", "posterior", "field_posterior"}
    assert result["posterior"] == {"draws": 50, "interval_status": "ok"}
    assert not result["insufficient_data"]
    assert result["ranked_metrics"]["a"]["day2_share_lower"] <= result["ranked_metrics"]["a"]["day2_share_upper"]


@pytest.mark.unit
def test_matchup_panel_marks_thin_pair_unreliable_and_keeps_unknown_decks():
    names = ["Crustle", "N's Zoroark", "a", "b", "c", "d", "e"]
    matrix = np.full((7, 7), 0.5)
    details = {}
    for i, left in enumerate(names):
        for j, right in enumerate(names):
            details[(left, right)] = {"win_rate": 0.5, "match_count": 0 if i == j else 3000}
    details[("Crustle", "a")] = {"win_rate": 0.7, "match_count": 6}
    details[("a", "Crustle")] = {"win_rate": 0.3, "match_count": 6}
    alpha, beta, _ = monte_carlo.build_hierarchical_beta_posteriors(names, matrix, details)
    panel = monte_carlo.build_matchup_panel(
        names,
        np.array([0.30, 0.20, 0.15, 0.12, 0.10, 0.08, 0.05]),
        alpha,
        beta,
        details,
        panel_decks=["Crustle", "N's Zoroark", "Unknown deck"],
    )

    crustle_a = next(row for row in panel["rows"]["Crustle"] if row["opponent"] == "a")
    crustle_b = next(row for row in panel["rows"]["Crustle"] if row["opponent"] == "b")
    zoroark_mirror = next(row for row in panel["rows"]["N's Zoroark"] if row["mirror"])
    assert crustle_a["reliable"] is False
    assert crustle_a["upper"] - crustle_a["lower"] > crustle_b["upper"] - crustle_b["lower"]
    assert crustle_b["reliable"] is True
    assert zoroark_mirror["mean"] == 0.5
    assert panel["unmatched"] == ["Unknown deck"]


@pytest.mark.unit
def test_exposes_matchup_panel(monkeypatch):
    monkeypatch.setattr(monte_carlo.tcg_engine, "initialize_rayon", lambda cores: None)
    monkeypatch.setattr(
        monte_carlo.tcg_engine,
        "run_parallel_monte_carlo",
        lambda *args: ([10, 10], [5, 5], [2, 2], [1, 1]),
    )

    result = monte_carlo.run_monte_carlo_analytics(
        deck_names=["Crustle", "a"],
        win_matrix=np.array([[0.5, 0.6], [0.4, 0.5]]),
        meta_distribution={"Crustle": 0.7, "a": 0.3},
        matchup_details={
            ("Crustle", "a"): {"win_rate": 0.6, "match_count": 3000},
            ("a", "Crustle"): {"win_rate": 0.4, "match_count": 3000},
        },
        d1_rounds=1,
        cut_points=1,
        d2_rounds=1,
        top_cut=1,
        iterations=1,
        posterior_draws=1,
        panel_decks=["Crustle"],
    )
    assert "matchup_panel" in result
    assert result["matchup_panel"]["rows"]["Crustle"]



@pytest.mark.unit
def test_posterior_draws_keep_matchup_matrix_complementary(monkeypatch):
    matrices = []
    monkeypatch.setattr(monte_carlo.tcg_engine, "initialize_rayon", lambda cores: None)

    def run_parallel(iterations, players, meta, matrix, *args):
        matrices.append(np.asarray(matrix))
        return [1, 1], [1, 1], [1, 1], [1, 1]

    monkeypatch.setattr(monte_carlo.tcg_engine, "run_parallel_monte_carlo", run_parallel)

    monte_carlo.run_monte_carlo_analytics(
        deck_names=["a", "b"],
        win_matrix=np.array([[0.5, 0.7], [0.3, 0.5]]),
        meta_distribution={"a": 0.5, "b": 0.5},
        matchup_details={
            ("a", "b"): {"win_rate": 0.7, "match_count": 100},
            ("b", "a"): {"win_rate": 0.3, "match_count": 100},
        },
        d1_rounds=1,
        cut_points=1,
        d2_rounds=1,
        top_cut=1,
        iterations=2,
        posterior_draws=2,
    )

    assert matrices
    assert all(np.allclose(matrix + matrix.T, 1.0) for matrix in matrices)


@pytest.mark.unit
def test_posterior_draws_preserve_requested_iterations(monkeypatch):
    iterations_seen = []
    monkeypatch.setattr(monte_carlo.tcg_engine, "initialize_rayon", lambda cores: None)
    monkeypatch.setattr(
        monte_carlo.tcg_engine,
        "run_parallel_monte_carlo",
        lambda iterations, *args: (iterations_seen.append(iterations) or ([1], [1], [1], [1])),
    )

    monte_carlo.run_monte_carlo_analytics(
        deck_names=["a"],
        win_matrix=np.array([[0.5]]),
        meta_distribution={"a": 1.0},
        matchup_details={("a", "a"): {"win_rate": 0.5, "match_count": 0}},
        d1_rounds=1,
        cut_points=1,
        d2_rounds=1,
        top_cut=1,
        iterations=999,
        posterior_draws=25,
    )

    assert sum(iterations_seen) == 999



@pytest.mark.unit
@pytest.mark.parametrize("order", list(itertools.permutations(["S", "T", "W"])))
def test_posterior_is_invariant_to_deck_order(order):
    observed = {("S", "W"): 0.7, ("W", "S"): 0.3, ("T", "W"): 0.7, ("W", "T"): 0.3}
    details = {
        (deck, opponent): {
            "win_rate": observed.get((deck, opponent), 0.5),
            "match_count": 2_000 if (deck, opponent) in observed else 0,
        }
        for deck in order
        for opponent in order
        if deck != opponent
    }

    alpha, beta, _ = monte_carlo.build_hierarchical_beta_posteriors(
        list(order), np.full((3, 3), 0.5), details,
    )

    index = {deck: position for position, deck in enumerate(order)}
    means = {
        (deck, opponent): alpha[index[deck], index[opponent]]
        / (alpha[index[deck], index[opponent]] + beta[index[deck], index[opponent]])
        for deck in order
        for opponent in order
        if deck != opponent
    }
    assert means[("S", "T")] == pytest.approx(0.5)
    assert means[("S", "W")] == pytest.approx(means[("T", "W")])
    for (deck, opponent), value in means.items():
        assert value + means[(opponent, deck)] == pytest.approx(1.0)
def _pairwise_posterior(pairs, n_decks):
    alpha = np.ones((n_decks, n_decks))
    beta = np.ones((n_decks, n_decks))
    for (i, j), (p, n) in pairs.items():
        alpha[i, j], beta[i, j] = p * n, (1 - p) * n
        alpha[j, i], beta[j, i] = (1 - p) * n, p * n
    return alpha, beta


@pytest.mark.unit
def test_field_posterior_best_pick_probabilities_sum_to_one_and_follow_evidence():
    alpha, beta = _pairwise_posterior({(0, 1): (0.6, 2_000), (0, 2): (0.65, 2_000), (1, 2): (0.5, 2_000)}, 3)
    metrics = monte_carlo.posterior_field_metrics(
        ["a", "b", "c"], np.full(3, 1 / 3), alpha, beta, match_format="BO1", seed=3,
    )
    assert sum(m["best_pick_probability"] for m in metrics.values()) == pytest.approx(1.0)
    assert metrics["a"]["best_pick_probability"] > 0.99
    assert metrics["a"]["expected_win_rate"] == pytest.approx((0.5 + 0.6 + 0.65) / 3, abs=0.005)
    assert metrics["a"]["expected_win_rate_lower"] < metrics["a"]["expected_win_rate"] < metrics["a"]["expected_win_rate_upper"]


@pytest.mark.unit
def test_field_posterior_applies_best_of_three_transform():
    alpha, beta = _pairwise_posterior({(0, 1): (0.6, 1e7)}, 2)
    metrics = monte_carlo.posterior_field_metrics(["a", "b"], np.array([0.0, 1.0]), alpha, beta, match_format="BO3")
    assert metrics["a"]["expected_win_rate"] == pytest.approx(3 * 0.6 ** 2 - 2 * 0.6 ** 3, abs=1e-3)


@pytest.mark.unit
def test_field_posterior_is_reproducible_and_collapses_with_evidence():
    alpha, beta = _pairwise_posterior({(0, 1): (0.55, 1e7), (0, 2): (0.5, 1e7), (1, 2): (0.45, 1e7)}, 3)
    meta = np.array([0.5, 0.3, 0.2])
    first = monte_carlo.posterior_field_metrics(["a", "b", "c"], meta, alpha, beta, seed=11)
    second = monte_carlo.posterior_field_metrics(["a", "b", "c"], meta, alpha, beta, seed=11)
    assert first == second
    assert all(m["expected_win_rate_upper"] - m["expected_win_rate_lower"] < 1e-3 for m in first.values())
@pytest.mark.integration
def test_chunked_run_simulates_the_same_tournaments_as_one_engine_call():
    decks = ["a", "b", "c", "d"]
    matrix = np.array([
        [0.5, 0.6, 0.45, 0.55],
        [0.4, 0.5, 0.65, 0.5],
        [0.55, 0.35, 0.5, 0.6],
        [0.45, 0.5, 0.4, 0.5],
    ])
    meta = [0.25, 0.25, 0.25, 0.25]
    result = monte_carlo.run_monte_carlo_analytics(
        deck_names=decks, win_matrix=matrix, meta_distribution=dict(zip(decks, meta)),
        d1_rounds=5, cut_points=99, d2_rounds=0, top_cut=8, players=32,
        iterations=25_000, match_format="BO1", seed=1312,
    )
    initial, _, _, champions = monte_carlo.tcg_engine.run_parallel_monte_carlo(
        25_000, 32, meta, matrix.tolist(), 5, 99, 0, 8, 1312, True, monte_carlo.GLOBAL_TIE_RATE, False,
    )

    assert [result["metrics"][deck]["win_probability"] for deck in decks] == pytest.approx(
        [champion / entrants for champion, entrants in zip(champions, initial)], abs=1e-12,
    )
