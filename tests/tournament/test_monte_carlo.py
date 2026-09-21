import numpy as np
import pytest
import json
from pathlib import Path

from src.tournament import monte_carlo


@pytest.mark.unit
def test_chunk_seeds_come_from_the_supplied_seed(monkeypatch):
    seeds = []

    monkeypatch.setattr(monte_carlo.tcg_engine, "initialize_rayon", lambda cores: None)

    def run_parallel(iterations, players, meta, matrix, d1, cut, d2, top, seed, *args):
        seeds.append(seed)
        return [1], [1], [1], [1]

    monkeypatch.setattr(monte_carlo.tcg_engine, "run_parallel_monte_carlo", run_parallel)
    monkeypatch.setattr(monte_carlo.time, "sleep", lambda _: None)

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

    assert seeds == [1234, 1235, 1234, 1235]


@pytest.mark.unit
def test_empty_meta_distribution_uses_a_uniform_field(monkeypatch):
    distributions = []

    monkeypatch.setattr(monte_carlo.tcg_engine, "initialize_rayon", lambda cores: None)

    def run_parallel(iterations, players, meta, matrix, d1, cut, d2, top, seed, *args):
        distributions.append(meta)
        return [1, 1], [1, 1], [1, 1], [1, 1]

    monkeypatch.setattr(monte_carlo.tcg_engine, "run_parallel_monte_carlo", run_parallel)
    monkeypatch.setattr(monte_carlo.time, "sleep", lambda _: None)

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
    assert 0.45 < posterior_mean < 0.6
    assert posterior_mean + reverse_mean == pytest.approx(1.0)


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
def test_posterior_report_contains_intervals_and_ranked_split(monkeypatch):
    monkeypatch.setattr(monte_carlo.tcg_engine, "initialize_rayon", lambda cores: None)
    monkeypatch.setattr(
        monte_carlo.tcg_engine,
        "run_parallel_monte_carlo",
        lambda *args: ([10, 10], [5, 5], [2, 2], [1, 1]),
    )
    monkeypatch.setattr(monte_carlo.time, "sleep", lambda _: None)

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
        iterations=4,
        posterior_draws=2,
        report=True,
    )

    assert set(result) == {"metrics", "ranked_metrics", "insufficient_data", "matchup_panel", "best60_recommendations", "h1_report"}
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
def test_report_true_exposes_matchup_panel(monkeypatch):
    monkeypatch.setattr(monte_carlo.tcg_engine, "initialize_rayon", lambda cores: None)
    monkeypatch.setattr(
        monte_carlo.tcg_engine,
        "run_parallel_monte_carlo",
        lambda *args: ([10, 10], [5, 5], [2, 2], [1, 1]),
    )
    monkeypatch.setattr(monte_carlo.time, "sleep", lambda _: None)
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
        report=True,
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
    monkeypatch.setattr(monte_carlo.time, "sleep", lambda _: None)

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
    monkeypatch.setattr(monte_carlo.time, "sleep", lambda _: None)

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
