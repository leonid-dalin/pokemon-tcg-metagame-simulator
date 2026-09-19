import numpy as np
import pytest

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
