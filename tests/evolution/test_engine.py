import numpy as np
import pytest

from src.core.types import SimulationConfig
from src.evolution.engine import find_evolutionary_stable_state


@pytest.mark.unit
def test_tournament_evolution_samples_configured_field_and_returns_fitness(monkeypatch):
    from src.evolution import engine

    class FakeRng:
        def __init__(self):
            self.choice_calls = []

        def choice(self, n, size, p):
            self.choice_calls.append((n, size, p.copy()))
            return np.array([0, 1, 0, 1], dtype=int)

        def integers(self, _upper_bound):
            return 1234

        def normal(self, *args, **kwargs):
            raise AssertionError("noise should be disabled")

        def random(self, *args, **kwargs):
            raise AssertionError("mutation should not be used")

    fake_rng = FakeRng()
    monkeypatch.setattr(engine.np.random, "default_rng", lambda _seed: fake_rng)
    seen_fields = []

    def fake_worker(args):
        field_indices, worker_config, _win_matrix, _matchup_details, _seed = args
        seen_fields.append((field_indices.copy(), worker_config["deck_names"]))
        n_decks = len(worker_config["deck_names"])
        return np.ones(n_decks), np.full(n_decks, 2.0)

    monkeypatch.setattr(engine, "_pure_swiss_worker", fake_worker)
    config = SimulationConfig(
        mode="tournament",
        max_generations=1,
        extinction_threshold=0.0,
        stability_threshold=-1.0,
        convergence_window=1,
        max_inactive_generations=1,
        use_bayesian_winrates=False,
        tournament_size=4,
        num_tournaments_per_gen=1,
        num_rounds=1,
        use_multiproc=False,
        seed=1234,
        mutation_rate=0.0,
        noise_scale=0.0,
        selection_pressure=1.0,
    )

    results, history, _ = find_evolutionary_stable_state(
        ["a", "b"],
        np.array([[0.5, 0.6], [0.4, 0.5]]),
        {},
        config,
    )

    assert fake_rng.choice_calls[0][1] == config.tournament_size
    assert fake_rng.choice_calls[0][2].tolist() == [0.5, 0.5]
    assert seen_fields[0][0].tolist() == [0, 1, 0, 1]
    assert seen_fields[0][1] == ["a", "b"]
    frequencies = np.array([result["frequency"] for result in results])
    assert np.isfinite(frequencies).all()
    assert frequencies.sum() == pytest.approx(1.0)
    assert len(history) >= 2


@pytest.mark.unit
def test_evolution_uses_the_configured_seed_for_global_random_steps(monkeypatch):
    def unexpected_global_random(*args, **kwargs):
        raise AssertionError("global NumPy randomness was used")

    monkeypatch.setattr(np.random, "normal", unexpected_global_random)
    monkeypatch.setattr(np.random, "random", unexpected_global_random)
    config = SimulationConfig(
        mode="replicator",
        max_generations=3,
        extinction_threshold=0.4,
        stability_threshold=-1.0,
        convergence_window=1,
        max_inactive_generations=1,
        use_bayesian_winrates=False,
        tournament_size=4,
        num_tournaments_per_gen=1,
        num_rounds=1,
        use_multiproc=False,
        seed=1234,
        mutation_rate=0.5,
        noise_scale=0.1,
        selection_pressure=10.0,
    )

    find_evolutionary_stable_state(
        ["a", "b"],
        np.array([[0.5, 1.0], [0.0, 0.5]]),
        {},
        config,
    )