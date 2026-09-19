import numpy as np
import pytest

from src.core.types import SimulationConfig
from src.evolution.engine import find_evolutionary_stable_state


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