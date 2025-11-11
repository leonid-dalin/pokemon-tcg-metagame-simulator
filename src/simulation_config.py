# simulation_config.py
from dataclasses import dataclass
from typing import Literal


@dataclass
class SimulationConfig:
    mode: Literal["replicator", "tournament"]
    max_generations: int
    min_generations: int
    extinction_threshold: float
    stability_threshold: float
    convergence_window: int
    max_inactive_generations: int
    use_bayesian_winrates: bool
    tournament_size: int
    num_tournaments_per_gen: int
    num_rounds: int
    use_multiproc: bool
    seed: int
    dynamic_deck_intro_prob: float
    mutation_floor: float
    noise_scale: float
    selection_pressure: float
