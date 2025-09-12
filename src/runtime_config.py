#!/usr/bin/env python3
# runtime_config.py
from typing import NamedTuple, Literal
from .config import *

class RuntimeConfig(NamedTuple):
    # Core
    mode: Literal['replicator', 'tournament']
    max_generations: int
    min_games: int
    extinction_threshold: float
    stability_threshold: float
    convergence_window: int
    max_inactive_generations: int
    rng_seed: int

    # Derived (computed from above)
    min_generations: int
    soft_convergence_gen: int

    # Tournament
    use_bayesian_winrates: bool
    tournament_size: int
    num_tournaments_per_gen: int
    num_rounds: int
    use_multiproc: bool

    # Simulation Enhancements
    dynamic_deck_intro_prob: float
    mutation_floor: float
    noise_scale: float
    selection_pressure: float

    @classmethod
    def from_args(cls, args):
        # Compute derived values DYNAMICALLY based on CLI/default args
        min_generations = max(1, int(args.gens * MIN_GENERATIONS_PROP))  
        soft_convergence_gen = max(1, int(args.gens * SOFT_CONVERGENCE_PROP))

        return cls(
            # Core — pulled from args
            mode=args.mode,
            max_generations=args.gens,
            min_games=args.min_games,
            extinction_threshold=args.extinction_threshold,
            stability_threshold=STABILITY_THRESHOLD,
            convergence_window=CONVERGENCE_WINDOW,
            max_inactive_generations=MAX_INACTIVE_GENERATIONS,
            rng_seed=args.seed,

            # Derived — computed from args
            min_generations=min_generations,
            soft_convergence_gen=soft_convergence_gen,

            # Tournament — from defaults (or could be CLI args later)
            use_bayesian_winrates=USE_BAYESIAN_WINRATES,
            tournament_size=TOURNAMENT_SIZE,
            num_tournaments_per_gen=NUM_TOURNAMENTS_PER_GEN,
            num_rounds=NUM_ROUNDS,
            use_multiproc=USE_MULTIPROC,

            # Simulation Enhancements
            dynamic_deck_intro_prob=args.intro_prob,
            mutation_floor=MUTATION_FLOOR,
            noise_scale=args.noise,
            selection_pressure=SELECTION_PRESSURE
        )