#!/usr/bin/env python3
# cli_args.py
import argparse
import os
from typing import NamedTuple
from .config import *


class Args(NamedTuple):
    input: str
    output: str
    mode: str
    gens: int
    min_games: int
    extinction_threshold: float
    noise: float
    intro_prob: float
    seed: int
    stability_threshold: float
    convergence_window: int
    max_inactive_generations: int
    use_bayesian_winrates: bool
    tournament_size: int
    num_tournaments_per_gen: int
    num_rounds: int
    use_multiproc: bool
    mutation_floor: float
    selection_pressure: float

    log_level: str
    batch: bool
    batch_config: str | None
    no_plot: bool
    cluster: bool
    predict: bool
    players: int
    meta: str


def parse_args() -> Args:
    parser = argparse.ArgumentParser(
        description="Pokémon TCG Metagame Evolution Simulator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data and Output
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default=INPUT_DATA,  # 'data/input/ea_input.json'
        metavar="FILE",
        help="Path to matchup data JSON file",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=OUTPUT_DIR,  # 'output/'
        metavar="DIR",
        help="Output directory for results",
    )

    # Core Simulation Parameters
    parser.add_argument(
        "-M",
        "--mode",
        type=str,
        choices=["replicator", "tournament"],
        default=SIMULATION_MODE,
        help="Simulation mode (default: %(default)s)",
    )
    parser.add_argument(
        "-g",
        "--gens",
        type=int,
        default=MAX_GENERATIONS,
        help="Maximum number of generations to run (default: %(default)s)",
    )
    parser.add_argument(
        "--min-games",
        "-min",
        type=int,
        default=MIN_GAMES,
        help="Minimum required game count for matchup data (default: %(default)s)",
    )
    parser.add_argument(
        "--extinction-threshold",
        "-E",
        type=float,
        default=EXTINCTION_THRESHOLD,
        help="Deck share below which it is considered extinct (default: %(default)s)",
    )
    parser.add_argument(
        "--noise",
        "-N",
        type=float,
        default=NOISE_SCALE,
        help="Scale of random noise added to payoffs each generation (default: %(default)s)",
    )
    parser.add_argument(
        "--intro-prob",
        "-intro",
        type=float,
        default=DYNAMIC_DECK_INTRO_PROB,
        help="Probability of re-introducing an extinct deck (default: %(default)s)",
    )
    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=RNG_SEED,
        help="Random seed for reproducibility (default: %(default)s)",
    )

    # Stability Parameters
    parser.add_argument(
        "--stability-threshold",
        "-S",
        type=float,
        default=STABILITY_THRESHOLD,
        help="Threshold for metagame stability (default: %(default)s)",
    )
    parser.add_argument(
        "--conv-window",
        "-convergence",
        "-conv",
        type=int,
        default=CONVERGENCE_WINDOW,
        dest="convergence_window",
        help="Generations window for stability check (default: %(default)s)",
    )
    parser.add_argument(
        "--max-inactive-gens",
        "-inactive",
        type=int,
        default=MAX_INACTIVE_GENERATIONS,
        dest="max_inactive_generations",
        help="Generations to stop if no changes detected (default: %(default)s)",
    )

    # Tournament Parameters
    parser.add_argument(
        "--use-bayesian",
        "-B",
        action="store_true",
        default=USE_BAYESIAN_WINRATES,
        dest="use_bayesian_winrates",
        help="Use Bayesian-adjusted win rates for tournament mode (default: %(default)s)",
    )
    parser.add_argument(
        "--t-size",
        "-size",
        type=int,
        default=TOURNAMENT_SIZE,
        dest="tournament_size",
        help="Tournament size per generation (default: %(default)s)",
    )
    parser.add_argument(
        "--t-per-gen",
        "-tpg",
        type=int,
        default=NUM_TOURNAMENTS_PER_GEN,
        dest="num_tournaments_per_gen",
        help="Number of tournaments per generation (default: %(default)s)",
    )
    parser.add_argument(
        "--rounds",
        "-r",
        type=int,
        default=NUM_ROUNDS,
        dest="num_rounds",
        help="Rounds per tournament (default: %(default)s)",
    )
    parser.add_argument(
        "--no-multiproc",
        action="store_false",
        default=USE_MULTIPROC,
        dest="use_multiproc",
        help="Disable multiprocessing for tournament mode",
    )

    # Enhancement Parameters
    parser.add_argument(
        "--mutation-floor",
        "-MF",
        type=float,
        default=MUTATION_FLOOR,
        help="Minimum mutation rate (default: %(default)s)",
    )
    parser.add_argument(
        "--selection-pressure",
        "-SP",
        type=float,
        default=SELECTION_PRESSURE,
        help="Exponent for selection pressure (default: %(default)s)",
    )

    # Utility
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Set the logging verbosity (default: %(default)s)",
    )
    parser.add_argument(
        "-b",
        "--batch",
        action="store_true",
        help="Run simulation in batch mode (for parameter sweep)",
    )
    parser.add_argument(
        "--batch-config",
        type=str,
        default=None,
        help="Path to JSON file with batch parameters",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable plotting of simulation results",
    )
    parser.add_argument(
        "-C",
        "--cluster",
        action="store_true",
        help="Enable post-simulation deck clustering analysis",
    )

    parser.add_argument(
        "--predict",
        action="store_true",
        help="Run metagame prediction instead of simulation",
    )
    parser.add_argument(
        "--players",
        "-p",
        type=int,
        default=32,
        help="Expected total players in the tournament (used for --predict mode only) (default: %(default)s)",
    )
    parser.add_argument(
        "--meta",
        type=str,
        default="",
        help="User-specified meta for prediction (e.g., 'DeckA:0.2,DeckB:0.1')"
    )

    args = parser.parse_args()

    # === VALIDATION ===
    if not os.path.isfile(args.input):
        parser.error(f"❌ Input file not found: {args.input}")

    os.makedirs(args.output, exist_ok=True)

    if args.batch and not args.batch_config:
        parser.error("❌ --batch mode requires --batch-config to be specified!")

    if not (0.0 <= args.extinction_threshold <= 1.0):
        parser.error("❌ --extinction-threshold must be between 0.0 and 1.0.")
    if args.noise < 0.0:
        parser.error("❌ --noise scale cannot be negative.")

    # Convert args to the NamedTuple for immutability
    return Args(
        input=args.input,
        output=args.output,
        mode=args.mode,
        gens=args.gens,
        min_games=args.min_games,
        extinction_threshold=args.extinction_threshold,
        noise=args.noise,
        intro_prob=args.intro_prob,
        seed=args.seed,
        log_level=args.log_level,
        batch=args.batch,
        batch_config=args.batch_config,
        no_plot=args.no_plot,
        cluster=args.cluster,
        predict=args.predict,
        players=args.players,
        meta=args.meta,
        stability_threshold=args.stability_threshold,
        convergence_window=args.convergence_window,
        max_inactive_generations=args.max_inactive_generations,
        use_bayesian_winrates=args.use_bayesian_winrates,
        tournament_size=args.tournament_size,
        num_tournaments_per_gen=args.num_tournaments_per_gen,
        num_rounds=args.num_rounds,
        use_multiproc=args.use_multiproc,
        mutation_floor=args.mutation_floor,
        selection_pressure=args.selection_pressure,
    )