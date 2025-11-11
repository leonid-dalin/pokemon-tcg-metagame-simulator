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
        "-m",
        "--mode",
        type=str,
        choices=["replicator", "tournament"],
        default=SIMULATION_MODE,
        help="Dynamics simulation mode",
    )
    parser.add_argument(
        "-g",
        "--gens",
        type=int,
        default=MAX_GENERATIONS,
        metavar="N",
        help="Max number of generations to simulate",
    )
    parser.add_argument(
        "-M",
        "--min-games",
        type=int,
        default=MIN_GAMES,
        metavar="N",
        help="Min games required to include a deck in the simulation",
    )
    parser.add_argument(
        "-e",
        "--extinction-threshold",
        type=float,
        default=EXTINCTION_THRESHOLD,
        metavar="F",
        help="Population threshold (0.0–1.0) for deck extinction",
    )
    parser.add_argument(
        "-N",
        "--noise",
        type=float,
        default=NOISE_SCALE,
        metavar="F",
        help="Scale of stochastic noise in replicator dynamics",
    )
    parser.add_argument(
        "-p",
        "--intro-prob",
        type=float,
        default=DYNAMIC_DECK_INTRO_PROB,
        metavar="F",
        help="Probability (0.0–1.0) of introducing a new deck each generation",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=RNG_SEED,
        metavar="INT",
        help="Seed for random number generation",
    )

    # Logging and Execution
    parser.add_argument(
        "-l",
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity level",
    )
    parser.add_argument(
        "-b",
        "--batch",
        action="store_true",
        help="Run batch experiments (requires --batch-config)",
    )
    parser.add_argument(
        "-c",
        "--batch-config",
        type=str,
        metavar="FILE",
        help="Path to batch configuration JSON file",
    )
    parser.add_argument(
        "-P",
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
        type=int,
        default=32,
        help="Expected tournament size (default: 32)",
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
        parser.error("❌ --extinction-threshold must be between 0.0 and 1.0")
    if not (0.0 <= args.intro_prob <= 1.0):
        parser.error("❌ --intro-prob must be between 0.0 and 1.0")
    if args.noise < 0.0:
        parser.error("❌ --noise must be non-negative")

    if args.gens <= 0:
        parser.error("❌ --gens must be a positive integer")

    return Args(**vars(args))