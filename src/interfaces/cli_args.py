#!/usr/bin/env python3
# cli_args.py | argparse definitions
import argparse
import os
from typing import NamedTuple, Optional
from src.core.config import *


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
    batch_config: Optional[str]
    no_plot: bool
    cluster: bool
    predict: bool
    players: int
    meta: str
    tournament_style: str


def parse_args() -> Args:
    parser = argparse.ArgumentParser(
        description="Pokémon TCG Metagame Simulator & Predictor",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # 📁 Data & I/O
    io_group = parser.add_argument_group("📁 Data & I/O")
    io_group.add_argument(
        "-i", "--input",
        type=str,
        default=INPUT_DATA,
        metavar="FILE",
        help="Path to the JSON matchup matrix file. (default: %(default)s)",
    )
    io_group.add_argument(
        "-o", "--output",
        type=str,
        default=OUTPUT_DIR,
        metavar="DIR",
        help="Directory where simulation logs and interactive plots will be saved. (default: %(default)s)",
    )
    io_group.add_argument(
        "-m", "--min-games",
        type=int,
        default=MIN_GAMES,
        help="Minimum required match volume to include an archetype in the baseline field. (default: %(default)s)",
    )

    # ⚙️ Core Simulation Setup
    sim_group = parser.add_argument_group("⚙️ Core Simulation Setup")
    sim_group.add_argument(
        "-M", "--mode",
        type=str,
        choices=["replicator", "tournament"],
        default=SIMULATION_MODE,
        help="Select the simulation engine. 'replicator' for ESS, 'tournament' for agent brackets. (default: %(default)s)",
    )
    sim_group.add_argument(
        "-g", "--gens",
        type=int,
        default=MAX_GENERATIONS,
        help="Maximum number of generations/epochs to simulate. (default: %(default)s)",
    )
    sim_group.add_argument(
        "-s", "--seed",
        type=int,
        default=RNG_SEED,
        help="Fixed RNG seed for perfectly reproducible experiments. (default: %(default)s)",
    )

    # 🧬 Evolutionary Dynamics
    evo_group = parser.add_argument_group("🧬 Evolutionary Dynamics (Replicator Mode)")
    evo_group.add_argument(
        "-e", "--extinction-threshold",
        type=float,
        default=EXTINCTION_THRESHOLD,
        help="Metagame frequency drop-off point where a deck is considered mathematically dead. (default: %(default)s)",
    )
    evo_group.add_argument(
        "-N", "--noise",
        type=float,
        default=NOISE_SCALE,
        help="Scale of Gaussian noise injected into generation payoffs to simulate pilot error/luck. (default: %(default)s)",
    )
    evo_group.add_argument(
        "-I", "--intro-prob",
        type=float,
        default=DYNAMIC_DECK_INTRO_PROB,
        dest="intro_prob",
        help="Stochastic probability per generation of a rogue/extinct deck re-entering the meta. (default: %(default)s)",
    )
    evo_group.add_argument(
        "-S", "--stability-threshold",
        type=float,
        default=STABILITY_THRESHOLD,
        help="Delta threshold below which the metagame is considered to have reached Nash Equilibrium. (default: %(default)s)",
    )
    evo_group.add_argument(
        "-W", "--convergence-window",
        type=int,
        default=CONVERGENCE_WINDOW,
        dest="convergence_window",
        help="Consecutive generations required below the stability threshold to halt simulation. (default: %(default)s)",
    )
    evo_group.add_argument(
        "-X", "--max-inactive-gens",
        type=int,
        default=MAX_INACTIVE_GENERATIONS,
        dest="max_inactive_generations",
        help="Generations a deck must remain extinct before being permanently culled from memory. (default: %(default)s)",
    )
    evo_group.add_argument(
        "--mutation-floor",
        type=float,
        default=MUTATION_FLOOR,
        help="Absolute frequency floor applied when a deck is randomly reintroduced. (default: %(default)s)",
    )
    evo_group.add_argument(
        "--selection-pressure",
        type=float,
        default=SELECTION_PRESSURE,
        help="Exponent weight scaling how aggressively player populations migrate to winning decks. (default: %(default)s)",
    )

    # 🎲 Tournament Agent Engine
    trn_group = parser.add_argument_group("🎲 Tournament Agent Engine (Tournament Mode)")
    trn_group.add_argument(
        "--tournament-style",
        type=str,
        choices=["pure_swiss", "championship_series"],
        default="pure_swiss",
        help="Bracket execution logic: 'pure_swiss' (fast) or 'championship_series' (BO3 Tie Convergence). (default: %(default)s)",
    )
    trn_group.add_argument(
        "-T", "--tournament-size",
        type=int,
        default=TOURNAMENT_SIZE,
        help="Number of agent pilots spawned per tournament iteration. (default: %(default)s)",
    )
    trn_group.add_argument(
        "--tournaments-per-gen",
        type=int,
        default=NUM_TOURNAMENTS_PER_GEN,
        dest="num_tournaments_per_gen",
        help="Total parallel tournaments executed to calculate the average generation payoff. (default: %(default)s)",
    )
    trn_group.add_argument(
        "-r", "--rounds",
        type=int,
        default=NUM_ROUNDS,
        dest="num_rounds",
        help="Number of Swiss rounds played per tournament (Overridden by Championship Series). (default: %(default)s)",
    )
    trn_group.add_argument(
        "-B", "--use-bayesian",
        action="store_true",
        default=USE_BAYESIAN_WINRATES,
        dest="use_bayesian_winrates",
        help="Sample matchup win rates from a Beta distribution derived from match volume confidence.",
    )

    # 🔮 Static Predictor
    prd_group = parser.add_argument_group("🔮 Static Predictor (Prediction Mode)")
    prd_group.add_argument(
        "--predict",
        action="store_true",
        help="Bypass evolutionary simulation and run the static Monte Carlo tournament EV solver.",
    )
    prd_group.add_argument(
        "-P", "--players",
        type=int,
        default=32,
        help="Expected total field size for the upcoming event. (default: %(default)s)",
    )
    prd_group.add_argument(
        "--meta",
        type=str,
        default="",
        help="Comma-separated custom field constraints (e.g., 'Gholdengo:0.10,Joltik Box:0.15').",
    )

    # 💻 System Utility
    sys_group = parser.add_argument_group("💻 System & Execution Utility")
    sys_group.add_argument(
        "-l", "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Set terminal logging verbosity. (default: %(default)s)",
    )
    sys_group.add_argument(
        "-b", "--batch",
        action="store_true",
        help="Enable batch execution mode for automated parameter sweeping.",
    )
    sys_group.add_argument(
        "-c", "--batch-config",
        type=str,
        default=None,
        help="Path to the JSON configuration file defining batch parameters.",
    )
    sys_group.add_argument(
        "--no-plot",
        action="store_true",
        help="Suppress generation of interactive HTML Plotly graphs.",
    )
    sys_group.add_argument(
        "-C", "--cluster",
        action="store_true",
        help="Enable Silhouette-optimized K-Means strategic archetype clustering.",
    )
    sys_group.add_argument(
        "--no-multiproc",
        action="store_false",
        default=USE_MULTIPROC,
        dest="use_multiproc",
        help="Force the tournament engine to execute on a single thread (disables multiprocessing).",
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
        tournament_style=args.tournament_style,
    )