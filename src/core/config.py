# config.py | Global constants
from typing import Literal, Dict, Tuple
import os
import multiprocessing

# ----------------------------
# Type Definitions
# ----------------------------
MatchFormat = Literal["BO1", "BO3"]
SimulationMode = Literal["replicator", "tournament"]

# ----------------------------
# I/O Defaults
# ----------------------------
INPUT_DIR = "data/input/"
MATCHUP_DIR = "data/matchups/"
INPUT_FILE = "ea_input.json"
INPUT_DATA = INPUT_DIR + INPUT_FILE
OUTPUT_DIR = "output/"

# ----------------------------
# Global Setup
# ----------------------------
SIMULATION_MODE: SimulationMode = "replicator"
RNG_SEED = 1312
MIN_GAMES = 100

# ----------------------------
# Evolutionary Dynamics (Replicator Engine)
# ----------------------------
MAX_GENERATIONS = 100_000
MAX_INACTIVE_GENERATIONS = 1_000
CONVERGENCE_WINDOW = 50
STABILITY_THRESHOLD = 5e-5
EXTINCTION_THRESHOLD = 1e-6
MUTATION_RATE = 1e-4
NOISE_SCALE = 0.0
SELECTION_PRESSURE = 1
NASH_EQUILIBRIUM = 0.0025

# ----------------------------
# Tournament Agent Engine (Monte Carlo)
# ----------------------------
USE_BAYESIAN_WINRATES = True
USE_MULTIPROC = True
TOURNAMENT_SIZE = 32
NUM_TOURNAMENTS_PER_GEN = 16
NUM_ROUNDS = 5

# TPCi Variant #5 Official Structure Logic 
_STRUCTURE_THRESHOLDS: Tuple[int, ...] = (8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)
# (Day 1 Rounds, Cut, Day 2 Rounds, Top Cut)
_STRUCTURE_RESULTS: Tuple[Tuple[int, int, int, int], ...] = (
    (3, 99, 0, 0),   # <= 8
    (4, 99, 0, 2),   # <= 16
    (6, 99, 0, 4),   # <= 32
    (7, 99, 0, 6),   # <= 64
    (7, 13, 2, 8),   # <= 128
    (8, 16, 2, 8),   # <= 256
    (8, 16, 3, 8),   # <= 512
    (8, 16, 4, 8),   # <= 1024
    (8, 16, 5, 8),   # <= 2048
    (8, 16, 6, 8),   # <= 4096
    (9, 19, 6, 8)    # <= 8192
)

# ----------------------------
# Analytics & Post-Simulation
# ----------------------------
WIN_THRESHOLD = 0.6
CONSISTENCY_MEAN_EPSILON = 1e-6
CONSISTENCY_STD_EPSILON = 1e-9

# Meta Score / Win Rate Tier Thresholds
TIER_0_THRESHOLD = 0.55     # Dominant
TIER_1_THRESHOLD = 0.525    # Top Contender
TIER_2_THRESHOLD = 0.5     # Competitive
TIER_3_THRESHOLD = 0.475      # Niche
TIER_4_THRESHOLD = 0.45     # Struggling
TIER_5_THRESHOLD = 0.0       # Unviable

TIER_THRESHOLDS: Dict[str, float] = {
    "T0": TIER_0_THRESHOLD,
    "T1": TIER_1_THRESHOLD,
    "T2": TIER_2_THRESHOLD,
    "T3": TIER_3_THRESHOLD,
    "T4": TIER_4_THRESHOLD,
    "T5": TIER_5_THRESHOLD
}

# Dynamically derive the order from the dictionary keys
TIER_ORDER: Tuple[str, ...] = tuple(TIER_THRESHOLDS.keys())

# [LEGACY] Composite Scoring Weights 
COMPOSITE_SCORE_WR_WEIGHT = 0.50
COMPOSITE_SCORE_PRESENCE_WEIGHT = 0.30
COMPOSITE_SCORE_CONSISTENCY_WEIGHT = 0.20

# ----------------------------
# Plotting & UI Defaults
# ----------------------------
aggressive_colorscale = [
    [0.0, 'rgb(215, 48, 39)'],
    [0.45, 'rgb(244, 109, 67)'],
    [0.49, 'rgb(253, 219, 199)'],
    [0.51, 'rgb(224, 243, 248)'],
    [0.55, 'rgb(116, 173, 209)'],
    [1.0, 'rgb(69, 117, 180)']
]

# ----------------------------
# Utils
# ----------------------------
def get_container_cores() -> int:
    """
    Safely determines available CPU cores within a Docker cgroup.
    Prevents CPU oversubscription and context-switching thrash in containers.
    """

    if "MAX_CORES" in os.environ:
        return int(os.environ["MAX_CORES"])
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        # macOS/Windows
        return max(1, multiprocessing.cpu_count() // 2)