# config.py | Global constants
from typing import Literal, Dict, Tuple

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
INPUT_DATA = "data/input/ea_input.json"
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
MAX_GENERATIONS = 1000
MIN_GENERATIONS_PROP = 0.2
STABILITY_THRESHOLD = 1e-8       # 0.01
CONVERGENCE_WINDOW = 100
EXTINCTION_THRESHOLD = 1e-10     # 0.005
MAX_INACTIVE_GENERATIONS = 1_000_000
DYNAMIC_DECK_INTRO_PROB = 0      # 1e-4
MUTATION_FLOOR = 0               # 1e-4
NOISE_SCALE = 0                  # 1e-4
SELECTION_PRESSURE = 6

# ----------------------------
# Tournament Agent Engine (Monte Carlo)
# ----------------------------
USE_BAYESIAN_WINRATES = True
USE_MULTIPROC = True
TOURNAMENT_SIZE = 32
NUM_TOURNAMENTS_PER_GEN = 16
NUM_ROUNDS = 5

# TPCi Variant #5 Official Structure Logic 
_STRUCTURE_THRESHOLDS: Tuple[int, ...] = (8, 16, 32, 64, 128, 256, 512, 1024, 2048)
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
    (8, 16, 6, 8),   # > 2048 (Default)
)

# ----------------------------
# Analytics & Post-Simulation
# ----------------------------
WIN_THRESHOLD = 0.6
CONSISTENCY_MEAN_EPSILON = 1e-6
CONSISTENCY_STD_EPSILON = 1e-9

# Meta Score / Win Rate Tier Thresholds
TIER_0_THRESHOLD = 0.525     # Dominant
TIER_0_5_THRESHOLD = 0.50    # Top Contender
TIER_1_THRESHOLD = 0.475     # Competitive
TIER_2_THRESHOLD = 0.45      # Niche
TIER_3_THRESHOLD = 0.425     # Struggling
TIER_4_THRESHOLD = 0.0       # Unviable

TIER_THRESHOLDS: Dict[str, float] = {
    "T0": TIER_0_THRESHOLD,
    "T0.5": TIER_0_5_THRESHOLD,
    "T1": TIER_1_THRESHOLD,
    "T2": TIER_2_THRESHOLD,
    "T3": TIER_3_THRESHOLD,
    "T4": TIER_4_THRESHOLD
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
    [0.0, "rgb(178, 34, 34)"],      # Tier E (Firebrick)
    [0.45, "rgb(255, 69, 0)"],      # Tier C (Red-Orange)
    [0.475, "rgb(255, 215, 0)"],    # Tier B (Gold/Yellow)
    [0.50, "rgb(50, 205, 50)"],     # Tier A (Lime Green)
    [0.525, "rgb(0, 100, 0)"],      # Tier S (Dark Green)
    [1.0, "rgb(0, 30, 0)"],         # Tier S+ (Very Dark Green)
]