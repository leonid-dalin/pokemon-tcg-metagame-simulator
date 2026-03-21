# config.py | Global constants
from typing import Literal

# ----------------------------
# Core Simulation Defaults
# ----------------------------
SIMULATION_MODE: Literal["replicator", "tournament"] = "replicator"
MAX_GENERATIONS = 1000
MIN_GENERATIONS_PROP = 0.2
MIN_GAMES = 100
EXTINCTION_THRESHOLD = 1e-10 # 0.005
STABILITY_THRESHOLD = 1e-8 # 0.01
CONVERGENCE_WINDOW = 100
RNG_SEED = 1312
MatchFormat = Literal["BO1", "BO3"]

# ----------------------------
# Tournament Defaults
# ----------------------------
USE_BAYESIAN_WINRATES = True
TOURNAMENT_SIZE = 32
NUM_TOURNAMENTS_PER_GEN = 16
NUM_ROUNDS = 5
USE_MULTIPROC = True
_STRUCTURE_THRESHOLDS = (8, 16, 32, 64, 128, 256, 512, 1024, 2048)
_STRUCTURE_RESULTS = (
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
aggressive_colorscale = [
    [0.0, "rgb(178, 34, 34)"],  # 0% - Firebrick
    [0.45, "rgb(255, 153, 153)"],  # ~45% - Light red
    [0.49, "rgb(255, 255, 224)"],  # ~49% - Light Yellow
    [0.51, "rgb(255, 255, 224)"],  # ~51% - Light Yellow
    [0.55, "rgb(159, 218, 169)"],  # ~55% - Light Green
    [1.0, "rgb(0, 68, 27)"],  # 100% - Very Dark Green
]

# ----------------------------
# Simulation Enhancements
# ----------------------------
DYNAMIC_DECK_INTRO_PROB = 0 # 1e-4
MUTATION_FLOOR = 0 # 1e-4
MAX_INACTIVE_GENERATIONS = 1_000_000
NOISE_SCALE = 0 # 1e-4
SELECTION_PRESSURE = 6

# ----------------------------
# Analysis.py Constants
# ----------------------------
TIER_S_THRESHOLD = 0.525
TIER_A_THRESHOLD = 0.50
TIER_B_THRESHOLD = 0.475
TIER_C_THRESHOLD = 0.45
tier_thresholds = {
    "S": TIER_S_THRESHOLD,
    "A": TIER_A_THRESHOLD,
    "B": TIER_B_THRESHOLD,
    "C": TIER_C_THRESHOLD,
}
CONSISTENCY_MEAN_EPSILON = 1e-6
CONSISTENCY_STD_EPSILON = 1e-9
COMPOSITE_SCORE_WR_WEIGHT = 0.50
COMPOSITE_SCORE_PRESENCE_WEIGHT = 0.30
COMPOSITE_SCORE_CONSISTENCY_WEIGHT = 0.20
WIN_THRESHOLD = 0.6

# ----------------------------
# I/O Defaults
# ----------------------------
INPUT_DATA = "data/input/ea_input.json"
OUTPUT_DIR = "output/"
MATCHUP_DIR = "data/matchups/"
INPUT_DIR = "data/input/"
