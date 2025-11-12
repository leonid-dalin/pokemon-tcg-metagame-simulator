# config.py
from typing import Literal

# ----------------------------
# Core Simulation Defaults
# ----------------------------
SIMULATION_MODE: Literal["replicator", "tournament"] = "replicator"
MAX_GENERATIONS = 1000
MIN_GENERATIONS_PROP = 0.2
MIN_GAMES = 400
EXTINCTION_THRESHOLD = 1e-10 # 0.005
STABILITY_THRESHOLD = 1e-8 # 0.01
CONVERGENCE_WINDOW = 100
RNG_SEED = 1312

# ----------------------------
# Tournament Defaults
# ----------------------------
USE_BAYESIAN_WINRATES = True
TOURNAMENT_SIZE = 32
NUM_TOURNAMENTS_PER_GEN = 16
NUM_ROUNDS = 5
USE_MULTIPROC = True

# ----------------------------
# Simulation Enhancements
# ----------------------------
DYNAMIC_DECK_INTRO_PROB = 0 # 1e-4
MUTATION_FLOOR = 0 # 1e-4
MAX_INACTIVE_GENERATIONS = 1_000_000
NOISE_SCALE = 0 # 1e-4
SELECTION_PRESSURE = 6

# ----------------------------
# I/O Defaults
# ----------------------------
CONSISTENCY_MEAN_EPSILON = 1e-6
CONSISTENCY_STD_EPSILON = 1e-9
INPUT_DATA = "data/input/ea_input.json"
OUTPUT_DIR = "output/"
MATCHUP_DIR = "data/matchups/"
INPUT_DIR = "data/input/"
