# config.py
from typing import Literal

# ----------------------------
# Core Simulation Defaults
# ----------------------------
SIMULATION_MODE: Literal['replicator', 'tournament'] = 'replicator'
MAX_GENERATIONS = 1000
SOFT_CONVERGENCE_PROP = 0.6
MIN_GENERATIONS_PROP = 0.2
MIN_GAMES = 700
EXTINCTION_THRESHOLD = 0.005
STABILITY_THRESHOLD = 0.01
CONVERGENCE_WINDOW = 100
MAX_INACTIVE_GENERATIONS = 50
RNG_SEED = 1312

# ----------------------------
# Tournament Defaults
# ----------------------------
USE_BAYESIAN_WINRATES = True
TOURNAMENT_SIZE = 32
NUM_TOURNAMENTS_PER_GEN = 16
NUM_ROUNDS = 6
USE_MULTIPROC = True

# ----------------------------
# Simulation Enhancements
# ----------------------------
DYNAMIC_DECK_INTRO_PROB = 1e-4
MUTATION_FLOOR = 1e-4
NOISE_SCALE = 1e-4
SELECTION_PRESSURE = 6

# ----------------------------
# I/O Defaults
# ----------------------------
INPUT_DATA = 'data/input/ea_input.json'
OUTPUT_DIR = 'output/'
MATCHUP_DIR = 'data/matchups/'
INPUT_DIR = 'data/input/'