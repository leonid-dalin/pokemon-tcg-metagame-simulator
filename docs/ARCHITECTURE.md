# ARCHITECTURE.md

This document provides a comprehensive overview of the project's architecture for fellow developers. It details the purpose, structure, and key components (functions, variables, classes) of each module to ensure a clear understanding of the system's scope and design.

The project simulates the long-term evolutionary dynamics of a competitive Pokémon TCG metagame using evolutionary game theory (Replicator Dynamics) and agent-based tournament simulations. It predicts the stable equilibrium state (Evolutionary Stable State) and provides advanced analytics, visualizations, and an interactive web application for recommendations.

⚠️ Caution: This project is undergoing active refactoring. Expect frequent changes to the architecture, and verify information with the latest source code. I'll do my best to keep it updated, but you know, I'm only human.

---

## Core Modules

### `main.py`

**Purpose:** The main CLI entry point. Parses command-line arguments, orchestrates the simulation, analysis, and plotting pipeline. Also handles the CLI-based prediction mode. Manages unique output directory creation and experiment metadata.

**Key Functions/Classes:**

* `parse_args() -> Args`: Parses and validates CLI arguments using `argparse` (defined in `cli_args.py`).
* `run_single_experiment(args: Args, config_override: Optional[Dict]) -> Dict`: The primary orchestration function for a single simulation run. It:
    * Sets up logging to a unique, timestamped output directory.
    * Loads matchup data using `data.load_matchup_data`.
    * **Constructs a `SimulationConfig` object *directly* from the parsed `args`.**
    * Runs the simulation using `simulation.find_evolutionary_stable_state`, passing the config and a path for incremental history logging.
    * Performs post-simulation analysis using functions from `analysis.py`.
    * Generates plots using functions from `plotting.py`.
    * Renames the output directory based on the convergence outcome.
    * Returns metadata for batch runs.
* `run_batch_experiments(args: Args)`: Manages running multiple simulation experiments with different configurations.
* `main()`: The top-level entry point that delegates to single/batch runners or the CLI prediction mode.

### `app.py`

**Purpose:** A standalone Streamlit web application, providing a user-friendly GUI for the `predictor` module.

**Key Functions:**

* `get_valid_deck_names() -> List[str]`: Cached function to load deck names for UI selectors.
* `load_full_win_matrix()`: Cached function to load the win matrix for matchup analysis.
* `get_pro_meta() -> np.ndarray`: **(Performance Critical)** A cached function that runs the full `find_evolutionary_stable_state` simulation *once* to get the "Pro" metagame. The result is cached for the lifetime of the app server, making "Pro" mode recommendations instantaneous after the first load.
* `main()`: Renders the Streamlit UI, collects user inputs (tournament size, known meta), and calls `predictor.predict_best_decks` (passing the cached "Pro" meta if selected) to display recommendations, avoidance lists, and performance metrics for the user's chosen deck.

### `predictor.py`

**Purpose:** Contains the core logic for the recommendation engine. It takes a user-defined metagame, calculates a plausible meta, and then computes advanced performance metrics for all decks.

**Key Functions:**

* `swiss_rounds_from_players(n_players: int) -> int`: Calculates the number of Swiss rounds based on `log2(n)`.
* `predict_best_decks(user_meta_spec: UserMetaSpec, ...) -> PredictionResult`: The main prediction function. It:
    * Loads all matchup data.
    * Constructs a plausible metagame vector by blending user-specified "fixed" decks with a "fallback" meta (either "Pro" simulation or "Casual" uniform).
    * Calculates expected win rate (WR), Strength of Schedule (SoS), and Opponent's Match Win % (OMW) for all decks **using vectorized NumPy operations.**
    * Calculates an "undefeated probability" and a final "composite score".
    * Returns a `PredictionResult` dictionary containing recommendations, avoidance lists, and full metrics.

### `data.py`

**Purpose:** Handles loading, cleaning, validating, and preprocessing the metagame matchup data. Also provides utilities for clustering and diagnostics.

**Key Functions:**

* `safe_normalize(vec: np.ndarray) -> np.ndarray`: Normalizes a vector to sum to 1.0, returning a uniform distribution if the sum is zero.
* `load_matchup_data(file_path: str, min_matches_required: int) -> Tuple[List[str], np.ndarray, Dict]`: Loads matchup data from a JSON file, filters decks by match volume, and returns reliable deck names, a **non-symmetric** win-rate matrix, and raw matchup details.
* `cluster_decks_by_matchup_profile(win_matrix: np.ndarray, deck_names: List[str], n_clusters: int, method: Literal["kmeans", "hierarchical"]) -> Dict[str, Any]`: Groups decks into clusters based on the similarity of their matchup vectors using K-Means or Hierarchical clustering.
* `compute_deck_dominance(win_matrix: np.ndarray, deck_names: List[str]) -> np.ndarray`: Computes and logs the most dominant deck based on its expected win rate against an even field.

### `simulation.py`

**Purpose:** Contains the core engine for simulating metagame evolution over generations.

**Key Functions:**

* `_tournament_worker(args: Tuple) -> Tuple[np.ndarray, np.ndarray]`: A helper function for multiprocessing. Simulates a single Swiss-style tournament and returns aggregated wins and matches per deck.
* `run_tournament_generation(current_freq: np.ndarray, ...) -> np.ndarray`: Runs one generation of stochastic tournaments (using `_tournament_worker` optionally in parallel) and returns the new metagame frequency vector. Uses `selection_pressure` from its config dict.
* `update_replicator_dynamics(current_freq: np.ndarray, win_matrix: np.ndarray, rng: np.random.Generator, noise_scale: float) -> np.ndarray`: Implements the Replicator Dynamics equation with optional Gaussian noise, using a passed-in RNG for reproducibility.
* `reintroduce_extinct_decks(current_freq: np.ndarray, ...) -> np.ndarray`: Reintroduces extinct decks with a small probability and ensures a mutation floor for all decks.
* `find_evolutionary_stable_state(deck_names: List[str], win_matrix: np.ndarray, matchup_details: Dict, config: SimulationConfig, history_file_path: Optional[str]) -> Tuple[List[Dict], List[np.ndarray], List[Optional[int]]]`: The main simulation function. It runs the metagame evolution for a specified number of generations, handling deck extinction, reintroduction, and convergence checking. **Accepts a single `SimulationConfig` object for all parameters.** Can write history incrementally to a file via `history_file_path` to manage memory. Returns the final results, a buffer of the recent history, and extinction generation data.

### `analysis.py`

**Purpose:** Performs post-simulation analysis to generate insights like tier lists, convergence metrics, and cycle detection.

**Key Functions:**

* `compute_convergence_metrics(history: List[np.ndarray], stability_threshold: float) -> Dict[str, Any]`: Quantifies how quickly and stably the metagame converged by analyzing the history of frequency changes.
* `generate_final_state_tier_list(deck_names: List[str], metagame_history: List[np.ndarray], win_matrix: np.ndarray, ...) -> Dict[str, List[Dict]]`: Generates a tier list (S, A, B, C, D) based on the final state of the metagame, prioritizing meta-weighted win rate and presence.
* `generate_all_time_tier_list(deck_names: List[str], metagame_history: List[np.ndarray], win_matrix: np.ndarray) -> Dict[str, List[Dict]]`: Generates a tier list based on a deck's overall performance, consistency, and impact across the entire simulation history. **This function is highly optimized using vectorized NumPy operations.**
* `compute_matchup_cycles(win_matrix: np.ndarray, deck_names: List[str], cycle_length: int) -> List[List[str]]`: Identifies unique Rock-Paper-Scissors (RPS) cycles in the matchup graph.
* `compute_deck_similarity(win_matrix: np.ndarray, deck_names: List[str], final_active_mask: Optional[List[bool]]) -> np.ndarray`: Computes pairwise cosine similarity between decks based on their matchup profiles. Optionally filters out extinct decks and performs K-Means clustering on the active subset.

### `plotting.py`

**Purpose:** Generates interactive visualizations using Plotly for the simulation results.

**Key Functions:**

* `plot_metagame_evolution_interactive(history: List[np.ndarray], deck_names: List[str], ...) -> Optional[go.Figure]`: Creates an interactive line plot showing the metagame share of top decks over time.
* `plot_matchup_heatmap_interactive(win_matrix: np.ndarray, deck_names: List[str], ...) -> Optional[go.Figure]`: Creates an interactive heatmap of the win-rate matrix, optionally sorted by tier.
* `plot_matchup_network(win_matrix: np.ndarray, deck_names: List[str], cycles: List[List[str]], ...) -> Optional[go.Figure]`: Creates an interactive network graph visualizing significant win-rate edges and highlighting detected RPS cycles.

### `scraper.py`

**Purpose:** A utility script to scrape matchup data from HTML files (e.g., from Limitless TCG) and convert it into the required JSON format (`ea_input.json`).

**Key Functions:**

* `normalize_archetype(name: str) -> str`: Normalizes archetype names for consistent matching.
* `scrape_matchup_data(file_path: str, ..._ -> List[Dict]`: Scrapes the matchup table from an HTML file.
* `build_complete_matchup_matrix(all_matchup_data: List[Dict]) -> Dict`: Builds a complete matchup matrix from the scraped data, filtering out archetypes with insufficient total matches.
* `main()`: The main entry point for the scraper.

---

## Configuration Modules

### `config.py`

**Purpose:** Defines the default configuration values used throughout the project.

**Key Variables:**

* `SIMULATION_MODE`: Default simulation mode (`'replicator'` or `'tournament'`).
* `MAX_GENERATIONS`, `MIN_GAMES`, `EXTINCTION_THRESHOLD`, `STABILITY_THRESHOLD`, `CONVERGENCE_WINDOW`, `MAX_INACTIVE_GENERATIONS`, `RNG_SEED`: Core simulation defaults.
* `USE_BAYESIAN_WINRATES`, `TOURNAMENT_SIZE`, `NUM_TOURNAMENTS_PER_GEN`, `NUM_ROUNDS`, `USE_MULTIPROC`: Tournament simulation defaults.
* `DYNAMIC_DECK_INTRO_PROB`, `MUTATION_FLOOR`, `NOISE_SCALE`, `SELECTION_PRESSURE`: Simulation enhancement defaults.

### `simulation_config.py`

**Purpose:** Defines the `SimulationConfig` dataclass, which bundles all parameters required for the core simulation engine into a single, typed object.

**Key Class:**

* `SimulationConfig(dataclass)`: A mutable configuration object. Its fields include `mode`, `max_generations`, `min_generations`, `extinction_threshold`, `stability_threshold`, `convergence_window`, `max_inactive_generations`, `use_bayesian_winrates`, `tournament_size`, `num_tournaments_per_gen`, `num_rounds`, `use_multiproc`, `seed`, `dynamic_deck_intro_prob`, `mutation_floor`, `noise_scale`, and `selection_pressure`.

### `cli_args.py`

**Purpose:** Defines the command-line interface (CLI) and argument parsing logic.

**Key Functions/Classes:**

* `Args(NamedTuple)`: A typed structure for holding parsed CLI arguments. Includes fields for both simulation (`gens`, `noise`, etc.) and prediction (`predict`, `players`, `meta`).
* `parse_args() -> Args`: Sets up the `argparse.ArgumentParser`, defines all CLI flags with their defaults (sourced from `config.py`), performs validation (e.g., file existence, value ranges), and returns an `Args` object.

---

## Project Configuration Files

### `pyrightconfig.json`

**Purpose:** Configuration file for the Pyright static type checker.

**Key Settings:**

* `"typeCheckingMode": "basic"`: Sets the strictness level.
* `"pythonVersion": "3.12"`: Specifies the target Python version.
* `"exclude"`: Lists directories and files to ignore during type checking (e.g., `input/`, `output/`, `__pycache__/`).

### `requirements.txt`

**Purpose:** Lists the Python package dependencies required to run the project (e.g., `numpy`, `scipy`, `scikit-learn`, `plotly`, `tqdm`, `beautifulsoup4`, `streamlit`).

Last Edited: 11.11.2025, 22:00 UTC+2