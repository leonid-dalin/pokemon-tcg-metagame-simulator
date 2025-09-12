# ARCHITECTURE.md

This document provides a comprehensive overview of the project's architecture for fellow developers. It details the purpose, structure, and key components (functions, variables, classes) of each module to ensure a clear understanding of the system's scope and design.

The project simulates the long-term evolutionary dynamics of a competitive Pokémon TCG metagame using evolutionary game theory (Replicator Dynamics) and agent-based tournament simulations. It predicts the stable equilibrium state (Evolutionary Stable State) and provides advanced analytics and visualizations.

---

## Core Modules

### `main.py`

**Purpose:** The entry point. Parses command-line arguments, orchestrates the simulation, analysis, and plotting pipeline. Manages unique output directory creation and experiment metadata.

**Key Functions/Classes:**

*   `parse_args() -> Args`: Parses and validates CLI arguments using `argparse`.
*   `RuntimeConfig`: A `NamedTuple` class (defined in `runtime_config.py`) that holds the complete, validated configuration for a simulation run, including derived values like `min_generations` and `soft_convergence_gen`.
*   `run_single_experiment(args: Args, config_override: Optional[Dict]) -> Dict`: The primary orchestration function for a single run. It:
    *   Parses arguments and builds a `RuntimeConfig`.
    *   Sets up logging to a unique, timestamped output directory.
    *   Loads matchup data using `data.load_matchup_data`.
    *   Constructs a `SimulationConfig` object.
    *   Runs the simulation using `simulation.find_evolutionary_stable_state`, passing the config and a path for incremental history logging.
    *   Performs post-simulation analysis using functions from `analysis.py`.
    *   Generates plots using functions from `plotting.py`.
    *   Renames the output directory based on the convergence outcome for easy identification.
    *   Returns metadata for batch runs.
*   `run_batch_experiments(args: Args)`: Manages running multiple experiments with different configurations.
*   `main()`: The top-level entry point that delegates to single or batch runners.

### `data.py`

**Purpose:** Handles loading, cleaning, validating, and preprocessing the metagame matchup data. Also provides utilities for clustering and diagnostics.

**Key Functions:**

*   `safe_normalize(vec: np.ndarray) -> np.ndarray`: Normalizes a vector to sum to 1.0, returning a uniform distribution if the sum is zero.
*   `load_matchup_data(file_path: str, min_matches_required: int, symmetry_tolerance: float) -> Tuple[List[str], np.ndarray, Dict]`: Loads matchup data from a JSON file, filters decks by match volume, and returns reliable deck names, a **non-symmetric** win-rate matrix, and raw matchup details. (Symmetrization has been removed).
*   `cluster_decks_by_matchup_profile(win_matrix: np.ndarray, deck_names: List[str], n_clusters: int, method: Literal["kmeans", "hierarchical"]) -> Dict[str, Any]`: Groups decks into clusters based on the similarity of their matchup vectors using K-Means or Hierarchical clustering.
*   `compute_deck_dominance(win_matrix: np.ndarray, deck_names: List[str]) -> np.ndarray`: Computes and logs the most dominant deck based on its expected win rate against an even field.

### `simulation.py`

**Purpose:** Contains the core engine for simulating metagame evolution over generations.

**Key Functions:**

*   `_tournament_worker(args: Tuple) -> Tuple[np.ndarray, np.ndarray]`: A helper function for multiprocessing. Simulates a single Swiss-style tournament and returns aggregated wins and matches per deck.
*   `run_tournament_generation(current_freq: np.ndarray, ...) -> np.ndarray`: Runs one generation of stochastic tournaments (using `_tournament_worker` optionally in parallel) and returns the new metagame frequency vector. Uses `selection_pressure` from its config dict.
*   `update_replicator_dynamics(current_freq: np.ndarray, win_matrix: np.ndarray, noise_scale: float) -> np.ndarray`: Implements the Replicator Dynamics equation with optional Gaussian noise.
*   `reintroduce_extinct_decks(current_freq: np.ndarray, ...) -> np.ndarray`: Reintroduces extinct decks with a small probability and ensures a mutation floor for all decks.
*   `find_evolutionary_stable_state(deck_names: List[str], win_matrix: np.ndarray, matchup_details: Dict, config: SimulationConfig, history_file_path: Optional[str]) -> Tuple[List[Dict], List[np.ndarray], List[Optional[int]]]`: The main simulation function. It runs the metagame evolution for a specified number of generations, handling deck extinction, reintroduction, convergence checking. It enters a "Soft Convergence" phase to refine the equilibrium. **Accepts a single `SimulationConfig` object for all parameters.** Can write history incrementally to a file via `history_file_path` to manage memory. Returns the final results, a buffer of the recent history, and extinction generation data.
    *   *Removed:* `adaptive_history_interval` function. History is now recorded every generation.

### `analysis.py`

**Purpose:** Performs post-simulation analysis to generate insights like tier lists, convergence metrics, and cycle detection.

**Key Functions:**

*   `compute_convergence_metrics(history: List[np.ndarray], stability_threshold: float) -> Dict[str, Any]`: Quantifies how quickly and stably the metagame converged by analyzing the history of frequency changes.
*   `generate_final_state_tier_list(deck_names: List[str], metagame_history: List[np.ndarray], win_matrix: np.ndarray, ...) -> Dict[str, List[Dict]]`: Generates a tier list (S, A, B, C, D) based on the final state of the metagame, prioritizing meta-weighted win rate and presence. **Now logs the full contents of S, A, and D tiers.**
*   `generate_all_time_tier_list(deck_names: List[str], metagame_history: List[np.ndarray], win_matrix: np.ndarray) -> Dict[str, List[Dict]]`: Generates a tier list based on a deck's overall performance, consistency, and impact across the entire simulation history. **Now logs the full contents of S, A, and D tiers.**
*   `compute_matchup_cycles(win_matrix: np.ndarray, deck_names: List[str], cycle_length: int) -> List[List[str]]`: Identifies unique Rock-Paper-Scissors (RPS) cycles in the matchup graph (currently only 3-cycles) using an efficient combination-based approach.
*   `compute_deck_similarity(win_matrix: np.ndarray, deck_names: List[str], extinction_gens: Optional[List[Optional[int]]], final_active_mask: Optional[List[bool]]) -> np.ndarray`: Computes pairwise cosine similarity between decks based on their matchup profiles. Optionally filters out extinct decks (using `final_active_mask`) and performs K-Means clustering on the active subset.

### `plotting.py`

**Purpose:** Generates interactive visualizations using Plotly for the simulation results.

**Key Functions:**

*   `plot_metagame_evolution_interactive(history: List[np.ndarray], deck_names: List[str], ...) -> Optional[go.Figure]`: Creates an interactive line plot showing the metagame share of top decks over time, with extinction markers. **Now formats hover values as percentages (e.g., "54.27%").**
*   `plot_matchup_heatmap_interactive(win_matrix: np.ndarray, deck_names: List[str], ...) -> Optional[go.Figure]`: Creates an interactive heatmap of the win-rate matrix, optionally sorted by tier.
*   `plot_matchup_network(win_matrix: np.ndarray, deck_names: List[str], cycles: List[List[str]], ...) -> Optional[go.Figure]`: Creates an interactive network graph visualizing significant win-rate edges and highlighting detected RPS cycles.

### `scraper.py`

**Purpose:** A utility script to scrape matchup data from HTML files (e.g., from Limitless TCG) and convert it into the required JSON format (`ea_input.json`).

**Key Functions:**

*   `normalize_archetype(name: str) -> str`: Normalizes archetype names for consistent matching.
*   `extract_deck_info_from_filename(filename: str) -> Tuple[str, str]`: Attempts to extract the deck archetype and format from the HTML filename.
*   `get_deck_archetype(file_path: str, filename: str) -> Tuple[str, str]`: Extracts the deck archetype and format, falling back to parsing the HTML file's title or infobox if filename parsing fails.
*   `scrape_matchup_data(file_path: str, deck_archetype: str, format_name: str, canonical_map: Dict) -> List[Dict]`: Scrapes the matchup table from an HTML file, extracting win rates, match counts, and scores for each opponent.
*   `build_complete_matchup_matrix(all_matchup_data: List[Dict]) -> Dict`: Builds a complete, symmetric matchup matrix from the scraped data, filtering out archetypes with insufficient total matches.
*   `save_to_csv(data: List[Dict], input_path: str)`, `save_matrix_to_csv(matrix_data: Dict, input_path: str)`: Helper functions to save data to CSV files.
*   `main()`: The main entry point for the scraper, which processes all HTML files in the `matchups/` directory.

---

## Configuration Modules

### `config.py`

**Purpose:** Defines the default configuration values used throughout the project.

**Key Variables:**

*   `SIMULATION_MODE`: Default simulation mode (`'replicator'` or `'tournament'`).
*   `MAX_GENERATIONS`, `MIN_GAMES`, `EXTINCTION_THRESHOLD`, `STABILITY_THRESHOLD`, `CONVERGENCE_WINDOW`, `MAX_INACTIVE_GENERATIONS`, `RNG_SEED`: Core simulation defaults.
*   `USE_BAYESIAN_WINRATES`, `TOURNAMENT_SIZE`, `NUM_TOURNAMENTS_PER_GEN`, `NUM_ROUNDS`, `USE_MULTIPROC`: Tournament simulation defaults.
*   `DYNAMIC_DECK_INTRO_PROB`, `MUTATION_FLOOR`, `NOISE_SCALE`, `SELECTION_PRESSURE`: Simulation enhancement defaults.

### `runtime_config.py`

**Purpose:** Defines the `RuntimeConfig` class, which consolidates and computes derived configuration values based on CLI arguments and defaults from `config.py`.

**Key Class:**

*   `RuntimeConfig(NamedTuple)`: A typed, immutable configuration object. Its `from_args` class method takes parsed CLI arguments and returns a `RuntimeConfig` instance, calculating derived values like `min_generations` (20% of `max_generations`) and `soft_convergence_gen` (60% of `max_generations`).

### `simulation_config.py`

**Purpose:** Defines the `SimulationConfig` dataclass, which bundles all parameters required for the core simulation engine into a single, typed object.

**Key Class:**

*   `SimulationConfig(dataclass)`: A mutable configuration object. Its fields include `mode`, `max_generations`, `soft_convergence_gen`, `min_generations`, `extinction_threshold`, `stability_threshold`, `convergence_window`, `max_inactive_generations`, `use_bayesian_winrates`, `tournament_size`, `num_tournaments_per_gen`, `num_rounds`, `use_multiproc`, `seed`, `dynamic_deck_intro_prob`, `mutation_floor`, `noise_scale`, and `selection_pressure`.

### `cli_args.py`

**Purpose:** Defines the command-line interface (CLI) and argument parsing logic.

**Key Functions/Classes:**

*   `Args(NamedTuple)`: A typed structure for holding parsed CLI arguments.
*   `parse_args() -> Args`: Sets up the `argparse.ArgumentParser`, defines all CLI flags with their defaults (often sourced from `config.py`), performs validation (e.g., file existence, value ranges), and returns an `Args` object.

---

## Project Configuration Files

### `pyrightconfig.json`

**Purpose:** Configuration file for the Pyright static type checker.

**Key Settings:**

*   `"typeCheckingMode": "basic"`: Sets the strictness level.
*   `"pythonVersion": "3.12"`: Specifies the target Python version.
*   `"exclude"`: Lists directories and files to ignore during type checking (e.g., `input/`, `output/`, `__pycache__/`).

### `requirements.txt`

**Purpose:** Lists the Python package dependencies required to run the project (e.g., `numpy`, `scipy`, `scikit-learn`, `plotly`, `tqdm`, `beautifulsoup4`).

Last Edited: 12.09.2025, 17:00 UTC+2