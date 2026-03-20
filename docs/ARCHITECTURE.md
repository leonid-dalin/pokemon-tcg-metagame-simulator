# ARCHITECTURE.md

This document provides a comprehensive overview of the project's architecture for fellow developers. It details the purpose, structure, and key components (functions, variables, classes) of each module to ensure a clear understanding of the system's scope and design.

The project simulates the long-term evolutionary dynamics of a competitive Pokémon TCG metagame using evolutionary game theory (Replicator Dynamics) and agent-based tournament simulations. It predicts the stable equilibrium state (Evolutionary Stable State) and provides advanced analytics, visualizations, and an interactive web application for real-time tournament equity predictions.

⚠️ Caution: This project is undergoing active refactoring. Expect frequent changes to the architecture, and verify information with the latest source code. I'll do my best to keep it updated, but you know, I'm only human.

---

## Directory Structure

The codebase utilizes a domain-driven layout to strictly separate long-term evolutionary mechanics from immediate tournament equity evaluation:

```text
pokemon-tcg-metagame-simulator/
├── data/                       # Raw and processed JSON/CSV data
├── output/                     # Simulation results, interactive HTML plots, and logs
└── src/                        # Source code
    ├── main.py                 # Top-level CLI orchestrator
    ├── scraper.py              # Limitless TCG HTML data ingestion
    ├── core/                   # Shared infrastructure and truths
    ├── evolution/              # Engine 1: Long-term Metagame Evolution (Game Theory)
    ├── tournament/             # Engine 2: Immediate Tournament Solver (Monte Carlo)
    └── interfaces/             # User entry points (Streamlit & CLI args)
```

---

## Top-Level Orchestrators (`src/`)

### `main.py`

**Purpose:** The main CLI entry point. Parses command-line arguments, orchestrates the simulation, analysis, and plotting pipeline. Also handles the CLI-based prediction mode. Manages unique output directory creation and experiment metadata.

**Key Functions/Classes:**

*   `parse_args() -> Args`: Parses and validates CLI arguments using `argparse` (defined in `cli_args.py`).
*   `run_single_experiment(args: Args, config_override: Optional[Dict]) -> Dict`: The primary orchestration function for a single simulation run. It:
    *   Sets up logging to a unique, timestamped output directory.
    *   Loads matchup data using `data.load_matchup_data`.
    *   **Constructs a `SimulationConfig` object *directly* from the parsed `args`.**
    *   Runs the simulation using `simulation.find_evolutionary_stable_state`, passing the config and a path for incremental history logging.
    *   Performs post-simulation analysis using functions from `analysis.py`.
    *   Generates plots using functions from `plotting.py`.
    *   Renames the output directory based on the convergence outcome.
    *   Returns metadata for batch runs.
*   `run_batch_experiments(args: Args)`: Manages running multiple simulation experiments with different configurations.
*   `main()`: The top-level entry point that delegates to single/batch runners or the CLI prediction mode.

### `scraper.py`

**Purpose:** A utility script to scrape and aggregate matchup data from HTML files (specifically from Limitless TCG) and convert it into the required JSON format (`ea_input.json`). It enforces strict data hygiene before passing the matrix to the core simulation engine.

**Key Functions:**

*   `normalize_archetype(name: str) -> str`: Normalizes archetype names using fast string replacement for consistent matrix mapping.
*   `extract_deck_info_from_filename(filename: str) -> Tuple[str, str]`: Strictly parses metadata relying on the required `Archetype - Format - Website` file naming convention.
*   `get_deck_archetype(file_path: str, filename: str) -> Tuple[str, str]`: Orchestrates metadata extraction, falling back to precise HTML `<div class="format">` scraping if the filename does not match conventions.
*   `scrape_matchup_data(file_path: str, ...) -> List[Dict[str, Any]]`: Scrapes the matchup table from an HTML file. **Crucially enforces data hygiene by explicitly ignoring the non-cohesive `"Other"` archetype** to protect downstream clustering accuracy.
*   `build_complete_matchup_matrix(all_matchup_data: List[Dict]) -> Dict`: Aggregates scraped data into a complete, mirrored matchup matrix.
*   `save_matrix_to_csv(...) / save_to_csv(...)`: Exports the aggregated data, actively enforcing a **4-decimal (`.4f`) float precision** to preserve mathematical sensitivity for the Replicator Dynamics calculations.

---

## Domain: Core (`src/core/`)

Shared data, configurations, and typing definitions used by both engines.

### `config.py`

**Purpose:** Defines the default configuration values used throughout the project.

**Key Variables:**

*   `SIMULATION_MODE`: Default simulation mode (`'replicator'` or `'tournament'`).
*   `MAX_GENERATIONS`, `MIN_GAMES`, `EXTINCTION_THRESHOLD`, `STABILITY_THRESHOLD`, `CONVERGENCE_WINDOW`, `MAX_INACTIVE_GENERATIONS`, `RNG_SEED`: Core simulation defaults.
*   `USE_BAYESIAN_WINRATES`, `TOURNAMENT_SIZE`, `NUM_TOURNAMENTS_PER_GEN`, `NUM_ROUNDS`, `USE_MULTIPROC`, `_STRUCTURE_THRESHOLDS`, `_STRUCTURE_RESULTS`: Tournament simulation defaults and official TPCi tier logic.
*   `DYNAMIC_DECK_INTRO_PROB`, `MUTATION_FLOOR`, `NOISE_SCALE`, `SELECTION_PRESSURE`: Simulation enhancement defaults.
*   `TIER_S_THRESHOLD`, `CONSISTENCY_MEAN_EPSILON`, `COMPOSITE_SCORE_WR_WEIGHT`: Post-analysis thresholds and weights.

### `data.py`

**Purpose:** Handles loading, cleaning, validating, and preprocessing the metagame matchup data. Also provides utilities for clustering and diagnostics.

**Key Functions:**

*   `safe_normalize(vec: np.ndarray) -> np.ndarray`: Normalizes a vector to sum to 1.0, returning a uniform distribution if the sum is zero.
*   `load_matchup_data(file_path: str, min_matches_required: int) -> Tuple[List[str], np.ndarray, Dict]`: Loads matchup data from a JSON file, filters decks by match volume, and returns reliable deck names, a **non-symmetric** win-rate matrix, and raw matchup details.
*   `cluster_decks_by_matchup_profile(win_matrix: np.ndarray, deck_names: List[str], n_clusters: int | str, method: str)`: Groups decks into strategic families based on their matchup profile "shape." It applies `StandardScaler` to normalize raw win-rate magnitudes and supports `"auto"` dynamic Silhouette Score optimization to find the mathematically optimal number of tiers/clusters.
*   `compute_deck_dominance(win_matrix: np.ndarray, deck_names: List[str]) -> np.ndarray`: Computes and logs the most dominant deck based on its expected win rate against an even field.

### `types.py`

**Purpose:** Defines the `SimulationConfig` dataclass, which bundles all parameters required for the core simulation engine into a single, typed object.

**Key Class:**

*   `SimulationConfig(dataclass)`: A mutable configuration object constructed directly from the CLI parsed arguments. Its fields include `mode`, `max_generations`, `min_generations`, `extinction_threshold`, `stability_threshold`, `convergence_window`, `max_inactive_generations`, `use_bayesian_winrates`, `tournament_size`, `num_tournaments_per_gen`, `num_rounds`, `use_multiproc`, `seed`, `dynamic_deck_intro_prob`, `mutation_floor`, `noise_scale`, `selection_pressure`, and the newly added `tournament_style`.

---
## Domain: Evolution (`src/evolution/`)

The theoretical game-theory engine designed to predict the Evolutionary Stable State (ESS) over thousands of generations.

### `analysis.py`

**Purpose:** Performs post-simulation analysis to generate insights like tier lists, convergence metrics, and cycle detection.

**Key Functions:**

*   `compute_convergence_metrics(history: List[np.ndarray], stability_threshold: float) -> Dict[str, Any]`: Quantifies how quickly and stably the metagame converged by analyzing the history of frequency changes.
*   `generate_final_state_tier_list(deck_names: List[str], metagame_history: List[np.ndarray], win_matrix: np.ndarray, ...) -> Dict[str, List[Dict]]`: Generates a tier list (S, A, B, C, D) based on the final state of the metagame, prioritizing meta-weighted win rate and presence.
*   `generate_all_time_tier_list(deck_names: List[str], metagame_history: List[np.ndarray], win_matrix: np.ndarray) -> Dict[str, List[Dict]]`: Generates a tier list based on a deck's overall performance, consistency, and impact across the entire simulation history. **This function is highly optimized using vectorized NumPy operations.**
*   `compute_matchup_cycles(win_matrix: np.ndarray, deck_names: List[str], cycle_length: int) -> List[List[str]]`: Identifies unique Rock-Paper-Scissors (RPS) cycles in the matchup graph utilizing efficient `itertools.combinations`.
*   `compute_deck_similarity(win_matrix: np.ndarray, deck_names: List[str], final_active_mask: Optional[List[bool]]) -> np.ndarray`: Computes pairwise cosine similarity between decks based on their matchup profiles. Filters out extinct decks and performs K-Means clustering on the active subset utilizing the new `"auto"` Silhouette optimization.

### `engine.py`

**Purpose:** Contains the core engine for simulating metagame evolution over generations.

**Key Functions:**

*   `get_variant_5_structure`: Implements the official Play! Pokémon Handbook standards. It dynamically maps player volume to specific Day 1/Day 2 round counts and match-point advancement thresholds (e.g., 16 or 19 points).
*   `_championship_series_worker`: Simulates a 2-day event and scales performance into "Win-Equivalents" for the engine.
*   `_pure_swiss_worker`: Simulates a standard, fast, single-phase Swiss tournament for faster evolutionary scaling when Championship structures aren't required.
*   `run_tournament_generation(current_freq: np.ndarray, ...) -> np.ndarray`: Runs one generation of stochastic tournaments (using workers optionally in parallel) and returns the new metagame frequency vector based on `selection_pressure`.
*   `update_replicator_dynamics(current_freq: np.ndarray, win_matrix: np.ndarray, rng: np.random.Generator, noise_scale: float) -> np.ndarray`: Implements the Replicator Dynamics equation with optional Gaussian noise, using a passed-in RNG for reproducibility.
*   `find_evolutionary_stable_state(deck_names: List[str], win_matrix: np.ndarray, matchup_details: Dict, config: SimulationConfig, history_file_path: Optional[str]) -> Tuple[List[Dict], List[np.ndarray], List[Optional[int]]]`: The core evolutionary loop. It iterates generations via either tournament simulation or replicator dynamics, manages deck extinctions and reintroductions, handles incremental CSV history logging, and detects when stability thresholds have been achieved.

### `plotting.py`

**Purpose:** Generates interactive visualizations using Plotly for the simulation results.

**Key Functions:**

*   `plot_metagame_evolution_interactive(history: List[np.ndarray], deck_names: List[str], ...) -> Optional[go.Figure]`: Creates an interactive line plot showing the metagame share of top decks over time.
*   `plot_matchup_heatmap_interactive(win_matrix: np.ndarray, deck_names: List[str], ...) -> Optional[go.Figure]`: Creates an interactive heatmap of the win-rate matrix, optionally sorted by tier.
*   `plot_matchup_network(win_matrix: np.ndarray, deck_names: List[str], cycles: List[List[str]], ...) -> Optional[go.Figure]`: Creates an interactive network graph visualizing significant win-rate edges and highlighting detected RPS cycles. Nodes scale dynamically based on the deck's all-time presence.

---

## Domain: Tournament (`src/tournament/`)

The practical, immediate predictive engine designed to evaluate individual Expected Value (EV) for upcoming events.

## `monte_carlo.py`

**Purpose:** High-performance computational core of the `app.py`. It uses parallelized NumPy workers to simulate thousands of complete tournament brackets to determine individual and archetype equity.

**Key Components:**

*   `_mc_worker:` The parallelized worker function. It pairs players based on current match points and executes seeded Top Cut brackets. It features an optimized **Parabolic Tie Convergence (BETA)** model utilizing an inverted boolean mask (`~(p1_wins | p2_wins)`) to efficiently simulate match-point decay in BO3 Swiss rounds.
*   `run_monte_carlo_analytics:` The primary entry point for simulations. It manages the multiprocessing pool, aggregates raw conversion counts, and calculates Win Probability and conversion shares (Day 2 / Top 8).

### `solver.py`

**Purpose:** Contains the core logic for the static recommendation engine. It takes a user-defined metagame, calculates a plausible meta utilizing real-world empirical smoothing, and computes advanced baseline performance metrics for all decks.

**Key Functions:**

*   `resolve_meta_constraints` **(Vectorized Water-Filling Algorithm)**: Strictly enforces user-defined constraints (Exact or Range). Utilizing high-performance pure NumPy boolean masking, it distributes the remaining tournament mass proportionally based on Laplace-smoothed empirical data.
*   `swiss_rounds_from_players(n_players: int) -> int`: Calculates the number of Swiss rounds based on `math.ceil(math.log2(n))`.
*   `predict_best_decks(user_meta_spec: UserMetaSpec, ...) -> PredictionResult`: Orchestrates the baseline calculation. It applies the water-filling constraints and generates foundational Swiss metrics (SoS, OMW, Expected Win Rate) before passing the results to the Monte Carlo engine for bracket execution.

---

## Domain: Interfaces (`src/interfaces/`)

UI and argument definitions. These files contain no core math and exist purely to bridge the user and the engines.

### `app.py`

**Purpose:** The primary interactive web dashboard built with Streamlit. It allows users to define custom metagame constraints, ingest real-world Limitless TCG data, and evaluate tournament equity (Expected Value) from both an individual player's perspective and a macro-metagame impact perspective.

**Key Functions & UI Elements:**

*   `parse_limitless_html(html_str: str, valid_decks: List[str])`: A native HTML parser that extracts recognized deck archetypes and player counts directly from Limitless Labs exports to pre-populate custom constraints.
*   `calculate_ultimate_score(res, mc_res, d2_rounds, top_cut)`: The core evaluation engine. It merges base win rates with Monte Carlo results, applying a strict **Z-Score Standardization** mapped to a sigmoid curve (`np.tanh`). It calculates *two* scores simultaneously: `score_player` (Individual EV) and `score_archetype` (Macro Metagame Impact).
*   **Dual-Perspective Dashboard:** A dynamic UI component that seamlessly swaps data sorting, tier assignments, and top recommendations based on the user's selected POV without re-triggering the Monte Carlo engine.
*   **Head-to-Head Comparator:** An interactive `st.dataframe` utilizing Pandas `Styler` to color-code favorable ($\ge 55\%$) and unfavorable ($\le 45\%$) matchups between a primary and challenger deck across the predicted field.

### `cli_args.py`

**Purpose:** Defines the command-line interface (CLI) and argument parsing logic.

**Key Functions/Classes:**

*   `Args(NamedTuple)`: A typed structure for holding parsed CLI arguments. Includes fields for both simulation (`gens`, `noise`, etc.) and prediction (`predict`, `players`, `meta`).
*   `parse_args() -> Args`: Sets up the `argparse.ArgumentParser`, defines all CLI flags with their defaults (sourced from `config.py`), performs validation (e.g., file existence, value ranges), and returns an `Args` object.

---

## Project Configuration Files

### `pyrightconfig.json`

**Purpose:** Configuration file for the Pyright static type checker.

**Key Settings:**

*   `"typeCheckingMode": "basic"`: Sets the strictness level.
*   `"pythonVersion": "3.12"`: Specifies the target Python version.
*   `"exclude"`: Lists directories and files to ignore during type checking (e.g., `input/`, `output/`, `__pycache__/`).

### `requirements.txt`

**Purpose:** Lists the Python package dependencies required to run the project (e.g., `numpy`, `scipy`, `scikit-learn`, `plotly`, `tqdm`, `beautifulsoup4`, `streamlit`).

Last Edited: 20.03.2026, 02:00 UTC+2