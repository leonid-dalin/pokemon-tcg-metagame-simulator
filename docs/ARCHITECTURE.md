# ARCHITECTURE.md

This document provides a comprehensive overview of the project's architecture for fellow developers. It details the purpose, structure, and key components (functions, variables, classes) of each module to ensure a clear understanding of the system's scope and design.

The project simulates the long-term evolutionary dynamics of a competitive Pokémon TCG metagame using advanced evolutionary game theory (Optimistic Multiplicative Weights Update) and agent-based tournament simulations. It predicts the stable equilibrium state (Evolutionary Stable State) and provides advanced analytics, visualizations, and an interactive web application for real-time tournament equity predictions.

⚠️ Caution: This project is undergoing active refactoring. Expect frequent changes to the architecture, and verify information with the latest source code. I'll do my best to keep it updated, but you know, I'm only human.

---

## Simulation Assumptions & Limitations

To maintain high-speed parallel performance and prevent statistical noise from overwhelming the signal, the `IRL Tournament Engine` and `Evolution Engine` operate under a specific set of constrained assumptions:

1. **Perfect Equal Skill (Flat Elo):** The simulator assumes all pilots(/players) are of perfectly equal, "average" skill level. The engine pulls statically from the provided win-rate matrix (e.g., Limitless TCG data) and does not inject Elo-style distributions. A Top-10 global player and a Day 1 casual are treated identically by the math. If there's interest, I can implement `elo` as an arg for Advanced Users to tweak over, but as it stands it does **not** cover the scope of this project at this stage.
2. **Matchup Determinism:** Basically, **no luck.** While the tournament bracket pairings are stochastic (RNG-driven), the actual match results are resolved purely as a weighted coin flip based on the archetype matchup data. There is no simulation of bricked hands, dead draws, or top-decks. We make the assumption that the winrate and data we already gathered are **absolute** in their nature.
3. **Speed-Agnostic Tie Convergence:** The Parabolic Tie Convergence feature forces match-point bleed (ties) based purely on the mathematical closeness of the matchup (e.g., a 50/50 matchup ties more than an 80/20 matchup). It is entirely blind to deck archetype speed. I don't want to insert more *magic numbers* on which archetypes I consider 'easy' to pilot and which I don't. Do they time waste because they're a `Gholdengho` player, because they're versing a difficult match-up, or because that's the *right* decision to do Competitively by leading first game 1-0? Nobody knows. And **ignorance is bliss.** 
4. **No Mid-Tournament Teching:** I won't cover the JP Tournament system. A deck's strategic profile remains frozen for the duration of the tournament or generation. The engine does not account for players altering tech cards (e.g., adding a specific counter-card) in response to expected Day 2 shifts. *If only I had data on Mulligan WR, Draw WR, Played WR, etc. ...*
5. **Structured Variance:** The Monte Carlo engine simulates tournament realities (e.g., BO3 math conversions via $3p^2 - 2p^3$, time-limit Tie Convergence, and X-3 Drop Logic), but assumes perfectly randomized pairings within equivalent point brackets (Swiss).
6. **Thermodynamic Data Purity (Zero-Sum):** Real-world matchup data is inherently General-Sum due to reporting biases. The simulation mathematically forces empirical data into a strict Zero-Sum environment. The Scraper achieves this via **Zero-Sum Match Equity** (splitting Ties evenly), and the Engine enforces symmetry to eliminate "free game-theoretic energy."

---

## Directory Structure

The codebase utilizes a domain-driven layout to strictly separate long-term evolutionary mechanics from immediate tournament equity evaluation:

```text
pokemon-tcg-metagame-simulator/
├── data/                       # Raw and processed JSON/CSV data
├── output/                     # Simulation results, interactive Plotly HTML, and logs
└── src/                        # Source code
    ├── main.py                 # Top-level CLI orchestrator
    ├── scraper.py              # Limitless TCG HTML data ingestion
    ├── core/                   # Shared infrastructure and truths
    ├── evolution/              # Engine 1: Long-term ESS Replicator Engine & Analytics (Game Theory)
    ├── tournament/             # Engine 2: Short-term Tournament Solver (Monte Carlo Bracket Engine)
    └── interfaces/             # User entry points (Streamlit & CLI parsers)
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
*   `TIER_THRESHOLDS`, `TIER_ORDER`, `WIN_THRESHOLD`: Post-analysis thresholds defining the strict T0 through T4 tiering system.

> Note: Older composite score weights like `COMPOSITE_SCORE_WR_WEIGHT` are now retained only as `[LEGACY]`.

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

*   `compute_convergence_metrics(history: List[np.ndarray], stability_threshold: float) -> Dict[str, Any]`: Quantifies how quickly and stably the metagame converged by analysing the history of frequency changes.
*   `generate_final_state_tier_list(deck_names: List[str], metagame_history: List[np.ndarray], win_matrix: np.ndarray) -> Dict[str, List[Dict]]`: Generates a tier list (T0, T0.5, T1, T2, T3, T4) based on the final state of the metagame. It uses the updated Meta Score logic, evaluating a deck's Power Score (win rate against the field) and Frequency Score (presence).
*   `compute_matchup_cycles(win_matrix: np.ndarray, deck_names: List[str], cycle_length: int) -> List[List[str]]`: Identifies unique Rock-Paper-Scissors (RPS) cycles in the matchup graph utilising efficient `itertools.combinations`.
*   `compute_deck_similarity(win_matrix: np.ndarray, deck_names: List[str], final_active_mask: Optional[List[bool]]) -> np.ndarray`: Computes pairwise cosine similarity between decks based on their matchup profiles. Filters out extinct decks and performs K-Means clustering on the active subset utilising `"auto"` Silhouette optimisation.

### `engine.py`

**Purpose:** Contains the core engine for simulating metagame evolution over generations.

**Key Functions:**

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

*   `_mc_worker:` The parallelised worker function. It pairs players based on current match points and executes seeded Top Cut brackets. It features an optimised **Parabolic Tie Convergence (BETA)** model utilising an inverted boolean mask (`~(p1_wins | p2_wins)`) to efficiently simulate match-point decay in BO3 Swiss rounds. It now also integrates **Heuristic Rematch Prevention** (stopping players from facing the same opponent twice) and **X-3 Drop Logic** (simulating realistic tournament attrition).
*   `run_monte_carlo_analytics:` The primary entry point for simulations. It manages the multiprocessing pool, aggregates raw conversion counts, and calculates Win Probability and conversion shares (Day 2 / Top 8).


### `solver.py`

**Purpose:** Contains the core logic for the static recommendation engine. It takes a user-defined metagame, calculates a plausible meta utilizing real-world empirical smoothing, and computes advanced baseline performance metrics for all decks.

**Key Functions:**

*   `get_variant_5_structure`: Implements the official Play! Pokémon Handbook standards. It dynamically maps player volume to specific Day 1/Day 2 round counts and match-point advancement thresholds.
*   `resolve_meta_constraints` **(Vectorised Water-Filling Algorithm)**: Strictly enforces user-defined constraints (Exact or Range). Utilising high-performance pure NumPy boolean masking, it distributes the remaining tournament mass proportionally based on Laplace-smoothed empirical data.
*   `predict_best_decks(user_meta_spec: UserMetaSpec, ...) -> PredictionResult`: Orchestrates the baseline calculation. It applies the water-filling constraints and generates foundational Swiss metrics. Crucially, it calculates the `power_score` (a 0–100 min-max normalisation of expected win rate), `frequency_score`, and `base_meta_score` directly, passing these refined metrics forward to the UI and Monte Carlo engine.

---

## Domain: Interfaces (`src/interfaces/`)

UI and argument definitions. These files contain no core math and exist purely to bridge the user and the engines.

### `app.py`

**Purpose:** The primary interactive web dashboard built with Streamlit. It allows users to define custom metagame constraints, ingest real-world Limitless TCG data, and evaluate tournament equity (Expected Value) from both an individual player's perspective and a macro-metagame impact perspective.

**Key Functions & UI Elements:**

*   `parse_limitless_html(html_str: str, valid_decks: List[str])`: A native HTML parser that extracts recognised deck archetypes and player counts directly from Limitless Labs exports to pre-populate custom constraints.
*   **Dual-Perspective Dashboard:** A dynamic UI component that seamlessly swaps data sorting, tier assignments (T0 through T4), and top recommendations based on the user's selected POV (Individual EV vs. Macro Metagame Impact). It reads these metrics directly from the pre-calculated results in `solver.py` without needing to re-trigger the Monte Carlo engine.
*   **Head-to-Head Comparator:** An interactive `st.dataframe` utilising Pandas `Styler` to colour-code favourable (≥ 55%) and unfavourable (≤ 45%) matchups between a primary and challenger deck across the predicted field.

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

Last Edited: 22.03.2026, 03:00 UTC+2