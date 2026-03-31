# Codebase Reference Dictionary

This reference document maps out the specific files, modules, classes, and functions within the Pokémon TCG Metagame Simulator. Use this as a technical dictionary to locate machinery within the codebase.

> ⚠️ **Caution:** This project is undergoing active refactoring. Expect frequent changes to the architecture, and verify information with the latest source code. I'll do my best to keep it updated or have it updated ASAP, but bear in mind this is a 1-person passion project.

---

## Gateway & Orchestration

### `main.py`
**Purpose:** Acts as the primary RESTful API gateway.
* Instantiates the FastAPI application and configures `CORSMiddleware` to allow Streamlit communications.
* `start_prediction(...)`: Enqueues the simulation job into the Huey task queue and immediately returns a task ID.
* `get_task_status(...)`: A polling endpoint to check if the background Huey worker has completed the calculations.

### `cli.py`
**Purpose:** The main CLI entry point for local execution, parsing command-line arguments, orchestrating the simulation pipeline, and managing batch experiments.
* `run_single_experiment(...)`: Orchestrates a single simulation run. It sets up timestamped logging, loads matchup data, constructs a `SimulationConfig`, runs `find_evolutionary_stable_state`, performs tier analysis, and plots the results.
* `run_batch_experiments(...)`: Iterates over multiple simulation configurations defined in a JSON file to automate parameter sweeping.
* `main()`: Delegates to either the static predictor solver, the batch runner, or the single experiment runner based on parsed arguments.

### `queue.py`
**Purpose:** Manages the background execution of computationally expensive tasks and scheduled cron jobs.
* `execute_simulation_job(...)`: A background task that loads data, runs the water-filling logic, determines tournament structure, and triggers the parallelized Monte Carlo analytics.
* `automated_daily_pipeline()`: A periodic background task (`crontab`) that automatically triggers the Limitless TCG scraper, validates the incoming thermodynamic matrix using Pydantic, and updates the local JSON data store.

### `models.py`
**Purpose:** Defines strict Pydantic type schemas for API endpoint validation.
* `ExactSpec` & `RangeSpec`: Enforces minimum, maximum, and exact percentage boundaries (0.0 to 1.0) for metagame constraints.
* `PredictionRequest`: Bundles the metagame specification along with parameters like iterations, tie rates, and match formats.

---

## Core Domain (`src/core/`)

### `config.py`
**Purpose:** Defines default configuration values, tournament structures, and mathematical constants.
* `MAX_GENERATIONS`, `STABILITY_THRESHOLD`, `NASH_EQUILIBRIUM`, `EXTINCTION_THRESHOLD`: Evolutionary halting conditions.
* `_STRUCTURE_THRESHOLDS` & `_STRUCTURE_RESULTS`:  Implements the official TPCi Variant #5 logic, mapping player counts to specific Day 1/Day 2 round lengths and cut points.
* `TIER_THRESHOLDS`: Delineates the strict mathematical cutoffs for Tiers (e.g., T0 is $\ge$ 0.525, T4 is $\ge$ 0.0).

### `data.py`
**Purpose:** Handles loading, cleaning, validating, clustering, and preprocessing the metagame matchup data.
* `load_matchup_data(...)`: Loads the JSON matrix and strictly enforces the 0.5 diagonal rule.
* `cluster_decks_by_matchup_profile(...)`: Groups decks using StandardScaler normalization and methods like `KMeans` or `AgglomerativeClustering`. It utilizes an "auto" Silhouette Score optimizer to dynamically find the best cluster count.
* `compute_deck_dominance(...)`: Identifies the top deck against an evenly distributed baseline.

### `scraper.py`
**Purpose:** Automates data ingestion from Limitless TCG, parses HTML DOM structures, and builds the foundational win-rate matrices.
* `fetch_live_matchup_data(...)`: Orchestrates HTTP requests to fetch live HTML matchup pages and populates the canonical archetype whitelist.
* `scrape_matchup_soup(...)`: Parses the HTML `<table>` elements to extract opponents, match volumes, and W-L-T records, returning calculated win percentages.
* `build_complete_matchup_matrix(...)`: Constructs the fully mirrored $N \times N$ matrix from flat matchup dictionaries and strictly enforces the 0.5 diagonal rule for mirror matches.
* `normalize_archetype(...)`: A string-cleaning utility that normalizes deck names (handling ampersands, whitespace, and special characters) to ensure stable dictionary mapping.

### `types.py`
**Purpose:** Houses typed data classes bridging configuration into the engines.
* `SimulationConfig`: A dataclass containing core properties like extinction thresholds, noise scales, and tournament styles.

---

## Evolution Engine (`src/evolution/`)

### `engine.py`
**Purpose:** Contains the core math for simulating metagame evolution over generations.
* `update_replicator_dynamics(...)`: Executes the Multiplicative Weights Update (MWU). It extrapolates gradients using previous payoffs to dampen zero-sum limit cycles, centers dynamics, and applies uniform mutation as entropy regularization.
* `_championship_series_worker(...)` & `_pure_swiss_worker(...)`: Multiprocessing worker functions executing array-sliced bracket simulations.
* `find_evolutionary_stable_state(...)`: The core loop that iterates until kinetic stability and G Game-Theoretic stability (Nash Equilibrium) is reached.

### `analysis.py`
**Purpose:** Generates analytical insights from raw simulation history.
* `compute_convergence_metrics(...)`: Determines the exact generation of stability and the oscillation index.
* `generate_final_state_tier_list(...)`: Sorts decks into Tiers utilizing a blended Meta Score.
* `compute_matchup_cycles(...)`: Extracts unique Rock-Paper-Scissors (RPS) 3-cycles from the active matchup graph.
* `compute_deck_similarity(...)`: Uses Pearson Correlation to find decks with strong strategic overlap (correlation > 0.70).

### `plotting.py`
**Purpose:** Generates interactive Plotly visualizations.
* `plot_metagame_evolution_interactive(...)`: Renders a line chart tracking metagame share over time.
* `plot_matchup_network(...)`: Builds an interactive nodes-and-edges graph visualizing win rates and historical presence.
* `plot_metagame_scatter(...)` & `plot_head_to_head_radar(...)`: Powers the Streamlit dashboard visuals, comparing Meta Scores, Power Scores, and Frequency metrics.

---

## Tournament Engine (`src/tournament/`)

### `monte_carlo.py`
**Purpose:** The computational core simulating thousands of tournament brackets.
* `_mc_worker(...)`: Parallelized NumPy worker running seeded Swiss brackets with Heuristic Rematch Prevention and X-3 Drop Logic.
* `run_monte_carlo_analytics(...)`: Manages the multiprocessing pool to aggregate total Top Cut appearances, Day 2 conversion rates, and overall event win probabilities.

### `solver.py`
**Purpose:** The static engine establishing metagame limits and predictive baselines.
* `calculate_empirical_baseline(...)`: Derives live metagame share using Laplace smoothing.
* `resolve_meta_constraints(...)`: A vectorized water-filling algorithm strictly enforcing exact, minimum, and maximum boundaries.
* `predict_best_decks(...)`: Orchestrates static recommendations, generating Power Scores and Meta Scores.

---

## User Interfaces (`src/ui/`)

### `app.py`
**Purpose:** The interactive Streamlit dashboard.
* `parse_limitless_html(...)`: Native HTML parsing utility for Limitless Labs exports.
* Dual-Perspective Dashboard: Toggles data between Player EV and Archetype Macro dominance.
* Head-to-Head Comparator: Utilizes Pandas Styler to color-code win rates and contrast stats directly.

### `cli_args.py`
**Purpose:** Defines the argparse configuration for CLI tools.
* `Args`: A NamedTuple strongly typing the returned namespace.
* `parse_args()`: Validates arguments to ensure parameters stay within acceptable mathematical bounds.