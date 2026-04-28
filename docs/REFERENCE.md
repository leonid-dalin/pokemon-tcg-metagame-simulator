# Codebase Reference Dictionary

This reference document maps out the specific files, modules, classes, and functions within the Pokémon TCG Metagame Simulator. Use this as a technical dictionary to locate machinery within the codebase.

> ⚠️ **Caution:** This project is undergoing active refactoring. Expect frequent changes to the architecture, and verify information with the latest source code. I'll do my best to keep it updated or have it updated ASAP, but bear in mind this is a 1-person passion project.

---

## Gateway & Orchestration

### `main.py`
**Purpose:** Acts as the primary RESTful API gateway.
* `lifespan`: Asynchronous context manager triggering the scraper on container boot.
* `log_requests`: Middleware recording HTTP response codes and duration using `structlog`.
* `start_prediction`: Enqueues a heavy Monte Carlo simulation via `execute_simulation_job` and returns a task ID.
* `stream_task_progress`: Server-Sent Events endpoint yielding real-time job percentages from Redis.

---

## Data Contracts and Background Task Management

### `models.py`
**Purpose:** Pydantic schemas enforcing data validation.
* `PredictionRequest`: Validates incoming API payloads. Includes `validate_matrix_integrity` to block asymmetric or invalid win-rate matrices.
* `ScrapedMatrix`: Acts as a firewall for live data. Includes `enforce_thermodynamic_purity` to ensure mirror matches are exactly 0.5.
* `DeckRecommendation`: Formats the final output schema.

### `queue.py`
**Purpose:** The Redis-backed Huey task broker.
* `execute_simulation_job(...)`: A background task that loads data, runs the water-filling logic, determines tournament structure, and triggers the parallelised Monte Carlo analytics.
* `automated_daily_pipeline()`: A periodic background task (`crontab`) that automatically triggers the Limitless TCG scraper, validates the incoming thermodynamic matrix using Pydantic, and updates the local JSON data store.

---

## Core Domain (`src/core/`)

### `config.py`
**Purpose:** Defines default configuration values, tournament structures, and mathematical constants.
* `MAX_GENERATIONS`, `STABILITY_THRESHOLD`, `NASH_EQUILIBRIUM`, `EXTINCTION_THRESHOLD`: Evolutionary halting conditions.
* `_STRUCTURE_THRESHOLDS` & `_STRUCTURE_RESULTS`: Implements official TPCi Variant #5 logic, mapping player counts to round lengths and cut points.
* `TIER_THRESHOLDS`: Delineates the strict mathematical cutoffs for Tiers (e.g., T0 is $\ge$ 0.525, T4 is $\ge$ 0.0).

### `data.py`
**Purpose:** Core matrix manipulation, data ingestion, and archetype clustering algorithms.
* `load_matchup_data(...)`: Parses JSON matrices into NumPy arrays, substituting unrecorded matchups with Bayesian beta distribution fallbacks.
* `cluster_decks_by_matchup_profile(...)`: Groups decks by strategic vector similarity using K-Means and dynamically optimises the final cluster count via Silhouette Scores.
* `compute_deck_dominance(...)`: Identifies the optimal archetype against a uniform field baseline using direct matrix-vector multiplication.

### `logger.py`
**Purpose:** Global structured logging configuration.
* `setup_structured_logging()`: Configures `structlog` to format application output as machine-readable JSON strings (`JSONRenderer`) with ISO 8601 timestamps for ELK/Loki ingestion.

### `scraper.py`
**Purpose:** Autonomous data ingestion pipeline targeting live Limitless TCG endpoints.
* `fetch_live_matchup_data(...)`: Utilises `concurrent.futures.ThreadPoolExecutor` to perform parallel HTTP requests.
* `scrape_matchup_soup(...)`: Parses raw HTML responses using `BeautifulSoup` to calculate strictly zero-sum adjusted win rates.
* `build_complete_matchup_matrix(...)`: Aggregates scraped payloads into a multidimensional JSON matrix ready for Pydantic validation.

### `telemetry.py`
**Purpose:** OpenTelemetry (OTel) distributed tracing pipeline.
* `setup_telemetry(...)`: Instantiates the `TracerProvider` and binds the `OTLPSpanExporter` via gRPC to stream execution spans using a `BatchSpanProcessor`.

### `types.py`
**Purpose:** Houses typed data classes bridging configuration into the engines.
* `SimulationConfig`: A dataclass containing core properties like extinction thresholds, noise scales, and tournament styles.

---

## Evolution Engine (`src/evolution/`)

### `engine.py`
**Purpose:** Replicator dynamics and tournament generation workers.
* `_championship_series_worker(...)` & `_pure_swiss_worker(...)`: Multiprocessing worker functions executing array-sliced bracket simulations.
* `update_replicator_dynamics(...)`: Executes the Multiplicative Weights Update (MWU). It extrapolates gradients using previous payoffs to dampen zero-sum limit cycles, centers dynamics, and applies uniform mutation as entropy regularisation.
* `find_evolutionary_stable_state(...)`: The core loop that iterates until kinetic stability and G Game-Theoretic stability (Nash Equilibrium) is reached.

### `analysis.py`
**Purpose:** Post-simulation diagnostics.
* `compute_convergence_metrics(...)`: Determines the exact generation of stability and the oscillation index.
* `generate_final_state_tier_list(...)`: Sorts decks into Tiers utilising a blended Meta Score.
* `compute_matchup_cycles(...)`: Extracts unique Rock-Paper-Scissors (RPS) 3-cycles from the active matchup graph.
* `compute_deck_similarity(...)`: Uses Pearson Correlation to find decks with strong strategic overlap (correlation > 0.70).

---

## Tournament Engine (`src/tournament/`)

### `monte_carlo.py` and `lib.rs`
**Purpose:** The computational core simulating thousands of tournament brackets.
* `run_monte_carlo_analytics`: Manages the Rayon thread pool to aggregate Top Cut appearances and win probabilities.
* `tcg_engine`: The Rust module bypassing the Python GIL.
* `play_rounds`: Native Rust implementation of Swiss pairing logic.

### `solver.py`
**Purpose:** The static evaluation engine establishing baseline predictive scoring and metagame limits.
* `get_variant_5_structure(...)`: Maps player counts to official TPCi Day 1/Day 2 Swiss round configurations via binary search.
* `swiss_rounds_from_players(...)`: Calculates standard logarithmic Swiss rounds for non-Championship events.
* `predict_best_decks(...)`: Resolves user-defined field constraints using a pure NumPy water-filling algorithm to generate normalised 0–100 `power_scores` and `base_meta_scores`.
* `calculate_empirical_baseline(...)`: Derives live metagame share using Laplace smoothing.
* `resolve_meta_constraints(...)`: A vectorised water-filling algorithm strictly enforcing exact, minimum, and maximum boundaries.

---

## Analytics and Visualisation

### `plotting.py`
**Purpose:** Interactive Plotly figure generation.
* `plot_metagame_evolution_interactive(...)`: Renders a line chart tracking metagame share over time.
* `plot_matchup_network(...)`: Renders `networkx` directed graphs for cycle visualisations.
* `plot_metagame_scatter(...)` & `plot_head_to_head_radar(...)`: Powers the Streamlit dashboard visuals, comparing Meta Scores, Power Scores, and Frequency metrics.

---

## User Interfaces (`src/ui/`)

### `app.py`
**Purpose:** The interactive Streamlit dashboard.
* `parse_limitless_html(...)`: Native HTML parsing utility for Limitless Labs exports.
* `RequestsInstrumentor`: Propagates distributed trace headers from the UI directly into the FastAPI backend.
* **SSE Polling:** Dispatches simulation jobs and polls task states using Server-Sent Events via `urllib3` streams.
* **Dual-Perspective Dashboard:** Renders analytics using interactive Plotly scatter and radar graphs, filtering statistical noise via a strict 1% field share cutoff.

### `cli.py`
**Purpose:** The main CLI entry point for local execution, parsing command-line arguments, orchestrating the simulation pipeline, and managing batch experiments.
* `run_single_experiment(...)`: Orchestrates a single simulation run. It sets up timestamped logging, loads matchup data, constructs a `SimulationConfig`, runs `find_evolutionary_stable_state`, performs tier analysis, and plots the results.
* `run_batch_experiments(...)`: Iterates over multiple simulation configurations defined in a JSON file to automate parameter sweeping.
* `main()`: Delegates to either the static predictor solver, the batch runner, or the single experiment runner based on parsed arguments.

### `cli_args.py`
**Purpose:** Defines the argparse configuration for CLI tools // Terminal argument parser.
* `Args`: A `NamedTuple` strongly typing the returned namespace.
* `parse_args()`: Validates arguments to ensure parameters stay within acceptable mathematical bounds.