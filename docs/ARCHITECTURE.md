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
5. **Structured Variance:** The Monte Carlo engine simulates tournament realities (e.g., BO3 math conversions via $3p^2 - 2p^3$, time-limit Tie Convergence, and X-3 Drop Logic). However, it assumes perfectly randomized pairings within equivalent point brackets using mergesort.
6. **Thermodynamic Data Purity (Zero-Sum):** Real-world matchup data is mathematically forced into a strict environment where the diagonal of the win matrix is exactly 0.5.

---

## System Architecture (Decoupled Microservices)

The simulator utilizes a containerized, three-tier architecture orchestrated via Docker Compose.

```text
┌─────────────────┐       HTTP / REST       ┌──────────────────┐
│                 │  (1) POST /api/v1/predict                   │
│  Streamlit UI   ├────────────────────────►│  FastAPI Server  │
│  (Thin Client)  │                         │  (Entry Point)   │
│                 │◄────────────────────────┤                  │
└─────────────────┘  (4) Return Task ID     └────────┬─────────┘
        ▲                                            │
        │ (5) Poll GET /api/v1/tasks/{task_id}       │ (2) Enqueue Job
        │                                            ▼
┌───────┴─────────┐                       ┌──────────────────┐
│                 │◄──── (3) Pull Job ────│                  │
│   Huey Worker   │                       │  SQLite Broker   │
│ (Heavy Compute) │───── (6) Write Res ──►│ (tcg_tasks.db)   │
│                 │                       │                  │
└─────────────────┘                       └──────────────────┘
```

1. `api` **(FastAPI)**: Serves as the gateway. Validates incoming simulation requests using strict Pydantic data contracts (`src.api.models`) and pushes tasks to the queue.
2. `worker` **(Huey):** A headless background processor running `execute_simulation_job`. It listens for tasks, executes the expensive `monte_carlo.py` and `solver.py` logic, and writes results back to a lightweight SQLite backend (`tcg_tasks.db`).
3. `ui` **(Streamlit)**: The visual frontend located in `app.py`. It gathers user parameters, dispatches them to the API via HTTP, polls for results, and renders the resulting Plotly charts and Pandas dataframes.

---

## Top-Level Orchestrators & Microservices

### `main.py` **(FastAPI Server)**

**Purpose:** Acts as the primary RESTful API gateway.

**Key Components:**

* Instantiates the FastAPI application and configures `CORSMiddleware` to allow Streamlit communications.
* `start_prediction(request: PredictionRequest)`: Enqueues the simulation job into the Huey task queue and immediately returns a task_id.
* `get_task_status(task_id: str)`: A polling endpoint to check if the background Huey worker has completed the calculations.

### `cli.py` **(Command Line Interface)**

**Purpose:** The main CLI entry point for local execution, parsing command-line arguments, orchestrating the simulation pipeline, and managing batch experiments.

**Key Functions:**

* `run_single_experiment(...)`: Orchestrates a single simulation run. It sets up timestamped logging, loads matchup data, constructs a `SimulationConfig`, runs `find_evolutionary_stable_state`, performs tier analysis, and plots the results.
* `run_batch_experiments(...)`: Iterates over multiple simulation configurations defined in a JSON file to automate parameter sweeping.
* `main()`: Delegates to either the static predictor solver, the batch runner, or the single experiment runner based on parsed arguments.


### `queue.py` **(Huey Worker)**

**Purpose:** Manages the background execution of computationally expensive tasks, freeing up the API.

**Key Functions:**

* `execute_simulation_job(...)`: A `@huey.task()` that loads data, runs the `solver.py` water-filling logic, determines the tournament structure based on the number of players, and finally triggers the heavily parallelized `run_monte_carlo_analytics`.

### `models.py` **(Pydantic Contracts)**

**Purpose:** Defines strict type schemas for the FastAPI endpoints.

**Key Classes:**

* `ExactSpec` / `RangeSpec`: Enforces minimum, maximum, and exact percentage boundaries (0.0 to 1.0) for user-defined metagame constraints.

* `PredictionRequest`: Bundles the metagame specification along with parameters like `mc_iterations`, `use_tie_convergence`, and `match_format`.

---

## Domain: Core (`src/core/`)

### `config.py` 

**Purpose:** Defines the default configuration values, tournament structures, and mathematical constants used throughout the project.

**Key Variables:**
* `MAX_GENERATIONS`, `STABILITY_THRESHOLD`, `NASH_EQUILIBRIUM`, `EXTINCTION_THRESHOLD`: Defines evolutionary halting conditions.
* `_STRUCTURE_THRESHOLDS` & `_STRUCTURE_RESULTS`: Implements the official TPCi Variant #5 logic, mapping player counts to specific Day 1/Day 2 round lengths and cut points.
* `TIER_THRESHOLDS:` Delineates the strict mathematical cutoffs for Tiers (e.g., T0 is $\ge$ 0.525, T4 is $\ge$ 0.0).

### `data.py`

**Purpose:** Handles loading, cleaning, validating, clustering, and preprocessing the metagame matchup data.

**Key Functions:**
* `load_matchup_data(...)`: Loads the JSON matrix, filters archetypes by a `min_matches_required` threshold, and strictly enforces that the diagonal of the win matrix equals 0.5.
* `cluster_decks_by_matchup_profile(...)`: Groups decks using StandardScaler normalization and methods like `KMeans` or `AgglomerativeClustering`. It utilizes an "auto" Silhouette Score optimizer to dynamically find the best cluster count.
* `compute_deck_dominance(...)`: Identifies the top deck against an evenly distributed baseline.

### `types.py`

**Purpose:** Houses the typed data classes used to bridge configuration into the engines.

**Key Class:**
* `SimulationConfig`: A dataclass containing properties like `mode`, `extinction_threshold`, `use_bayesian_winrates`, `noise_scale`, `selection_pressure`, and `tournament_style`.

---

## Domain: Evolution (`src/evolution/`)

### `engine.py`

**Purpose:** Contains the core math for simulating metagame evolution over generations.

**Key Functions:**

* `update_replicator_dynamics(...)`: Executes the Multiplicative Weights Update (MWU). It extrapolates gradients using previous payoffs to dampen zero-sum limit cycles, centers dynamics, and applies uniform mutation as entropy regularization.
* `_championship_series_worker(...) & _pure_swiss_worker(...)`: Multiprocessing worker functions that execute large-scale, array-sliced bracket simulations. The championship worker implements Parabolic Tie Convergence, BO3 scaling, and the Variant #5 cut logic.
* `find_evolutionary_stable_state(...)`: The core evolutionary loop that iterates generations until kinetic stability and Game-Theoretic stability (Nash Equilibrium) are reached.

### `analysis.py`

**Purpose:** Generates analytical insights and diagnostic metrics from the raw simulation history.

**Key Functions:**

* `compute_convergence_metrics(...)`: Analyzes the metagame history array to determine the exact generation of stability and the oscillation index.
* `generate_final_state_tier_list(...)`: Sorts decks into Tiers utilizing a blended Meta Score derived from a normalized Power Score (win rate) and Frequency Score (presence).
* `compute_matchup_cycles(...)`: Employs `itertools.combinations` to extract unique Rock-Paper-Scissors (RPS) 3-cycles from the active matchup graph.
* `compute_deck_similarity(...):` Uses Pearson Correlation to find decks with strong strategic overlap (correlation > 0.70).

### `plotting.py`

**Purpose:** Generates interactive Plotly visualizations.

**Key Functions:**

* `plot_metagame_evolution_interactive(...)`: Renders a line chart tracking metagame share over time, marking extinction events with red crosses.
* `plot_matchup_network(...)`: Uses NetworkX to build an interactive nodes-and-edges graph where edge weights represent win rates and node sizes represent historical presence.
* `plot_metagame_scatter(...) & plot_head_to_head_radar(...)`: Powers the Streamlit dashboard visuals, comparing Meta Scores, Power Scores, and Frequency metrics.

___

## Domain: Tournament (`src/tournament/`)

### `monte_carlo.py`

**Purpose:** The high-performance computational core that simulates thousands of tournament brackets.

**Key Components:**

* `_mc_worker(...)`: Parallelized worker utilizing NumPy to run seeded Swiss brackets. It features Heuristic Rematch Prevention to stop players facing the same opponent twice, Parabolic Tie Convergence mapping time limits, and X-3 Drop Logic simulating player attrition.
* `run_monte_carlo_analytics(...)`: Manages the multiprocessing pool to aggregate total Top Cut appearances, Day 2 conversion rates, and overall event win probabilities.

### `solver.py`

**Purpose:** The static engine that enforces custom metagame limits and establishes predictive baselines.

**Key Functions:**

* `calculate_empirical_baseline(...)`: Derives the live metagame share by evaluating raw match volumes with Laplace smoothing.
* `resolve_meta_constraints(...)`: A vectorized water-filling algorithm that strictly enforces user-provided exact, minimum, and maximum boundaries while proportionately distributing the remaining probability mass.
* `predict_best_decks(...)`: Orchestrates the static recommendation flow, generating foundational metrics like `power_score`, `frequency_score`, and `base_meta_score`.

---

### Domain: Interfaces (`src/ui/`)

## `app.py`

**Purpose:** The interactive Streamlit dashboard serving as the user-facing client.

**Key Functions & UI Elements:**

* `parse_limitless_html(...)`: Native HTML parsing utility that extracts deck archetypes and player counts from raw Limitless Labs export files to pre-populate Custom Constraints.
* **Dual-Perspective Dashboard:** A dynamic UI component toggling data views between Individual EV ("Player/Micro") and Bracket Dominance ("Archetype/Macro").
* **Head-to-Head Comparator:** A visual section utilizing Pandas `Styler` to color-code win rates and a Plotly Radar Chart (`plot_head_to_head_radar`) to contrast stats directly between a Baseline and Challenger deck.

## `cli_args.py`

**Purpose:** Defines the argparse configuration for the CLI tools.

**Key Classes/Functions:**

* `Args`: A `NamedTuple` strongly typing the returned namespace.
* `parse_args()`: Validates arguments ensuring variables like `extinction_threshold` remain within 0.0 to 1.0, and defaults are properly mapped from `config.py`.