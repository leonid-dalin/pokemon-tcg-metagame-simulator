## ## Commit `[idk]` (Apr 04, 2026)
### 🚀 Infrastructure & UX: Redis Migration and SSE Streaming

This update finalizes the architectural decoupling of the simulator, transitioning from synchronous polling to a reactive, event-driven communication model.

[Small Rant]

As I’m journalling these changes, I want to mention the absolute struggle **I** had with the "Network Blip" error, as I just couldn't move around it. The system was on a chokehold with the error:

> Network blip detected due to high CPU load. Reconnecting (Attempt X/10)...

... and it simply refused to communicate the status of the data solver. At first, I blamed the SQLite file locks, as I assumed the `Huey` worker hogs it and causes the FastAPI server to stall. Then, when I introduced the Rust engine, I suffered from literal over-efficiency 💀 as it didn't hesitate in the slightest to 100% all of my CPU cores. I thought Python’s FastAPI runs on a single-threaded async event loop, maybe the engine was starving the API of the 'tiny' CPU slices it needs to keep the connection alive or something.

To solve this, I went out of my way and performed two major architectural changes:
- Migrated SQLite → Redis so that the task broker and progress data from RAM bypass disk I/O bottlenecks
- Forcing Rust to yield control via `py.allow_threads` so the API could 'breathe' during all the heavy math

**And the problem still persisted.** So how come I'm under high CPU load with several caps on? I `docker-compose up` and look at the calls made, and then I noticed.

The real culprit was a single unhandled sentinel class. `EmptyData`.
Or rather my assumption that `EmptyData` === `None` (Let me explain.)

When Streamlit connected to the API to ask for progress, the background worker would sometimes be just a millisecond away from starting. In that tiny window, `huey.storage.peek_data()` doesn't return **None,** it returns its `EmptyData` object. Because I assumed it would be either bytes or None, my code was trying to run `.decode('utf-8')` on a Python class object (`EmptyData`). And yeah, it caused an unhandled exception that instantly severed the HTTP connection. So bothersome and frustrating. Anyway. I'll leave you with the Patch Notes as per usual.


#### **📡 Server-Sent Events (SSE) Implementation**
* **Persistent Connections:** Replaced the former polling loop in the UI with a single, persistent HTTP stream (`/api/v1/tasks/{task_id}/stream`).
* **Real-Time Progress:** The UI now receives granular updates directly from the Huey worker, allowing for accurate ETA calculations and a smooth progress bar.
* **Keep-Alive Heartbeats:** Implemented an unconditional heartbeat mechanism in the FastAPI event loop to prevent network timeouts during intensive 1M+ iteration Monte Carlo runs.
* **Numpy-Safe Serialization:** Integrated a custom JSON encoder to handle `numpy.float` types, preventing silent stream crashes during data transmission.

#### **🧠 Redis Broker Integration**
* **Memory-First State:** Migrated the Huey task queue from SQLite to Redis. This eliminates the database file-locking bottlenecks that previously caused "Silent Stalls" during high-concurrency writes.
* **Pydantic Data Integrity:** Updated the `PredictionRequest` schema to prevent the silent erasure of `job_id` and `tournament_style` fields during API ingestion.

#### **⚖️ Resource Orchestration**
* **GIL Bypass & Threading:** The Rust `tcg_engine` now explicitly releases the Python Global Interpreter Lock (GIL), allowing the FastAPI server to breathe while the CPU is under 100% load.
* **Deferred Engine Initialization:** Moved the Rayon thread-pool initialization inside the worker task to prevent deadlocks caused by Linux `fork()` mechanics.
* **Granular Chunking:** Inverted the chunking logic to ensure regular progress updates (every 10k iterations) regardless of the total simulation size.

---

## Commit `7aff00f` (Apr 02, 2026)
### 🚀 major: Rust integration for high-speed Monte Carlo simulations

This update replaces the Python/NumPy tournament bracket engine with a fully compiled, multithreaded Rust extension (`tcg_engine`). Faster execution times! 🥳

#### **🦀 Rust & PyO3 Integration**
* **GIL Bypass & Rayon Parallelization:** The `_mc_worker` has been completely rewritten in Rust. By utilizing `PyO3` and the `Rayon` crate, the engine now completely bypasses the Python Global Interpreter Lock (GIL), stealing work across all available CPU cores dynamically and efficiently.
* **Zero-Cost Abstractions:** Transitioned from Python array slicing (`np.where`) to native Rust `Vec` and in-place memory mutations. This eliminates the massive garbage collection overhead previously incurred during millions of simulated match loops.
* It maintains perfect Parabolic Tie Convergence math, X-3 Drop Logic, OWP sorting, and 1-Deep Lookahead heuristic pairings, seeded securely for perfect reproducibility.

#### **🛠️ Infrastructure & Stability**
* Updated the deployment architecture to utilize `maturin`, seamlessly compiling `.whl` binaries directly inside the Linux Docker container during build time.

---

## Commit `39d482d` (Mar 31, 2026)
### 🚀 feat(pipeline): implement autonomous Limitless TCG live scraper, Pydantic matrix validation, and Huey cron scheduling

This update completely eliminates the need for manual HTML file downloads from Limitless. The simulator is now able to fetch, normalize, validate, and ingest live Limitless TCG data automatically, as long as it knows the URLs.

#### **🕸️ Live Web Scraping (`scraper.py`)**
* **Direct HTML Fetching:** Replaced the local file-reading logic with Python's `requests` library. The scraper now dynamically fetches live matchup pages using a predefined list of active format URLs (`urls.py`). The offline/manual version will become **legacy** for now.
* **Memory-Based Soup Parsing:** `scrape_matchup_soup` was refactored to parse the HTML and extract the W-L-T matrix directly in memory, bypassing the need for an intermediate `/data/matchups/` storage folder.

#### **🛡️ Thermodynamic Data Validation (`models.py`)**
* **Strict Pydantic Bouncer:** Implemented a rigid Pydantic schema (`ScrapedMatrix`) that acts as a firewall between Limitless TCG and the `ea_input.json` matrix.
* **Mathematical Purity Enforcement:** The `@model_validator` instantly aborts the data write if the scraped data violates thermodynamic purity (e.g., if win rates fall outside the `0.0-1.0` boundary, or if a mirror match diagonal is anything other than exactly `0.5`). 

#### **⏰ Automated Background Pipeline (`queue.py`)**
* **Huey Cron Scheduler:** Built a native `@huey.periodic_task` that wakes up at 2:00 AM every day. It executes the web scraper, runs the Pydantic validation, and updates the `ea_input.json` matrix asynchronously.
* **Zero UI Interruption:** Because this runs entirely within the decoupled Huey background worker (`tcg_tasks.db`), the daily data refresh happens invisibly and does not block or freeze the Streamlit dashboard for end-users.

---

## Commit `088e98f` (Mar 30, 2026)
### 🚀 major: Decoupled Asynchronous Architecture & Containerization
This update transitions the project from a synchronous, locally-bound Streamlit application to a production-ready, three-tier distributed system. 
At least conceptually. 

#### **🏗️ Architecture & Decoupling**
* **FastAPI Gateway:** Extracted the core simulation engines (`solver.py` and `monte_carlo.py`) behind a high-performance REST API. 
* **Huey Background Worker:** Offloaded the heavy NumPy Monte Carlo calculations to a dedicated background task queue (`huey`). This completely resolves the Windows `multiprocessing.Pool` caching bugs by isolating the compute pool from the Streamlit UI thread.
* **SQLite Broker:** Configured `Huey` to use a local SQLite database (`tcg_tasks.db`) as its message broker, achieving an asynchronous task queue without the bloat of requiring a local Redis or RabbitMQ installation.
* **Pydantic Contracts:** Implemented strict data validation for tournament generation payloads via Pydantic models.

#### **🐳 Docker Orchestration**
* **Docker Compose:** Containerized the entire suite into three isolated microservices (`api`, `worker`, and `ui`). The system can now be booted on any OS with a single `docker compose up -d` command, ensuring perfect environment parity between Linux and Windows development machines.
* **Thin Client UI:** Streamlit has been refactored into a "dumb" thin client. It now dispatches HTTP `POST` requests to the API and runs a non-blocking polling loop until the background worker completes the tournament bracket calculations.

---

## Commit `5b74ae3` (Mar 24, 2026)
### 🚀 major: transition to high-fidelity Evolutionary Game Theory (EGT) solver
This massive architectural update transitions the project from a heuristic simulation to a rigorous Quantal Response Equilibrium (QRE) solver, achieving point-wise convergence to a true Evolutionary Stable State (ESS).

#### **🧬 Core Engine & Mathematics**
* **Multiplicative Weights Update:** Upgraded the Multiplicative Weights Update (MWU) to an **Optimistic** variant using gradient extrapolation ($2 \times payoffs - last\_payoffs$). This dampens zero-sum limit cycles, forcing the metagame to spiral into the Nash Equilibrium rather than orbiting it endlessly.
* **Replicator-Mutator Equation:** Replaced random rogue-deck reintroductions with a continuous ambient entropy regularizer ($MUTATION\_RATE = 1e-4$). This prevents gradient collapse and ensures the final equilibrium is globally unexploitable.
* **Stochastic Volatility:** Re-integrated `noise_scale` into the MWU engine, injecting Gaussian noise into the fitness growth exponent to simulate imperfect player information and local metagame turbulence. Defaults at `0.0` for the unforeseeable future.

#### **⚖️ Data Purity & Scraper**
* **Zero-Sum Match Equity:** Overhauled `scraper.py` to calculate win rates using the standard Elo formula: $(Wins + 0.5 \times Ties) / Total\ Matches$. This eliminates "energy leaks" where ties were previously penalized as double-losses, ensuring a mathematically perfect 100% zero-sum matrix.
* **Asymmetrical Matrix Centering:** Updated Expected Value (EV) calculations to use $payoffs - avg\_payoff$, correctly centering the gradient plane for asymmetrical TCG matchup data.

#### **⚡ Performance & Stability Sensors**
* **Kinetic & Nash Convergence:** The engine now mandates a dual-stability check: **Kinetic Stability** (frequency shifts < $5e-5$) and **Game-Theoretic Stability** (max unexploited advantage < $0.25\%$).
* **Artificial Floor Removal:** Completely removed `MIN_GENERATIONS_PROP`, allowing the engine to break and exit the moment the stability window is mathematically achieved. For complex 33-deck metas, this reduced convergence time to ~6,600 generations (0.25s).

#### **🛡️ Infrastructure & CLI**
* **Safe Attribute Ingestion:** Modified `main.py` to use `getattr()` when extracting CLI arguments like `min_games`, providing a safe fallback to `config.py` constants and preventing `AttributeError` crashes.
* **Hyperparameter Sync:** Re-tuned `SELECTION_PRESSURE` to `1.0`, leveraging OMWU’s stability to achieve blazing-fast descent without gradient explosion.

---

## Commit `b4b8c53` (Mar 24, 2026)
### feat: interactive Plotly analytics, error boundaries, and UI/UX refinements

This update drastically enhances the visual storytelling of the dashboard using Plotly and patches several silent failure states in the data ingestion pipeline.

#### **📊 Advanced Visualizations (`plotting.py`)**
* **Metagame Scatter Plot:** Introduced a 2D interactive scatter plot underneath the main data table. It visually maps a deck's Power Score against its Frequency Score, with the bubble's radius representing its overall Meta Score. Hover tooltips and static labels allow for rapid format comprehension.
* **Negative Power Score Toggle:** Added a sidebar toggle (`allow_negative_power`) that strictly filters out unviable decks (Power Score < 0) from the Scatter Plot by default. This keeps the visual plane clean, while allowing users to opt in to seeing the "graveyard" of negative EV decks.
* **Head-to-Head Radar Chart:** Complimenting the static text comparisons with a dynamic Plotly Spider/Radar chart. The chart automatically scales its axes to match the selected tournament structure (e.g., adding Day 2/Top 8 spokes only if applicable) and normalizes coordinates while preserving true metric values in the hover text.

#### **🛡️ Stability & Error Handling**
* **Limitless HTML Error Catching:** Wrapped the `Import Limitless Labs HTML` logic in a hard `try...except` boundary. Instead of the UI silently refreshing and hiding the error when encountering malformed JSON or unrecognized deck strings, the app now explicitly halts and prints the Python stack trace directly to the UI for easy debugging.
* **Persistent Success States:** Fixed a bug where successful HTML import alerts (`st.success`) would vanish in milliseconds due to the forced Streamlit rerun. Messages are now temporarily cached in `st.session_state` to survive the lifecycle refresh.

#### **💅 UI/UX Polish**
* **Matchup Table Visibility:** Fixed an issue where the background-colour styler on the Head-to-Head matchup table died in Dark Mode. Boosted rgba opacity to `0.55` and forced `font-weight: bold; color: #ffffff;`.
* **Deprecation Maintenance:** Replaced all instances of the deprecated `use_container_width=True` argument in `st.dataframe` and `st.plotly_chart` with the desired `width="stretch"` standard.
* **Radar Coordinate Bounding:** Enforced a visual `0.0` rendering floor on the Radar chart to prevent negative Power Scores from inverting the geometry of the polygon, while keeping the true negative text mapped to the user tooltips.

---

## Commit `5f7c7f1` (Mar 24, 2026)
### refactor: formalize tournament EV logic, stabilize UI state, and fix mathematical clipping

This update represents a fundamental shift in the recommendation engine, while resolving several critical stability bugs and mathematical floor errors. Updated the current `INPUT_DATA` to the Limitless TCG's stats of Mar 24, 2026 of ASC format.

#### **📊 Tournament EV & Dynamic Logic Overhaul**
* **Context-Aware EV:** The engine now dynamically switches primary metrics based on player count. Large-scale tournaments (e.g., 2,000+ players) prioritize **Day 2 Conversion** to mitigate Top 8 variance. Smaller events pivot to **Top 8/Top X Conversion**. This is very much still subject to change, and are a direct preference of mine's.
* **EV Delta Metric:** Introduced a relative performance stat that shows the "Equity Drop-off" compared to the #1 ranked deck, providing better context for the other choices.
* **Mathematical Floor Fix:** Updated `solver.py` to use `np.minimum(100.0)` instead of a hard `np.clip`. This allows for negative scores in the framework, accurately representing sub-baseline rogue decks.

#### **🛡️ Critical Stability & Crash Prevention**
* **Variable Alignment:** Fixed a fatal crash where `best_wr` was pointing to the deleted `recs_sorted_by_power` list.
* **Empty Meta Handling:** Sanitized `tab_rec` logic to prevent `IndexError` and `StreamlitAPIException`. The app no longer crashes if no decks exceed the Meta Score threshold (balanced meta) or if `st.columns()` receives a zero count.
* **Strict Column Ordering:** Implemented a `final_column_order` array to prevent Streamlit from silently dropping dynamic archetype columns in the Macro view.

#### **💅 UI/UX Refinement & Precision**
* **Bulletproof Formatting:** Enforced **2-decimal precision** (`f"{val:.2f}"`) across all metrics (Power, Freq, Meta) to ensure visual alignment and "column hugging."
* **Naming Convention:** Renamed "Exp. Avg. WR %" to **"Power Ranking (Day 1)"** to create a logical pair with "Power Ranking (Day 2)."
* **Noise Reduction:** Added a strict **1% field share cutoff** for threat lists to filter out statistically irrelevant rogue decks.
* **Default State:** The `Import Limitless Labs HTML` expander now defaults to "Open" instead of `Active Custom Constraints`.

#### **🔬 Architectural Diagnostics**
* **Multiprocessing:** Documenting the root cause of the `No runtime found` cache warning. Windows `multiprocessing.Pool` spawns child processes that re-evaluate `app.py`, triggering `@st.cache_data` decorators outside the main Streamlit thread. A problem for future I.

---

## Commit `6401713` (Mar 20, 2026)
### refactor: implement vS Meta Score, overhaul Swiss bracket engine, and redesign the app dashboard to be more intelligent

This update fundamentally shifts how the static baseline is evaluated, moving away from fabricating Swiss tiebreakers (SoS/OMW) before the bracket runs, and instead adopting the rigorous data philosophies already pioneered by [Vicious Syndicate](https://www.vicioussyndicate.com/).

* **Adopted vS Meta Score Logic:** `solver.py` now evaluates the Day 1 baseline using Vicious Syndicate's 2D coordinate system. 
    * **Power Score (0-100):** A normalized measure of a deck's Expected Win Rate against the field.
    * **Meta Score (0-100):** The simple average of a deck's Power Score and its Frequency Score (Popularity).
* **Strict Win-Rate Tiers:** Abandoned 0-100 normalized composite scores for tier assignments. Tiers are now ruthlessly assigned based on raw Expected Win Rate against the predicted field (S-Tier $\ge$ 52%, A-Tier $\ge$ 50%, B-Tier $\ge$ 47%).
* *Credit:* I'd like to extend a massive thank you to the [Vicious Syndicate](https://www.vicioussyndicate.com/drr/faq-data-reaper-report/) team for providing the gold standard in TCG data analytics. Their methodology for mapping "format dominance" versus "pilot profitability" directly inspired this refactor. It was long due, but I wanted to experiment on my own before I'd adopt a proven logic.

#### **Monte Carlo Engine Optimizations**
* **1-Deep Lookahead Heuristic:** Fixed a critical flaw in the Swiss pairing logic. Basically, there was a possibility for identical re-pairings (I observed it in ties). The engine now checks `opponents_history` and performs adjacent-table swaps to prevent players from immediately rematching the same opponent, without the crippling $O(N^3)$ performance cost of a full Blossom algorithm.
* **BETA: X-3 Drop Simulation:** Added a UI toggle to simulate real-world player attrition. If enabled, the engine will purge players from the active bracket once they accumulate 3 losses, accurately altering the tiebreaker math for the bubble.
* **Performance Boost:** Stripped redundant Opponent's Match Win (OMW) array aggregations out of the Day 1 Swiss loops. OWP is now only calculated exactly when needed (Top Cut sorting), significantly reducing NumPy overhead.

#### **UI/UX Intelligence Dashboard**
* **Tabbed Actionable Intelligence:** Replaced the flat recommendation list with a 3-tab layout: **🔥 Top Recommendations** (Sorted by Power Score), **🚨 Top Threats** (Sorted by Meta Score), and **🚫 Decks to Avoid** (Negative EV).
* **Threat Cross-Referencing:** Recommended decks now explicitly display a colour-coded matrix showing exactly how they fare against the Top 3 "Meta Dictator" threats.
* **Day 2 Expected Win Rate:** The UI now calculates a completely isolated Expected Win Rate against the *condensed* Day 2 meta-share, identifying "Day 2 Predators" that farm the top tables.

---

## Commit `4820359` (Mar 20, 2026)
### refactor: domain-driven restructure, massive performance optimization, and mathematical fidelity overhaul

Another major update that introduces a new strict domain-driven directory structure to separate the evolutionary engine from the tournament solver engine, alongside critical bug fixes that drastically increase simulation speed and mathematical purity. At least on paper.

#### **Architectural Restructure**
* **Domain-Driven Layout:** Reorganized `src/` into distinct domains: `core/`, `evolution/`, `tournament/`, and `interfaces/`. This physically separates the long-term game theory mechanics from the static Monte Carlo bracket predictor, preventing cross-contamination and clarifying the codebase's dual purpose. The classes should be easier to follow and understand as a result.
* **Unified Entry Points:** `app.py` and `cli_args.py` moved to `interfaces/`. `main.py` and `scraper.py` remain at the root as top-level orchestrators for accessibility reasons. Imports have been globally updated to support absolute pathing.

#### **Performance & Optimization (1,000,000 gens in ~18s)**
* **Vectorized Array Splicing:** Completely replaced the slow double `for` loops in `analysis.py`'s similarity matrix reconstruction with a single, highly-optimized NumPy index slice (`np.ix_`).
* **Swiss Worker Array Slicing:** Replaced sequential `np.where` boolean checks in the `_pure_swiss_worker` with high-speed `[0::2]` vs `[1::2]` array slicing for pairing logic.
* **Multiprocessing Pool Leak Fixed:** Fixed a critical memory leak in `evolution/engine.py` where a new `mp.Pool()` was instantiated and destroyed every generation. The pool is now initialized exactly once per simulation run and passed down safely.

#### **Mathematical Purity & Bug Fixes**
* **Parabolic Tie Convergence:** The `_championship_series_worker` now accurately executes the advertised BO3 match-point bleed via the formula $P_{tie} = T_{global} \times 4P(1-P)$, properly awarding 1 point for ties and accurately suppressing global OMW% by ~1–1.5 points.
* **Purged Phantom Confidence:** Fixed `data.py` assigning a default `match_count` of 100 to missing matchups. Missing data now defaults to 0, preventing the Bayesian beta distribution from inventing false statistical confidence. This was initially done for EGT where not enough matchups were recorded.
* **Restored Replicator Purity:** Removed the global `mutation_floor` broadcast in `reintroduce_extinct_decks`. Decks now naturally decay to a true `0.0` frequency unless explicitly selected for reintroduction, allowing the engine to find mathematically perfect Nash Equilibrium.
* **Removed Win-Equivalent Distortions:** Stripped arbitrary float modifiers from tournament placements in the evolutionary engine. Fitness is now derived purely from mathematically normalized match points.
* **Removed Data Falsification:** Deleted the `gaussian_filter1d` block at the end of the simulation loop, ensuring all post-analysis tools evaluate the raw, genuine stochastic output of the engine.
* **Synchronized Tier Thresholds:** `generate_final_state_tier_list` now dynamically imports the global `TIER_*_THRESHOLD` variables, preventing conflicting tier assignments between the final state and all-time lists. Forgot about this hardcode.

---

## Latest Commit `04a47ba` (Mar 19, 2026)
### refactor(core): fix Ultimate Score math, vectorize engine, optimize clustering, and overhaul UI POV

#### **Mathematical & Logic Fixes**
* **`app.py`:** 
    * **Fixed Ultimate Score:** Replaced the flawed "Double Min-Max" scoring system with a rigorous **Z-Score Standardization** model mapped to a sigmoid curve (`np.tanh`).
    * **Dual-Scoring Architecture:** The engine now simultaneously calculates both `score_player` (Individual EV) and `score_archetype` (Macro Metagame Impact). Decks with massive format presence (like Gholdengo) are appropriately rewarded in the Archetype view, while high-converting rogue decks shine in the Player view.
* **`data.py`:** 
    * **Clustering Overhaul:** Applied `StandardScaler` to the win-rate matrix before passing it to K-Means. This forces the algorithm to cluster decks by their strategic archetype "shape" rather than raw win-rate magnitude.
    * **Silhouette Optimization:** `cluster_decks_by_matchup_profile` now accepts `n_clusters="auto"`, dynamically calculating the Silhouette Score to pick the mathematically optimal number of tiers/clusters (between $K=2$ and $K=6$).

#### **Performance Improvements**
* **`predictor.py`:** 
    * **Vectorized Constraints:** Completely replaced the inner `for` loops in the `resolve_meta_constraints` water-filling algorithm with pure NumPy boolean masking, massively speeding up constraint resolution for large fields.
* **`monte_carlo.py`:**
    * **Boolean Micro-Optimization:** Replaced dual floating-point array comparisons with an inverted boolean mask (`~(p1_wins | p2_wins)`) inside the hottest loop of the Parabolic Tie Convergence engine, saving redundant memory allocations across millions of iterations.

#### **UI/UX Updates & Bug Fixes**
* **`app.py`:**
    * **Dynamic Perspective Toggle:** The "Odds Perspective" radio button now sits directly above the dataframe, allowing users to instantly swap the data sorting and Top Recommendations between Individual EV and Macro Impact without re-triggering the Monte Carlo engine.
    * **Head-to-Head Colour Coding:** Integrated a Pandas `Styler` into the matchup comparator to natively highlight favorable ($\ge 55\%$) matchups in green and unfavorable ($\le 45\%$) in red.
    * **Tooltips Added:** The Streamlit interactive dataframe now utilizes `st.column_config` `help` parameters, injecting native hover-tooltips for all table headers (e.g., explaining SoS, OMW, and composite scores).
    * **Layout & State Fixes:** Widened the custom constraint columns to prevent the delete button from clipping. Added defensive dictionary key initialization (`score_player`, `score_archetype`) to prevent Streamlit caching `KeyError`s during hot-reloads.

---

## Commit `6100540` (Mar 19, 2026)
### feat(sim): implement tournament equity engine with parabolic tie convergence

#### **Core Logic & Engine Updates**
* **`monte_carlo.py` (New File):** * Created a high-performance, parallelized bracket engine.
    * Implemented **Parabolic Tie Convergence (BETA)** which simulates match-point decay in Swiss rounds based on matchup closeness ($P_{tie} = T_{global} \times 4P(1-P)$).
    * Added **True Bracket Seeding**; Top Cut now pairs 1v8, 2v7, etc., rather than arbitrary pairings.
* **`predictor.py`:** * Vectorized the calculation of **Strength of Schedule (SoS)** and **Opponent's Match Win % (OMW)**.
    * Added **Undefeated Probability** metrics.
    * *Note:* Heuristic scores are now generated as a baseline, intended to be overwritten by the MC engine in the final UI.
* **`simulation.py` & `simulation_config.py`:**
    * Integrated **Official TPCi Variant #5** logic. The simulator now maps player counts to specific Day 1/Day 2 round counts and match-point cutoffs.
    * Added `tournament_style` toggles (`pure_swiss` vs `championship_series`).

#### **Complete Overhaul of the `app.py`: Logic & Purpose**

#### **Architectural Shift**
* **Deprecation of Replicator Dynamics:** Removed the `InferenceMode` ("casual" vs "pro") and the `find_evolutionary_stable_state` logic. The app no longer predicts "meta evolution" but instead performs "tournament equity" simulation.
* **Integration of Monte Carlo Engine:** Introduced `run_monte_carlo_analytics` as the primary engine, simulating thousands of individual tournament brackets (Swiss + Top Cut) to find conversion rates.
* **Official TPCi Structures:** Replaced static round counts with `get_variant_5_structure`, which automatically sets rounds, match-point cuts, and top-cut sizes based on official Play! Pokémon standards.

#### **Mathematical Refinements**
* **Ultimate Scoring System:** Replaced the old heuristic "Score" with a Min-Max normalized value (0–100) based on four tournament pillars: Win Rate, Day 2 Conversion, Top 8 Conversion, and Win Probability.
* **Parabolic Tie Convergence (BETA):** Implemented a new model for BO3 time-outs, allowing match points to "bleed" via ties (1 point) rather than forcing binary win/loss (3/0 points) outcomes.
* **Vectorized Swiss Metrics:** Shifted calculation of SoS (Strength of Schedule) and OMW (Opponent's Match Win %) to vectorized NumPy operations for speed.

#### **UI/UX & Frontend Overhaul**
* **Constraint Management:** Constraints moved from a vertical list to a **2-column grid** inside an expander, drastically reducing vertical clutter.
* **Execution Debugger:** Replaced the simple spinner with an `st.status` **Execution Console**, providing real-time feedback on data loading, water-filling, and parallel core progress.
* **Interactive Dashboard:** Replaced the "Full Inferred Meta" text list with a sortable, interactive `st.dataframe` featuring **Archetype vs. Player** perspective toggles.
* **Head-to-Head Comparator:** Added a "A vs. B" tool that allows users to compare a primary deck/archetype against another across the entire predicted field.
* **Dynamic Recommendation Badges:** Recommendations now include specific paragraph-style tags (e.g., `🏆 Best to Win Event`, `🛡️ Safest Day 2`).
* **Limitless HTML Ingestion:** Added a native parser to extract metagame shares directly from Limitless Labs HTML exports.

-----

## Commit `2ee8ade` (Mar 18, 2026)
### refactor(scraper): overhaul data extraction pipeline, enforce precision, and improve hygiene

Another overhaul to the `scraper.py` utility to ensure the data fed into the simulation engine is highly accurate, properly formatted, and strictly typed. 

### 🚀 Feature & Performance Improvements
- **Strict Naming Conventions:** The scraper now uses _slightly faster_ string splitting to extract metadata directly from the `Archetype - Standard - Website` file naming format.
- **Enhanced Mathematical Precision:** Win rates exported to `ea_input.json` and CSVs are now locked to a 4-decimal precision (`.4f`) from (`.2f`). 

### 🐛 Bug Fixes & Data Hygiene
- **HTML Fallback Fix:** If a file is misnamed, the scraper now successfully falls back to extracting both the deck name and the exact format directly from Limitless's HTML info-box tags.
- **Rogue Data Exclusion:** Hard-banned the generic **"Other"** archetype from being processed. This prevents aggregated, non-cohesive rogue decks from artificially skewing the K-Means clustering and Rock-Paper-Scissors cycle detection.
- **Architectural Alignment:** Removed premature data filtering from the scraper. The responsibility of filtering out decks with low match counts is now correctly delegated to the central `data.py` pipeline.

-----

## Commit `48fa2c3` (Nov 12, 2025)
### feat(analysis, config, docs): Vectorize tier list, centralize constants, and perform repository clean-up

Improves the metagame analysis pipeline by optimizing performance, ensuring full configuration transparency, and performing necessary repository maintenance.

### 🚀 Feature & Performance Improvements
- **Performance:** Implemented a full **vectorization** of the deck **consistency** calculation in `analysis.py`. This significantly speeds up tier list generation by moving from slow Python loops to efficient NumPy operations.
- **Centralized Configuration:** All numerical tuning parameters for the tier list are now centralized in `config.py`:
    - **Composite Weights:** Added `COMPOSITE_SCORE_WR_WEIGHT`, `_PRESENCE_WEIGHT`, and `_CONSISTENCY_WEIGHT`.
    - **Tier Thresholds:** Added `TIER_S_THRESHOLD`, `_A_THRESHOLD`, `_B_THRESHOLD`, and `_C_THRESHOLD`.
- **Tier List Tuning:** Updated default tier thresholds (e.g., S-Tier from 0.75 to 0.90) to enforce a stricter, more competitively realistic performance distribution.

### 🐛 Bug Fixes & Refactoring
- **Code Quality:** Eliminated the "referenced before assignment" linter warning in `analysis.py` by replacing a complex list comprehension with an explicit `for` loop.
- **Numerical Stability:** Formalized the hardcoded epsilon values (`1e-6`, `1e-9`) into explicit constants (`CONSISTENCY_MEAN_EPSILON`, `CONSISTENCY_STD_EPSILON`) in `config.py`.
- **Code Hygiene:** Removed the unused local variable `num_gens` from `analysis.py`.

### 🧹 Chore & Maintenance
- **I/O Clean-up:** Updated `.gitignore` or build scripts to manage and exclude excessive `/output/` directories and artefacts created during new simulation runs.
- **Documentation Hygiene:** Sanitized and refined `CHANGELOG.md` to ensure clarity and remove redundant or conversational language.

-----

## Commit `458ded8` (Nov 12, 2025)
### refactor(config, analysis): Formalize consistency epsilons and resolve linter warnings

Refactors the tier list generation logic to improve code quality, resolve linter warnings, and formalize ""magic numbers"" into explicit constants.

Key Changes in `analysis.py`:
- **Fixed scoping warning:** Replaced a complex, warning-prone list comprehension for deck consistency with a clear, explicit `for` loop to ensure correct assignment of the local variable `std_val`.
- **Removed unused variable:** Eliminated the unused local variable `num_gens`.
- **Refactored hardcoded values:** Replaced the hardcoded numerical stability values (`1e-6` and `1e-9`) with imported, named constants.

Key Changes in `config.py`:
- **Added constants:** Introduced `CONSISTENCY_MEAN_EPSILON` (1e-6) and `CONSISTENCY_STD_EPSILON` (1e-9) to promote clarity and centralize configuration for the consistency metric calculation.

-----

## Commit `8e97897` (Nov 12, 2025)
### fix(analysis, cli, plotting): Resolve stability issues, serialization errors, and improve core logic

This commit implements a series of fixes across the core simulation files to address static analysis warnings, a runtime serialization error, and a critical issue (kinda) in how strategic deck similarity is calculated.

Fixes:
- Resolves 'Local variable value is not used' warnings in main.py and plotting.py.
- Resolves 'TypeError: Object of type ndarray is not JSON serializable' by correctly converting NumPy arrays to lists before writing to JSON.
- Resolves naming convention and shadowing warnings in plotting.py.
- Improves the robustness of deck similarity by ensuring the comparison is based on the deck's full strategic profile.


---

### 1\. `src/main.py`

| Category                    | Line            | Change Description                                                                                                                                                                        |
|:----------------------------|:----------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Bug Fix (Serialization)** | (Lines 230-240) | **Fixed `TypeError: Object of type ndarray is not JSON serializable`** by calling `.tolist()` on the `similarity` variable before passing it to `json.dump`.                              |
| **Code Hygiene**            | (Lines 230-240) | **Resolved unused variable warning** by adding explicit code to save the calculated `cycles` and `similarity` variables to `matchup_cycles.json` and `deck_similarity.json` respectively. |



### 2\. `src/plotting.py`

| Category                     | Line                | Change Description                                                                                                                                                       |
|:-----------------------------|:--------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Code Hygiene (Naming)**    | (Line 215, various) | **Renamed capitalized variable `G` to `graph`** throughout the `plot_matchup_network` function to adhere to Python's lowercase variable convention.                      |
| **Code Hygiene (Unused)**    | (Line 217)          | **Removed the unused local variable `deck_to_idx`** from `plot_matchup_network`.                                                                                         |
| **Code Hygiene (Shadowing)** | (Line 54, various)  | **Renamed local variable `annotations` to `annotations_list`** within `plot_metagame_evolution_interactive` to avoid shadowing the imported name `go.layout.Annotation`. |



### 3\. `src/analysis.py`

| Category              | Line              | Change Description                                                                                                                                                                                                                                                                                                                                                                                                                                     |
|:----------------------|:------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Logic Improvement** | (Around Line 290) | **Corrected `compute_deck_similarity` logic.** Changed the comparison profiles from the **reduced submatrix** of active decks (`win_matrix[np.ix_(active_indices, active_indices)]`) to the **full win-rate profiles** against the entire metagame (`win_matrix[active_indices, :]`). This ensures strategic similarity is measured based on a deck's complete profile (all $N$ matchups), not just the matchups against other currently active decks. |

-----

## Commit `a727312` (Nov 11, 2025)
### refactor(core): Full-stack optimization, caching, and architectural streamlining

This is a major squashed commit that introduces significant performance
enhancements, architectural simplifications, and bug fixes across the
entire application, from the Streamlit front-end to a new 3rd simulation mode known as the `Predictor`, 
and to the analysis logic.
`ea_input.json` is now based on the Snapshot took from Limitless's Mega Evolution Online Tournament Meta as of the 10th of November.

The primary goals of this refactor were:
1.  **Improve Significantly the Performance Gains**
    - Implemented caching for the "Pro" mode in the Streamlit app that's supposed to replicate 'Idealistic' (Worst Case scenario) situations
    - Vectorized all major bottlenecks in `analysis.py` and `predictor.py`.
    - Optimized core simulation loops by pre-calculating static configs.
2.  **Architectural Simplification**
    - **DELETED** the now-redundant `runtime_config.py`.
    - `main.py` now builds the canonical `SimulationConfig` directly from
      `cli_args.Args`, creating a single, clear data flow. The data crossposting started to make me nauseous.
3.  **Bug Fixes & Maintainability**
    - Resolved numerous linting errors, unused variables, and potential
      "referenced before assignment" bugs.
    - Refactored complex, unreadable code (at least IMHO) (such as the Bayesian win-rate
      list comprehension) into a cleaner, more maintainable blocks. I hope.
4.  **Enhanced Predictor Accuracy & Metrics**
    - Implemented advanced Swiss tournament metrics (SoS, OMW) and an 'undefeated probability' proxy
      within the predictor to provide more nuanced and tournament-relevant recommendations.
    - Updated the Streamlit UI to display these new metrics, especially focusing on a user's
      deck's chance to go undefeated, aligning recommendations with the goal of consistent winners.

---

### Component-Level Changes

#### 🚀 `app.py` (Streamlit Front-end)

* **(Perf)** Implemented `@st.cache_data` on a new `get_pro_meta()` function.
    This runs the expensive "Pro" mode simulation *once* on app load
    and caches the result. Subsequent "Pro" recommendations are
    now **instantaneous**.
* **(Refactor)** The app now passes the cached `fallback_meta_pro` array
    to `predict_best_decks`, enabling the new caching strategy.
* **(Feat)** Enhanced the 'Your Deck Performance' section to display
    the new 'undefeated probability' and other Swiss metrics like SoS and OMW,
    providing more actionable insights than the previous simple expected record.
* **(UX)** Improved dynamic deck input: 'Add Deck' and dropdowns filter out already selected decks.
    Added tabs for Recommendations/Avoid/Full Meta.

#### ⚡ `predictor.py` (Recommendation Engine)

* **(Perf)** Vectorized the `omw_values` (Opponent's Match Win)
    calculation. This removes a slow, nested Python loop and replaces it
    with a single, high-speed NumPy operation, dramatically speeding up
    predictions.
* **(Refactor)** `swiss_rounds_from_players` now uses the correct
    `math.ceil(math.log2(n))` formula instead of a hardcoded list.
* **(Perf)** `predict_best_decks` now accepts an optional
    `fallback_meta_pro` argument to allow for caching (used by `app.py`).
* **(Perf)** The "avoid" list is now derived from the *same* sort as the
    recommendations (by taking the tail), avoiding a redundant sort.
* **(Fix)** Corrected a minor bug in the `frontrunners` logic to
    properly index the `composite_scores` array.
* **(Feat)** Added calculation for `sos`, `omw`, `undefeated_probability`, and a new
    `composite_score` that incorporates Swiss metrics for more robust recommendations.

#### 🔬 `simulation.py` (Core Engine)

* **(Fix)** Removed unused imports (`Literal`, `Iterator`).
* **(Fix)** Fixed a critical "might be referenced before assignment" bug
    with `history_writer` by initializing it to `None` and adding
    safer checks.
* **(Fix)** Removed all unused local variables (`deck_to_idx`,
    `sample_interval`, `n`).
* **(Fix)** Replaced a dangerous, broad `except Exception: pass` block
    with a safer, logging-based `except Exception as e:
    logging.debug(...)` to prevent silently swallowing all errors.
* **(Refactor)** Refactored the complex, single-line Bayesian win-rate
    list comprehension in `_tournament_worker` into a clear,
    maintainable, multi-line `for` loop with identical performance.
* **(Refactor)** Removed the unused `win_matrix` parameter from
    `reintroduce_extinct_decks`.
* **(Perf)** `update_replicator_dynamics` now accepts the main `rng`
    generator instead of creating a new one on every call, improving
section
    performance and ensuring reproducible noise.
* **(Perf)** In `run_tournament_generation`, the static `task_config`
    dictionary is now created *outside* the loop, avoiding redundant
    allocations.
* **(Refactor)** Removed the "Soft Convergence" logic, simplifying the
    simulation flow and configuration. The simulation now runs with
    consistent parameters, and convergence is determined purely by the
    stability threshold.

#### 📊 `analysis.py` (Post-Simulation)

* **(Perf)** Fully vectorized `generate_all_time_tier_list`. The slow
    Python `for` loop that iterated over the entire history is gone.
    This function now stacks the history into a single NumPy array and
    uses `np.mean` and a single matrix-vector multiplication
    (`win_matrix @ freq_history.T`) to compute all-time payoffs,
    resulting in a massive speedup for large histories.

#### 📦 `main.py` & `runtime_config.py` (Architecture)

* **(Refactor)** **DELETED `runtime_config.py`**. This class was
    redundant and created a confusing, indirect configuration flow.
* **(Refactor)** `main.py` has been rewritten to be the single source of
    truth. It now constructs the canonical `SimulationConfig` dataclass
    *directly* from the `cli_args.Args` object. This simplifies the
    entire configuration pipeline.
* **(Feat)** `main.py` now correctly handles the `--predict` flag and
    associated arguments, piping them to `predict_best_decks`.

#### 🔡 `cli_args.py` (Command Line)

* **(Feat)** Added new arguments to `Args` and `argparse` to support
    the CLI prediction mode: `--predict`, `--players`, and `--meta`.

#### 🧹 `data.py` (Data Loading)

* **(Refactor)** Removed all unused `pydantic` models (`ArchetypeData`,
    `MatchupDetail`) and their dependencies. This slims down the
    project's dependencies and cleans up the data-loading module.