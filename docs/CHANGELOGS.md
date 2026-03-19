## Latest Commit `?` (Mar 19, 2026)
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
    * **Head-to-Head Color Coding:** Integrated a Pandas `Styler` into the matchup comparator to natively highlight favorable ($\ge 55\%$) matchups in green and unfavorable ($\le 45\%$) in red.
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
- **HTML Fallback Fix:** If a file is misnamed, the scraper now successfully falls back to extracting both the deck name and the exact format directly from Limitless's HTML infobox tags.
- **Rogue Data Exclusion:** Hard-banned the generic **"Other"** archetype from being processed. This prevents aggregated, non-cohesive rogue decks from artificially skewing the K-Means clustering and Rock-Paper-Scissors cycle detection.
- **Architectural Alignment:** Removed premature data filtering from the scraper. The responsibility of filtering out decks with low match counts is now correctly delegated to the central `data.py` pipeline.

-----

## Commit `48fa2c3` (Nov 12, 2025)
### feat(analysis, config, docs): Vectorize tier list, centralize constants, and perform repository cleanup

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
- **I/O Cleanup:** Updated `.gitignore` or build scripts to manage and exclude excessive `/output/` directories and artifacts created during new simulation runs.
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

| Category | Line | Change Description |
| :--- | :--- | :--- |
| **Bug Fix (Serialization)** | (Lines 230-240) | **Fixed `TypeError: Object of type ndarray is not JSON serializable`** by calling `.tolist()` on the `similarity` variable before passing it to `json.dump`. |
| **Code Hygiene** | (Lines 230-240) | **Resolved unused variable warning** by adding explicit code to save the calculated `cycles` and `similarity` variables to `matchup_cycles.json` and `deck_similarity.json` respectively. |



### 2\. `src/plotting.py`

| Category | Line | Change Description |
| :--- | :--- | :--- |
| **Code Hygiene (Naming)** | (Line 215, various) | **Renamed capitalized variable `G` to `graph`** throughout the `plot_matchup_network` function to adhere to Python's lowercase variable convention. |
| **Code Hygiene (Unused)** | (Line 217) | **Removed the unused local variable `deck_to_idx`** from `plot_matchup_network`. |
| **Code Hygiene (Shadowing)** | (Line 54, various) | **Renamed local variable `annotations` to `annotations_list`** within `plot_metagame_evolution_interactive` to avoid shadowing the imported name `go.layout.Annotation`. |



### 3\. `src/analysis.py`

| Category | Line | Change Description |
| :--- | :--- | :--- |
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
    - Refactored complex, unreadable code (atleast IMHO) (such as the Bayesian win-rate
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