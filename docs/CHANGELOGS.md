## Commit a727312
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