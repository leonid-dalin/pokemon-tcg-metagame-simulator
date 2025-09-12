#!/usr/bin/env python3
# simulation.py — Core metagame evolution engine with performance & dynamics enhancements
from __future__ import annotations
import time
import logging
import numpy as np
from typing import List, Tuple, Dict, Any, Literal, Optional, Iterator, Iterable
import csv
# Optional modules — gracefully degrade
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    tqdm = None
try:
    import multiprocessing as mp
    MULTIPROC_AVAILABLE = True
except ImportError:
    MULTIPROC_AVAILABLE = False
    mp = None
# Local import
from .config import *
from .data import safe_normalize
from .simulation_config import SimulationConfig 

# ----------------------------
# Tournament Simulation (Vectorized & Parallel Optimized)
# ----------------------------
def _tournament_worker(args: Tuple) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulates a single Swiss-style tournament with vectorized operations where possible.
    Returns:
        wins_per_deck: np.ndarray of shape (n_decks,) — total wins per deck index
        matches_per_deck: np.ndarray of shape (n_decks,) — total matches per deck index
    """
    field_indices, config, win_matrix, matchup_details, rng_seed = args
    # Recreate RNG for reproducibility
    local_rng = np.random.default_rng(rng_seed)
    num_players = len(field_indices)
    player_wins = np.zeros(num_players, dtype=int)
    # Use adjacency matrix for opponent tracking (faster than set operations for small N)
    opponents = np.zeros((num_players, num_players), dtype=bool)
    num_rounds = config['num_rounds']
    use_bayesian = config['use_bayesian_winrates']
    deck_names = config['deck_names']
    n_decks = len(deck_names)
    deck_to_idx = {name: i for i, name in enumerate(deck_names)}
    for _ in range(num_rounds):
        # Vectorized grouping by win count
        unique_wins, inverse = np.unique(player_wins, return_inverse=True)
        sorted_indices = np.argsort(-unique_wins[inverse])  # descending wins
        unpaired = np.arange(num_players)[sorted_indices]
        paired = np.zeros(num_players, dtype=bool)
        matchups = []
        # Vectorized pairing: find first unpaired opponent not previously faced
        for i in range(num_players):
            if paired[i]:
                continue
            p1 = unpaired[i]
            # Vectorized check for unpaired and not previously faced
            candidates = np.where(
                (~paired) &
                (unpaired != p1) &
                (~opponents[p1, unpaired])
            )[0]
            if len(candidates) > 0:
                p2 = unpaired[candidates[0]]
            else:
                # Fallback: any unpaired
                candidates = np.where(~paired & (unpaired != p1))[0]
                if len(candidates) == 0:
                    continue
                p2 = unpaired[candidates[0]]
            paired[i] = True
            paired[np.where(unpaired == p2)[0][0]] = True
            matchups.append((p1, p2))
            opponents[p1, p2] = opponents[p2, p1] = True
        # Vectorized match resolution
        if len(matchups) == 0:
            continue
        p1_indices = np.array([field_indices[p1] for p1, _ in matchups])
        p2_indices = np.array([field_indices[p2] for _, p2 in matchups])
        if use_bayesian:
            win_probs = np.array([
                local_rng.beta(
                    matchup_details.get((deck_names[d1], deck_names[d2]), {"win_rate": 0.5, "match_count": 2})["win_rate"] * mc + 1,
                    mc - matchup_details.get((deck_names[d1], deck_names[d2]), {"win_rate": 0.5, "match_count": 2})["win_rate"] * mc + 1
                ) if (mc := matchup_details.get((deck_names[d1], deck_names[d2]), {"match_count": 2})["match_count"]) > 0
                else 0.5
                for d1, d2 in zip(p1_indices, p2_indices)
            ])
        else:
            win_probs = win_matrix[p1_indices, p2_indices]
        p1_wins = local_rng.random(len(matchups)) < win_probs
        for idx, (p1, p2) in enumerate(matchups):
            if p1_wins[idx]:
                player_wins[np.where(unpaired == p1)[0][0]] += 1
            else:
                player_wins[np.where(unpaired == p2)[0][0]] += 1
    # Vectorized aggregation
    wins_per_deck = np.zeros(n_decks, dtype=int)
    matches_per_deck = np.zeros(n_decks, dtype=int)
    np.add.at(wins_per_deck, field_indices, player_wins)
    np.add.at(matches_per_deck, field_indices, num_rounds)
    return wins_per_deck, matches_per_deck

def run_tournament_generation(
    current_freq: np.ndarray,
    deck_names: List[str],
    win_matrix: np.ndarray,
    matchup_details: Dict[Tuple[str, str], Dict[str, Any]],
    config: Dict[str, Any],
    rng: np.random.Generator
) -> np.ndarray:
    """
    Runs one generation of stochastic tournaments with optimized parallel processing.
    Uses imap_unordered for maximum throughput.
    """
    n_decks = len(deck_names)
    tasks = []
    for _ in range(config['num_tournaments_per_gen']):
        field_indices = rng.choice(n_decks, size=config['tournament_size'], p=current_freq)
        task_rng_seed = rng.integers(1 << 60)
        task_config = {
            'num_rounds': config['num_rounds'],
            'use_bayesian_winrates': config['use_bayesian_winrates'],
            'deck_names': deck_names
        }
        tasks.append((field_indices, task_config, win_matrix, matchup_details, task_rng_seed))
    deck_wins = np.zeros(n_decks)
    deck_matches = np.zeros(n_decks)
    use_pool = config['use_multiproc'] and MULTIPROC_AVAILABLE and len(tasks) > 1 and mp is not None
    if use_pool:
        assert mp is not None
        with mp.Pool() as pool:
            for wins, matches in pool.imap_unordered(_tournament_worker, tasks):
                deck_wins += wins
                deck_matches += matches
    else:
        for task in tasks:
            wins, matches = _tournament_worker(task)
            deck_wins += wins
            deck_matches += matches

    # Calculate payoffs only for decks that played matches, default to 0.5 otherwise.
    # Wrap in errstate to suppress harmless 'invalid value' warning when deck_matches is 0.
    with np.errstate(divide='ignore', invalid='ignore'):
        payoffs = np.where(deck_matches > 0, deck_wins / deck_matches, 0.5)

    # Apply exponential selection pressure from config
    new_freq = current_freq * np.exp(config['selection_pressure'] * (payoffs - 0.5))
    return safe_normalize(new_freq)

def update_replicator_dynamics(
    current_freq: np.ndarray,
    win_matrix: np.ndarray,
    noise_scale: float = NOISE_SCALE,
) -> np.ndarray:
    """
    Replicator dynamics with optional Gaussian noise and frequency-based dampening.
    """
    payoffs = win_matrix @ current_freq
    avg_payoff = current_freq @ payoffs
    if avg_payoff <= 0:
        return current_freq
    growth = (payoffs / avg_payoff)
    if noise_scale > 0:
        noise = np.random.default_rng().normal(0, noise_scale, size=growth.shape)
        growth *= np.exp(noise)
    new_freq = current_freq * growth
    return safe_normalize(new_freq)

# ----------------------------
# Evolutionary Stable State Solver
# ----------------------------
def find_evolutionary_stable_state(
    deck_names: List[str],
    win_matrix: np.ndarray,
    matchup_details: Dict[Tuple[str, str], Dict[str, Any]],
    config: SimulationConfig,  # Accepts a single SimulationConfig object
    history_file_path: Optional[str] = None, # Optional path for incremental history saving
) -> Tuple[List[Dict[str, Any]], List[np.ndarray], List[Optional[int]]]:
    """
    Simulates metagame evolution with enhanced dynamics: noise, reintroduction.
    Enters a "Soft Convergence" state after 50% of max_generations to find a true equilibrium.
    Writes history to a file incrementally to manage memory for long simulations.
    Args:
        deck_names (List[str]): List of deck archetype names.
        win_matrix (np.ndarray): Symmetric n x n matrix of win probabilities.
        matchup_details (Dict): Raw data for Bayesian sampling, keyed by (deck_a, deck_b).
        config (SimulationConfig): A dataclass containing all simulation parameters.
        history_file_path (Optional[str]): If provided, the full metagame history will be
                                          written to this CSV file incrementally.
    Returns:
        Tuple containing:
            - List[Dict]: Final results for each deck (frequency, activity, etc.).
            - List[np.ndarray]: In-memory history of the last `convergence_window` generations.
            - List[Optional[int]]: Generation number when each deck went extinct.
    """
    n = len(deck_names)
    if n == 0:
        logging.warning("No decks for simulation")
        return [], [], []

    # --- Extract all parameters from the `config` object ---
    mode = config.mode
    max_generations = config.max_generations
    soft_convergence_gen = config.soft_convergence_gen
    min_generations = config.min_generations
    extinction_threshold = config.extinction_threshold
    stability_threshold = config.stability_threshold
    convergence_window = config.convergence_window
    max_inactive_generations = config.max_inactive_generations
    use_bayesian_winrates = config.use_bayesian_winrates
    tournament_size = config.tournament_size
    num_tournaments_per_gen = config.num_tournaments_per_gen
    num_rounds = config.num_rounds
    use_multiproc = config.use_multiproc
    seed = config.seed
    dynamic_deck_intro_prob = config.dynamic_deck_intro_prob
    mutation_floor = config.mutation_floor
    noise_scale = config.noise_scale
    selection_pressure = config.selection_pressure

    # Initialize RNG and metagame state
    rng = np.random.default_rng(seed)
    current_freq = np.ones(n, dtype=float) / n
    usage_history = np.zeros(n, dtype=int)
    extinction_gens: List[Optional[int]] = [None] * n

    # --- History Management ---
    history: List[np.ndarray] = [] # This will now only hold the last `convergence_window` generations
    recent_history_buffer = [] # Buffer to hold frequencies for the convergence window

    # Open the history file if a path is provided
    history_file_handle = None
    if history_file_path:
        try:
            history_file_handle = open(history_file_path, "w", newline="", encoding="utf-8")
            history_writer = csv.writer(history_file_handle)
            # Write header
            history_writer.writerow(["generation"] + deck_names)
            # Write initial state
            history_writer.writerow([0] + current_freq.tolist())
        except Exception as e:
            logging.error(f"Failed to open history file {history_file_path}: {e}")
            history_file_handle = None

    # For convergence checking, we need a buffer of the last `convergence_window` changes.
    recent_max_changes = np.full(convergence_window, np.inf)
    
    # Sampling is now every generation
    sample_interval = 1

    # Log initial state
    initial_state_summary = {deck_names[i]: f"{current_freq[i]:.4%}" for i in range(len(deck_names)) if current_freq[i] > 0}
    logging.info(f"Intialized metagame with {len(initial_state_summary)} active decks: {dict(list(initial_state_summary.items())[:5])}{'...' if len(initial_state_summary) > 5 else ''}")
    
    # Set up generation iterator
    gens_iter: Iterable[int] = range(max_generations)
    if TQDM_AVAILABLE and tqdm is not None:
        gens_iter = tqdm(gens_iter, desc=f"Simulating Metagame ({mode})", leave=False)

    start_time = time.time()
    converged = False
    in_soft_convergence = False

    try:
        for gen in gens_iter:
            # --- Soft Convergence Check ---
            if not in_soft_convergence and gen >= soft_convergence_gen:
                logging.info(f"Entering SOFT CONVERGENCE state at generation {gen}.")
                in_soft_convergence = True
                # Turn off reintroduction and noise in soft convergence
                current_dynamic_intro_prob = 0.0
                current_noise_scale = 0.0
            else:
                current_dynamic_intro_prob = dynamic_deck_intro_prob
                current_noise_scale = noise_scale
            # --------------------------------

            # 1. Compute next frequency
            if mode == 'replicator':
                target_freq = update_replicator_dynamics(
                    current_freq, win_matrix, current_noise_scale
                )
            elif mode == 'tournament':
                # Create a config dict for the tournament generation
                tourney_config = {
                    'use_bayesian_winrates': use_bayesian_winrates,
                    'tournament_size': tournament_size,
                    'num_tournaments_per_gen': num_tournaments_per_gen,
                    'num_rounds': num_rounds,
                    'use_multiproc': use_multiproc,
                    'deck_names': deck_names
                }
                target_freq = run_tournament_generation(
                    current_freq, deck_names, win_matrix, matchup_details, tourney_config, rng
                )
            else:
                raise ValueError(f"Unknown mode: {mode}")

            # 2. Apply selection pressure and normalize
            next_freq = target_freq.copy()
            next_freq = safe_normalize(next_freq)

            # 3. Handle extinction (Frequency-based only)
            inactive_mask = next_freq < extinction_threshold
            usage_history = np.where(inactive_mask, usage_history + 1, 0)
            extinct_mask = (usage_history >= max_inactive_generations) & np.array([g is None for g in extinction_gens])
            for i in np.where(extinct_mask)[0]:
                extinction_gens[i] = gen
                next_freq[i] = 0.0

            # 4. Reintroduction & mutation floor (respecting Soft Convergence)
            next_freq = reintroduce_extinct_decks(
                next_freq, extinction_gens, deck_names, win_matrix, rng,
                intro_prob=current_dynamic_intro_prob, mutation_floor=mutation_floor,
                current_generation=gen
            )

            # 5. Convergence check
            max_change = float(np.max(np.abs(next_freq - current_freq)))
            # Update the rolling buffer for convergence
            recent_max_changes[gen % convergence_window] = max_change
            
            # Write to history file every generation
            if history_file_handle:
                try:
                    history_writer.writerow([gen + 1] + next_freq.tolist())
                    history_file_handle.flush() # Ensure data is written to disk
                except Exception as e:
                    logging.error(f"Failed to write to history file at gen {gen+1}: {e}")

            # Keep only the last `convergence_window` generations in memory for analysis
            if len(recent_history_buffer) >= convergence_window:
                recent_history_buffer.pop(0)
            recent_history_buffer.append(current_freq.copy())

            # Check for stability after min_generations
            if gen >= min_generations:
                is_stable = np.max(recent_max_changes) < stability_threshold
                if is_stable and not converged:
                    logging.info(f"✅ Metagame stabilized after {gen+1} generations.")
                    converged = True
                    break  # Exit early if converged

            # Update current state for next iteration
            current_freq = next_freq

        if not converged:
            logging.info(f"🏁 Max generations reached ({max_generations}).")

    except KeyboardInterrupt:
        logging.info("🛑 Simulation interrupted.")

    finally:
        # Close the history file
        if history_file_handle:
            history_file_handle.close()

        # Populate the `history` list with the final buffer for return to analysis functions
        history = recent_history_buffer.copy()
        # Add the final state if it's not already the last item
        if len(history) == 0 or not np.array_equal(history[-1], current_freq):
            history.append(current_freq.copy())

        logging.info(f"⏱️  Simulation took {time.time() - start_time:.2f} seconds")

    # Optional Smoothing (for visualization) - only on the in-memory buffer
    try:
        from scipy.ndimage import gaussian_filter1d
        if len(history) > 2:
            arr = np.stack(history)
            smoothed = np.zeros_like(arr)
            for i in range(n):
                smoothed[:, i] = gaussian_filter1d(arr[:, i], sigma=1.0)
            row_sums = smoothed.sum(axis=1, keepdims=True)
            row_sums[row_sums <= 0] = 1.0
            smoothed = np.maximum(smoothed, 0.0) / row_sums
            history = [row.copy() for row in smoothed]
    except Exception:
        pass

    # Build and return final results
    results = []
    for i in range(n):
        results.append({
            "deck": deck_names[i],
            "frequency": float(current_freq[i]),
            "is_active": current_freq[i] > extinction_threshold,
            "generations_inactive": int(usage_history[i]),
            "extinction_generation": extinction_gens[i]
        })

    return results, history, extinction_gens

# ----------------------------
# Deck Dynamics
# ----------------------------
def reintroduce_extinct_decks(
    current_freq: np.ndarray,
    extinction_gens: List[Optional[int]],
    deck_names: List[str],
    win_matrix: np.ndarray,
    rng: np.random.Generator,
    intro_prob: float = DYNAMIC_DECK_INTRO_PROB,
    mutation_floor: float = MUTATION_FLOOR,
    current_generation: int = 0
) -> np.ndarray:
    """
    Manages the population dynamics of a deck-based simulation by applying a mutation floor
    and, with a small probability, reintroducing extinct decks.
    This function performs two main operations to prevent the simulation from
    losing diversity and becoming stuck in a local equilibrium:
    1. Applies a **mutation floor** to the frequencies of all decks. This ensures
       that no deck's frequency drops below a minimum threshold, preventing any
       deck from being completely eliminated. This maintains a small, persistent
       presence for all decks in the population.
    2. Reintroduces a single **extinct deck** with a small probability. If the
       random event occurs, a randomly chosen extinct deck is given a small,
       non-zero frequency, effectively reviving it. This process helps to
       maintain genetic diversity and allows the simulation to explore new
       evolutionary paths.
    Args:
        current_freq (np.ndarray):
            The current normalized frequencies of all decks in the population.
        extinction_gens (List[Optional[int]]):
            A list tracking the generation in which each deck went extinct.
            'None' indicates the deck is currently active.
        deck_names (List[str]):
            A list of the names of the decks, corresponding to the indices
            in `current_freq`. Used for logging purposes.
        win_matrix (np.ndarray):
            A matrix representing the win probabilities between different decks.
            This parameter is included for context but is not used in this function.
        rng (np.random.Generator):
            A NumPy random number generator instance for reproducible randomness.
        intro_prob (float, optional):
            The probability (between 0 and 1) that a single extinct deck
            will be reintroduced in the current generation. Defaults to
            `DYNAMIC_DECK_INTRO_PROB`.
        mutation_floor (float, optional):
            The minimum frequency allowed for any deck in the population.
            This value also serves as the base for the revival frequency.
            Defaults to `MUTATION_FLOOR`.
        current_generation (int, optional):
            The current generation number of the simulation. Used for
            informational logging. Defaults to 0.
    Returns:
        np.ndarray: The updated deck frequencies after applying the mutation floor
                    and, if applicable, reintroducing an extinct deck. The returned
                    array is normalized.
    """
    n = len(current_freq)
    active_mask = np.array([g is None for g in extinction_gens])
    extinct_indices = np.where(~active_mask)[0]
    # Apply mutation floor to all decks
    current_freq = np.maximum(current_freq, mutation_floor)
    # Chance to reintroduce extinct decks
    if len(extinct_indices) > 0 and rng.random() < intro_prob:
        chosen_idx = rng.choice(extinct_indices)
        # Revive with small frequency
        current_freq[chosen_idx] = mutation_floor * 10
        extinction_gens[chosen_idx] = None  # Mark as active again
        logging.debug(f"Reintroduced deck '{deck_names[chosen_idx]}' at generation {current_generation}.")
    return safe_normalize(current_freq)