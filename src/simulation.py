#!/usr/bin/env python3
# simulation.py — Core metagame evolution engine with dual tournament modes
from __future__ import annotations
import time
import logging
import numpy as np
import csv
import bisect
from typing import List, Tuple, Dict, Any, Optional, Iterable

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
# Tournament Simulation Workers
# ----------------------------

def _pure_swiss_worker(args: Tuple) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulates a standard, fast, single-phase Swiss tournament.
    """
    field_indices, config, win_matrix, matchup_details, rng_seed = args
    local_rng = np.random.default_rng(rng_seed)

    num_players = len(field_indices)
    player_wins = np.zeros(num_players, dtype=int)
    opponents = np.zeros((num_players, num_players), dtype=bool)

    num_rounds = config["num_rounds"]
    use_bayesian = config["use_bayesian_winrates"]
    deck_names = config["deck_names"]
    n_decks = len(deck_names)

    for _ in range(num_rounds):
        unique_wins, inverse = np.unique(player_wins, return_inverse=True)
        sorted_indices = np.argsort(-unique_wins[inverse]) 
        unpaired = np.arange(num_players)[sorted_indices]
        paired = np.zeros(num_players, dtype=bool)
        matchups = []

        for i in range(num_players):
            if paired[i]:
                continue
            p1 = unpaired[i]
            candidates = np.where((~paired) & (unpaired != p1) & (~opponents[p1, unpaired]))[0]
            if len(candidates) > 0:
                p2 = unpaired[candidates[0]]
            else:
                candidates = np.where(~paired & (unpaired != p1))[0]
                if len(candidates) == 0:
                    continue 
                p2 = unpaired[candidates[0]]

            paired[i] = True
            paired[np.where(unpaired == p2)[0][0]] = True
            matchups.append((p1, p2))
            opponents[p1, p2] = opponents[p2, p1] = True

        if len(matchups) == 0:
            continue

        p1_indices = np.array([field_indices[p1] for p1, _ in matchups])
        p2_indices = np.array([field_indices[p2] for _, p2 in matchups])

        if use_bayesian:
            win_probs_list = []
            default_mu = {"win_rate": 0.5, "match_count": 2}
            for d1, d2 in zip(p1_indices, p2_indices):
                mu_key = (deck_names[d1], deck_names[d2])
                mu = matchup_details.get(mu_key, default_mu)
                wr = mu["win_rate"]
                mc = mu["match_count"]

                if mc > 0:
                    alpha = wr * mc + 1
                    beta = (1 - wr) * mc + 1
                    prob = local_rng.beta(alpha, beta)
                else:
                    prob = 0.5
                win_probs_list.append(prob)
            win_probs = np.array(win_probs_list)
        else:
            win_probs = win_matrix[p1_indices, p2_indices]

        p1_wins = local_rng.random(len(matchups)) < win_probs

        for idx, (p1, p2) in enumerate(matchups):
            if p1_wins[idx]:
                player_wins[np.where(unpaired == p1)[0][0]] += 1
            else:
                player_wins[np.where(unpaired == p2)[0][0]] += 1

    wins_per_deck = np.zeros(n_decks, dtype=float)
    matches_per_deck = np.zeros(n_decks, dtype=float)
    np.add.at(wins_per_deck, field_indices, player_wins)
    np.add.at(matches_per_deck, field_indices, num_rounds)

    return wins_per_deck, matches_per_deck

def get_variant_5_structure(players: int) -> Tuple[int, int, int, int]:
    """Returns: (Day1_Rounds, Match_Point_Cutoff, Day2_Rounds, Top_Cut) based on official handbook."""
    from .config import _STRUCTURE_RESULTS, _STRUCTURE_THRESHOLDS
    index = bisect.bisect_left(_STRUCTURE_THRESHOLDS, players)
    return _STRUCTURE_RESULTS[index]

def _championship_series_worker(args: Tuple) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulates a full 2-Day Play! Pokémon Regional/International (Variant #5).
    Converts placement into "Win-Equivalents" for evolutionary scaling.
    """
    field_indices, config, win_matrix, matchup_details, rng_seed = args
    local_rng = np.random.default_rng(rng_seed)
    
    num_players = len(field_indices)
    n_decks = len(config["deck_names"])
    
    d1_rounds, cut_points, d2_rounds, top_cut = get_variant_5_structure(num_players)
    
    # BO3 Conversion Formula: P_bo3 = 3p^2 - 2p^3
    bo3_win_matrix = 3 * (win_matrix ** 2) - 2 * (win_matrix ** 3)
    
    match_points = np.zeros(num_players, dtype=int)
    opponents_history = [[] for _ in range(num_players)]
    active_players = np.arange(num_players)
    
    # Helper for fast Swiss-style pairing
    def simulate_swiss_phase(rounds_to_play, current_active):
        for _ in range(rounds_to_play):
            if len(current_active) < 2: break
            
            # Shuffle to randomize tiebreakers, then sort by points
            local_rng.shuffle(current_active)
            order = np.argsort(-match_points[current_active], kind='mergesort')
            sorted_active = current_active[order]
            
            p1s = sorted_active[0::2]
            p2s = sorted_active[1::2]
            if len(p1s) > len(p2s): p1s = p1s[:-1] # Drop the bye for simulation speed
            
            p1_decks = field_indices[p1s]
            p2_decks = field_indices[p2s]
            
            win_probs = bo3_win_matrix[p1_decks, p2_decks]
            p1_wins = local_rng.random(len(p1s)) < win_probs
            
            # 3 match points for a win, 0 for loss
            match_points[p1s[p1_wins]] += 3
            match_points[p2s[~p1_wins]] += 3
            
            for p1, p2 in zip(p1s, p2s):
                opponents_history[p1].append(p2)
                opponents_history[p2].append(p1)

    # --- Phase 1 (Day 1) ---
    simulate_swiss_phase(d1_rounds, active_players)
    
    # --- The Cut ---
    day2_mask = match_points >= cut_points
    day2_players = np.where(day2_mask)[0]
    
    # --- Phase 2 (Day 2) ---
    simulate_swiss_phase(d2_rounds, day2_players)
    
    # --- Tiebreakers & Top Cut ---
    top_players = []
    champion = None
    
    if len(day2_players) > 0 and top_cut > 0:
        owp = np.zeros(num_players)
        for i in day2_players:
            opps = opponents_history[i]
            if len(opps) == 0: continue
            opp_win_pcts = [match_points[o] / (len(opponents_history[o]) * 3) for o in opps]
            opp_win_pcts = np.clip(opp_win_pcts, 0.25, 1.0) # Handbook min bounds
            owp[i] = np.mean(opp_win_pcts)
            
        # Lexsort sorts by last key first. We negate to get descending.
        top_order = np.lexsort((-owp[day2_players], -match_points[day2_players]))
        top_players = day2_players[top_order][:top_cut]
        
        # --- Single Elimination Playoffs ---
        standings = list(top_players)
        while len(standings) > 1:
            next_round = []
            p1s = np.array(standings[0::2])
            p2s = np.array(standings[1::2])
            if len(p1s) > len(p2s): p1s = p1s[:-1]

            p1_decks = field_indices[p1s]
            p2_decks = field_indices[p2s]
            p1_wins = local_rng.random(len(p1s)) < bo3_win_matrix[p1_decks, p2_decks]

            next_round.extend(p1s[p1_wins])
            next_round.extend(p2s[~p1_wins])
            standings = next_round
            
        champion = standings[0] if standings else None

    # --- Win-Equivalent Scaling ---
    # Translate tournament placement back into "Win Rate" equivalents for the engine
    wins_equiv = match_points / 3.0
    matches_equiv = np.full(num_players, float(d1_rounds))

    # Apply structural bonuses
    matches_equiv[day2_players] += d2_rounds
    wins_equiv[day2_players] += 1.5 
    matches_equiv[day2_players] += 1.5

    if len(top_players) > 0:
        matches_equiv[top_players] += 3.0
        wins_equiv[top_players] += 3.0 
        if champion is not None:
            wins_equiv[champion] += 4.0
            matches_equiv[champion] += 4.0

    wins_per_deck = np.zeros(n_decks, dtype=float)
    matches_per_deck = np.zeros(n_decks, dtype=float)
    np.add.at(wins_per_deck, field_indices, wins_equiv)
    np.add.at(matches_per_deck, field_indices, matches_equiv)

    return wins_per_deck, matches_per_deck


def run_tournament_generation(
        current_freq: np.ndarray,
        deck_names: List[str],
        win_matrix: np.ndarray,
        matchup_details: Dict[Tuple[str, str], Dict[str, Any]],
        config: Dict[str, Any],
        rng: np.random.Generator,
) -> np.ndarray:
    """
    Runs one generation of stochastic tournaments handling either Pure Swiss or Championship Series modes.
    """
    n_decks = len(deck_names)
    tasks = []
    tournament_style = config.get("tournament_style", "pure_swiss")

    task_config = {
        "num_rounds": config["num_rounds"],
        "use_bayesian_winrates": config["use_bayesian_winrates"],
        "deck_names": deck_names,
        "selection_pressure": config["selection_pressure"],
        "tournament_style": tournament_style
    }

    for _ in range(config["num_tournaments_per_gen"]):
        field_indices = rng.choice(n_decks, size=config["tournament_size"], p=current_freq)
        task_rng_seed = rng.integers(1 << 60)
        tasks.append((field_indices, task_config, win_matrix, matchup_details, task_rng_seed))

    deck_wins = np.zeros(n_decks)
    deck_matches = np.zeros(n_decks)
    
    worker_func = _championship_series_worker if tournament_style == "championship_series" else _pure_swiss_worker

    use_pool = config["use_multiproc"] and MULTIPROC_AVAILABLE and len(tasks) > 1 and mp is not None
    if use_pool:
        assert mp is not None
        with mp.Pool() as pool:
            for wins, matches in pool.imap_unordered(worker_func, tasks):
                deck_wins += wins
                deck_matches += matches
    else:
        for task in tasks:
            wins, matches = worker_func(task)
            deck_wins += wins
            deck_matches += matches

    with np.errstate(divide="ignore", invalid="ignore"):
        payoffs = np.where(deck_matches > 0, deck_wins / deck_matches, 0.5)

    new_freq = current_freq * np.exp(config["selection_pressure"] * (payoffs - 0.5))
    return safe_normalize(new_freq)


def update_replicator_dynamics(
        current_freq: np.ndarray,
        win_matrix: np.ndarray,
        rng: np.random.Generator,
        noise_scale: float = NOISE_SCALE,
) -> np.ndarray:
    payoffs = win_matrix @ current_freq
    avg_payoff = current_freq @ payoffs

    if avg_payoff <= 0:
        return current_freq 

    growth = payoffs / avg_payoff

    if noise_scale > 0:
        noise = rng.normal(0, noise_scale, size=growth.shape)
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
        config: SimulationConfig,
        history_file_path: Optional[str] = None, 
) -> Tuple[List[Dict[str, Any]], List[np.ndarray], List[Optional[int]]]:
    n = len(deck_names)
    if n == 0:
        logging.warning("No decks for simulation")
        return [], [], []

    mode = config.mode
    max_generations = config.max_generations
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
    
    # Graceful fallback if the user hasn't added this to SimulationConfig yet
    tournament_style = getattr(config, "tournament_style", "pure_swiss")

    rng = np.random.default_rng(seed)
    current_freq = np.ones(n, dtype=float) / n
    usage_history = np.zeros(n, dtype=int)
    extinction_gens: List[Optional[int]] = [None] * n

    history: List[np.ndarray] = [current_freq.copy()]
    recent_max_changes = np.full(convergence_window, np.inf)

    history_file_handle = None
    history_writer = None
    if history_file_path:
        try:
            history_file_handle = open(history_file_path, "w", newline="", encoding="utf-8")
            history_writer = csv.writer(history_file_handle)
            history_writer.writerow(["generation"] + deck_names)
            history_writer.writerow([0] + current_freq.tolist())
        except Exception as e:
            logging.error(f"Failed to open history file {history_file_path}: {e}")
            if history_file_handle:
                history_file_handle.close()
            history_file_handle = None
            history_writer = None

    initial_state_summary = {
        deck_names[i]: f"{current_freq[i]:.4%}" for i in range(len(deck_names)) if current_freq[i] > 0
    }
    logging.info(
        f"Intialized metagame with {len(initial_state_summary)} active decks: {dict(list(initial_state_summary.items())[:5])}{'...' if len(initial_state_summary) > 5 else ''}"
    )

    gens_iter: Iterable[int] = range(max_generations)
    if TQDM_AVAILABLE and tqdm is not None:
        gens_iter = tqdm(gens_iter, desc=f"Simulating Metagame ({mode})", leave=False)

    start_time = time.time()
    converged = False

    tourney_config = {
        "use_bayesian_winrates": use_bayesian_winrates,
        "tournament_size": tournament_size,
        "num_tournaments_per_gen": num_tournaments_per_gen,
        "num_rounds": num_rounds,
        "use_multiproc": use_multiproc,
        "deck_names": deck_names,
        "selection_pressure": selection_pressure,
        "tournament_style": tournament_style
    }

    try:
        for gen in gens_iter:
            if mode == "replicator":
                target_freq = update_replicator_dynamics(
                    current_freq, win_matrix, rng, noise_scale
                )
            elif mode == "tournament":
                target_freq = run_tournament_generation(
                    current_freq,
                    deck_names,
                    win_matrix,
                    matchup_details,
                    tourney_config,
                    rng,
                )
            else:
                raise ValueError(f"Unknown mode: {mode}")

            next_freq = target_freq.copy()
            next_freq = safe_normalize(next_freq)

            inactive_mask = next_freq < extinction_threshold
            usage_history = np.where(inactive_mask, usage_history + 1, 0)
            extinct_mask = (usage_history >= max_inactive_generations) & np.array([g is None for g in extinction_gens])

            for i in np.where(extinct_mask)[0]:
                extinction_gens[i] = gen
                next_freq[i] = 0.0

            next_freq = reintroduce_extinct_decks(
                next_freq,
                extinction_gens,
                deck_names,
                rng,
                intro_prob=dynamic_deck_intro_prob,
                mutation_floor=mutation_floor,
                current_generation=gen,
            )

            max_change = float(np.max(np.abs(next_freq - current_freq)))
            recent_max_changes[gen % convergence_window] = max_change

            history.append(current_freq.copy())

            if history_writer and history_file_handle:
                try:
                    history_writer.writerow([gen + 1] + current_freq.tolist())
                    history_file_handle.flush()
                except Exception as e:
                    logging.error(f"Failed to write to history file at gen {gen + 1}: {e}")

            if gen >= min_generations:
                is_stable = np.max(recent_max_changes) < stability_threshold
                if is_stable and not converged:
                    logging.info(f"✅ Metagame stabilized after {gen + 1} generations.")
                    converged = True
                    break 

            current_freq = next_freq

        if not converged:
            logging.info(f"🏁 Max generations reached ({max_generations}).")

    except KeyboardInterrupt:
        logging.info("🛑 Simulation interrupted.")
    finally:
        if history_file_handle:
            history_file_handle.close()

        if len(history) == 0 or not np.array_equal(history[-1], current_freq):
            history.append(current_freq.copy())

        logging.info(f"⏱️  Simulation took {time.time() - start_time:.2f} seconds")

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
    except Exception as e:
        logging.debug(f"Optional smoothing failed, returning raw history: {e}")
        pass

    results = []
    for i in range(n):
        results.append(
            {
                "deck": deck_names[i],
                "frequency": float(current_freq[i]),
                "is_active": current_freq[i] > extinction_threshold,
                "generations_inactive": int(usage_history[i]),
                "extinction_generation": extinction_gens[i],
            }
        )
    return results, history, extinction_gens


# ----------------------------
# Deck Dynamics
# ----------------------------

def reintroduce_extinct_decks(
        current_freq: np.ndarray,
        extinction_gens: List[Optional[int]],
        deck_names: List[str],
        rng: np.random.Generator,
        intro_prob: float = DYNAMIC_DECK_INTRO_PROB,
        mutation_floor: float = MUTATION_FLOOR,
        current_generation: int = 0,
) -> np.ndarray:
    
    active_mask = np.array([g is None for g in extinction_gens])
    extinct_indices = np.where(~active_mask)[0]

    current_freq = np.maximum(current_freq, mutation_floor)

    if len(extinct_indices) > 0 and rng.random() < intro_prob:
        chosen_idx = rng.choice(extinct_indices)
        current_freq[chosen_idx] = mutation_floor * 10
        extinction_gens[chosen_idx] = None 
        logging.debug(f"Reintroduced deck '{deck_names[chosen_idx]}' at generation {current_generation}.")

    return safe_normalize(current_freq)