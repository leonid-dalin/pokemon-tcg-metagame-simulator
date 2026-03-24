#!/usr/bin/env python3
# engine.py | Replicator & Tournament generations (formerly simulation.py)
from __future__ import annotations
import time
import logging
import numpy as np
import csv
from typing import List, Dict, Any, Optional, Iterable

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
from src.core.config import *
from src.core.data import safe_normalize
from src.core.types import SimulationConfig
from src.tournament.solver import get_variant_5_structure

# ----------------------------
# Tournament Simulation Workers
# ----------------------------

def _pure_swiss_worker(args: tuple) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulates a standard, fast, single-phase Swiss tournament using fast array slicing.
    """
    field_indices, config, win_matrix, matchup_details, rng_seed = args
    local_rng = np.random.default_rng(rng_seed)

    num_players = len(field_indices)
    player_wins = np.zeros(num_players, dtype=int)
    active_players = np.arange(num_players)

    num_rounds = config["num_rounds"]
    use_bayesian = config["use_bayesian_winrates"]
    deck_names = config["deck_names"]
    n_decks = len(deck_names)
    wins_per_deck = np.zeros(n_decks, dtype=float)
    matches_per_deck = np.zeros(n_decks, dtype=float)
    
    for _ in range(num_rounds):
        # Fast Swiss pairing logic
        local_rng.shuffle(active_players)
        order = np.argsort(-player_wins[active_players], kind='mergesort')
        sorted_active = active_players[order]

        p1s = sorted_active[0::2]
        p2s = sorted_active[1::2]
        if len(p1s) > len(p2s):
            p1s = p1s[:-1]
        if len(p1s) == 0:
            continue

        p1_indices = field_indices[p1s]
        p2_indices = field_indices[p2s]

        if use_bayesian:
            win_probs = np.zeros(len(p1s))
            for idx, (d1, d2) in enumerate(zip(p1_indices, p2_indices)):
                mu = matchup_details.get((deck_names[d1], deck_names[d2]), {"win_rate": 0.5, "match_count": 0})
                wr, mc = mu["win_rate"], mu["match_count"]
                
                if mc > 0:
                    win_probs[idx] = local_rng.beta(wr * mc + 1, (1 - wr) * mc + 1)
                else:
                    win_probs[idx] = 0.5
        else:
            win_probs = win_matrix[p1_indices, p2_indices]

        p1_wins = local_rng.random(len(p1s)) < win_probs
        player_wins[p1s[p1_wins]] += 1
        player_wins[p2s[~p1_wins]] += 1
        np.add.at(matches_per_deck, p1_indices, 1)
        np.add.at(matches_per_deck, p2_indices, 1)

    np.add.at(wins_per_deck, field_indices, player_wins)

    return wins_per_deck, matches_per_deck

def _championship_series_worker(args: tuple) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulates a full 2-Day Play! Pokémon Regional/International (Variant #5).
    Incorporates Parabolic Tie Convergence and vectorized memory allocation.
    """
    field_indices, config, win_matrix, matchup_details, rng_seed = args
    local_rng = np.random.default_rng(rng_seed)
    
    num_players = len(field_indices)
    n_decks = len(config["deck_names"])
    
    d1_rounds, cut_points, d2_rounds, top_cut = get_variant_5_structure(num_players)
    max_rounds = d1_rounds + d2_rounds
    
    # BO3 Conversion Formula: P_bo3 = 3p^2 - 2p^3
    bo3_win_matrix = 3 * (win_matrix ** 2) - 2 * (win_matrix ** 3)
    
    match_points = np.zeros(num_players, dtype=int)
    
    # Pre-allocated 2D array for speed instead of Python lists
    opponents_matrix = np.full((num_players, max_rounds), -1, dtype=int)
    rounds_played = np.zeros(num_players, dtype=int)
    
    active_players = np.arange(num_players)
    global_tie_rate = config.get("global_tie_rate", 0.15)
    
    def simulate_swiss_phase(rounds_to_play, current_active):
        for _ in range(rounds_to_play):
            if len(current_active) < 2: break
            
            local_rng.shuffle(current_active)
            order = np.argsort(-match_points[current_active], kind='mergesort')
            sorted_active = current_active[order]
            
            sw_p1s = sorted_active[0::2]
            sw_p2s = sorted_active[1::2]
            if len(sw_p1s) > len(sw_p2s): sw_p1s = sw_p1s[:-1]
            
            p1_decks = field_indices[sw_p1s]
            p2_decks = field_indices[sw_p2s]
            win_probs = bo3_win_matrix[p1_decks, p2_decks]
            
            # Parabolic Tie Convergence
            tie_probs = global_tie_rate * 4.0 * win_probs * (1.0 - win_probs)
            rolls = local_rng.random(len(sw_p1s))
            
            p1_win_thresh = np.clip(win_probs - (tie_probs / 2.0), 0.0, 1.0)
            tie_thresh = np.clip(p1_win_thresh + tie_probs, 0.0, 1.0)
            
            p1_wins_mask = rolls < p1_win_thresh
            p2_wins_mask = rolls >= tie_thresh
            ties_mask = ~(p1_wins_mask | p2_wins_mask)
            
            match_points[sw_p1s[p1_wins_mask]] += 3
            match_points[sw_p2s[p2_wins_mask]] += 3
            match_points[sw_p1s[ties_mask]] += 1
            match_points[sw_p2s[ties_mask]] += 1
            
            # Vectorized history tracking
            opponents_matrix[sw_p1s, rounds_played[sw_p1s]] = sw_p2s
            opponents_matrix[sw_p2s, rounds_played[sw_p2s]] = sw_p1s
            rounds_played[sw_p1s] += 1
            rounds_played[sw_p2s] += 1

    # Phase 1 (Day 1)
    simulate_swiss_phase(d1_rounds, active_players)
    
    # The Cut
    day2_mask = match_points >= cut_points
    day2_players = np.where(day2_mask)[0]
    
    # Phase 2 (Day 2)
    simulate_swiss_phase(d2_rounds, day2_players)
    
    # Tiebreakers & Top Cut
    
    if len(day2_players) > 0 and top_cut > 0:
        owp = np.zeros(num_players)
        for i in day2_players:
            rp = rounds_played[i]
            if rp == 0: continue
            opps = opponents_matrix[i, :rp]
            opp_points = match_points[opps]
            opp_rp = rounds_played[opps]
            # Avoid division by zero, enforce 0.25 floor per handbook
            opp_win_pcts = np.clip(opp_points / (np.maximum(opp_rp, 1) * 3), 0.25, 1.0)
            owp[i] = np.mean(opp_win_pcts)
            
        top_order = np.lexsort((-owp[day2_players], -match_points[day2_players]))
        top_players = day2_players[top_order][:top_cut]
        
        # Single Elimination Playoffs (No ties)
        standings = top_players.copy()
        while len(standings) > 1:
            next_round = []
            tc_p1s = np.array(standings[0::2])
            tc_p2s  = np.array(standings[1::2])
            if len(tc_p1s) > len(tc_p2s ): tc_p1s = tc_p1s[:-1]

            tc_p1_decks = field_indices[tc_p1s]
            tc_p2_decks = field_indices[tc_p2s ]
            p1_wins = local_rng.random(len(tc_p1s)) < bo3_win_matrix[tc_p1_decks, tc_p2_decks]

            next_round.extend(tc_p1s[p1_wins])
            next_round.extend(tc_p2s [~p1_wins])
            standings = np.array(next_round)
            
            # Grant extra equivalent points for advancing in Top Cut
            match_points[next_round] += 3
            rounds_played[tc_p1s] += 1
            rounds_played[tc_p2s] += 1

    # Pure scaling based on actual match points acquired
    wins_equiv = match_points / 3.0
    matches_equiv = rounds_played.astype(float)

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
        pool: Optional[mp.Pool] = None,
) -> np.ndarray:
    n_decks = len(deck_names)
    tasks = []
    tournament_style = config.get("tournament_style", "pure_swiss")

    task_config = {
        "num_rounds": config["num_rounds"],
        "use_bayesian_winrates": config["use_bayesian_winrates"],
        "deck_names": deck_names,
        "selection_pressure": config["selection_pressure"],
        "tournament_style": tournament_style,
        "global_tie_rate": config.get("global_tie_rate", 0.15)
    }

    for _ in range(config["num_tournaments_per_gen"]):
        field_indices = rng.choice(n_decks, size=config["tournament_size"], p=current_freq)
        task_rng_seed = rng.integers(1 << 60)
        tasks.append((field_indices, task_config, win_matrix, matchup_details, task_rng_seed))

    deck_wins = np.zeros(n_decks)
    deck_matches = np.zeros(n_decks)
    
    worker_func = _championship_series_worker if tournament_style == "championship_series" else _pure_swiss_worker

    if pool is not None and len(tasks) > 1:
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

    gens_iter: Iterable[int] = range(max_generations)
    if TQDM_AVAILABLE and tqdm is not None:
        gens_iter = tqdm(gens_iter, desc=f"Simulating Metagame ({mode})", leave=False)

    start_time = time.time()
    pool = None
    if mode == "tournament" and use_multiproc and MULTIPROC_AVAILABLE and mp is not None:
        pool = mp.Pool()
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
                    pool=pool
                )
            else:
                raise ValueError(f"Unknown mode: {mode}")

            next_freq = safe_normalize(target_freq.copy())

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
                if is_stable:
                    logging.info(f"✅ Metagame stabilized after {gen + 1} generations.")
                    break

            current_freq = next_freq

    except KeyboardInterrupt:
        logging.info("🛑 Simulation interrupted.")
    finally:
        if pool is not None:
            pool.close()
            pool.join()
        if history_file_handle:
            history_file_handle.close()

        if len(history) == 0 or not np.array_equal(history[-1], current_freq):
            history.append(current_freq.copy())

        logging.info(f"⏱️  Simulation took {time.time() - start_time:.2f} seconds")

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

    active_mask = np.array([g is None for g in extinction_gens], dtype=bool)
    extinct_indices = np.where(~active_mask)[0]

    # Purged the global mutation_floor application that broke Replicator purity.
    if len(extinct_indices) > 0 and rng.random() < intro_prob:
        chosen_idx = rng.choice(extinct_indices)
        current_freq[chosen_idx] = max(mutation_floor * 10, 1e-5)
        extinction_gens[chosen_idx] = None
        logging.debug(f"Reintroduced deck '{deck_names[chosen_idx]}' at generation {current_generation}.")

    return safe_normalize(current_freq)