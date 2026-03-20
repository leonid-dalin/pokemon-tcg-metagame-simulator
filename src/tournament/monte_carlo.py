#!/usr/bin/env python3
# monte_carlo.py | High-speed static bracket execution
import numpy as np
import multiprocessing as mp
from typing import Dict, List, Tuple, Any, Optional, Callable
import time

def _mc_worker(args: Tuple) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    iterations, players, meta_distribution, win_matrix, d1_rounds, cut_points, d2_rounds, top_cut, seed, use_tie_convergence, global_tie_rate = args
    rng = np.random.default_rng(seed)
    n_decks = win_matrix.shape[0]
    
    total_initial = np.zeros(n_decks, dtype=int)
    total_day2 = np.zeros(n_decks, dtype=int)
    total_topcut = np.zeros(n_decks, dtype=int)
    total_champ = np.zeros(n_decks, dtype=int)
    
    match_points = np.zeros(players, dtype=int)
    active = np.arange(players)
    
    for _ in range(iterations):
        field_indices = rng.choice(n_decks, size=players, p=meta_distribution)
        total_initial += np.bincount(field_indices, minlength=n_decks)
        
        match_points.fill(0)
        opponents_history = [[] for _ in range(players)]
        
        def play_rounds(rounds, current_active):
            for _ in range(rounds):
                if len(current_active) < 2: break
                rng.shuffle(current_active)
                order = np.argsort(-match_points[current_active], kind='mergesort')
                sorted_active = current_active[order]
                
                p1s = sorted_active[0::2]
                p2s = sorted_active[1::2]
                if len(p1s) > len(p2s): p1s = p1s[:-1]
                
                p1_decks = field_indices[p1s]
                p2_decks = field_indices[p2s]
                
                win_probs = win_matrix[p1_decks, p2_decks]
                
                # --- !!! BETA !!! TIE CONVERGENCE LOGIC ---
                if use_tie_convergence:
                    # Parabolic Tie Formula: T_global * 4 * P * (1-P)
                    tie_probs = global_tie_rate * 4.0 * win_probs * (1.0 - win_probs)
                    rolls = rng.random(len(p1s))
                    
                    p1_win_thresh = win_probs - (tie_probs / 2.0)
                    tie_thresh = p1_win_thresh + tie_probs
                    
                    # Micro-optimized boolean evaluation
                    p1_wins = rolls < p1_win_thresh
                    p2_wins = rolls >= tie_thresh
                    ties = ~(p1_wins | p2_wins)
                    
                    match_points[p1s[p1_wins]] += 3
                    match_points[p2s[p2_wins]] += 3
                    match_points[p1s[ties]] += 1
                    match_points[p2s[ties]] += 1
                else:
                    p1_wins = rng.random(len(p1s)) < win_probs
                    match_points[p1s[p1_wins]] += 3
                    match_points[p2s[~p1_wins]] += 3
                
                for p1, p2 in zip(p1s, p2s):
                    opponents_history[p1].append(p2)
                    opponents_history[p2].append(p1)

        play_rounds(d1_rounds, active)
        
        day2_players = np.where(match_points >= cut_points)[0]
        if len(day2_players) > 0 and d2_rounds > 0:
            total_day2 += np.bincount(field_indices[day2_players], minlength=n_decks)
        
        if d2_rounds > 0 and len(day2_players) > 1:
            play_rounds(d2_rounds, day2_players)
            
        top_players = []
        if top_cut > 0:
            pool = day2_players if (d2_rounds > 0 and len(day2_players) > 0) else active
            if len(pool) > 0:
                owp = np.zeros(players)
                for i in pool:
                    opps = opponents_history[i]
                    if not opps: continue
                    opp_win_pcts = np.clip([match_points[o] / (max(1, len(opponents_history[o])) * 3) for o in opps], 0.25, 1.0)
                    owp[i] = np.mean(opp_win_pcts)
                    
                top_order = np.lexsort((-owp[pool], -match_points[pool]))
                top_players = pool[top_order][:top_cut]
                total_topcut += np.bincount(field_indices[top_players], minlength=n_decks)
            
        if len(top_players) > 0:
            standings = list(top_players)
            while len(standings) > 1:
                half = len(standings) // 2
                p1s = np.array(standings[:half])
                p2s = np.array(standings[half:][::-1]) 
                
                if len(p1s) > len(p2s): p1s = p1s[:-1]
                
                p1_decks = field_indices[p1s]
                p2_decks = field_indices[p2s]
                # Top cut is single elimination; no ties allowed
                p1_wins = rng.random(len(p1s)) < win_matrix[p1_decks, p2_decks]
                
                next_round = []
                for i, p1_won in enumerate(p1_wins):
                    next_round.append(p1s[i] if p1_won else p2s[i])
                standings = next_round
                
            if standings:
                total_champ[field_indices[standings[0]]] += 1

    return total_initial, total_day2, total_topcut, total_champ

def run_monte_carlo_analytics(
    deck_names: List[str],
    win_matrix: np.ndarray,
    meta_distribution: Dict[str, float],
    d1_rounds: int,
    cut_points: int,
    d2_rounds: int,
    top_cut: int,
    players: int = 256,
    iterations: int = 1000,
    match_format: str = "BO3",
    progress_callback: Optional[Callable[[int, int], None]] = None,
    use_tie_convergence: bool = False,
    global_tie_rate: float = 0.15
) -> Dict[str, Dict[str, float]]:
    
    n_decks = len(deck_names)
    meta_vec = np.zeros(n_decks)
    for i, name in enumerate(deck_names):
        meta_vec[i] = meta_distribution.get(name, 0.0)
    meta_vec = meta_vec / np.sum(meta_vec) 
    
    working_matrix = win_matrix.copy()
    if match_format == "BO3":
        working_matrix = 3 * (working_matrix ** 2) - 2 * (working_matrix ** 3)
        
    n_cores = max(1, mp.cpu_count() - 1)
    iters_per_core = max(1, iterations // n_cores)
    tasks = []
    
    base_seed = int(time.time() * 1000) % (1 << 32)
    remaining_iters = iterations
    for i in range(n_cores):
        alloc = min(iters_per_core, remaining_iters)
        if i == n_cores - 1: alloc = remaining_iters 
        if alloc > 0:
            tasks.append((alloc, players, meta_vec, working_matrix, d1_rounds, cut_points, d2_rounds, top_cut, base_seed + i, use_tie_convergence, global_tie_rate))
            remaining_iters -= alloc
        
    total_initial = np.zeros(n_decks, dtype=int)
    total_day2 = np.zeros(n_decks, dtype=int)
    total_topcut = np.zeros(n_decks, dtype=int)
    total_champ = np.zeros(n_decks, dtype=int)
    
    completed_cores = 0
    with mp.Pool(n_cores) as pool:
        for res_init, res_day2, res_top, res_champ in pool.imap_unordered(_mc_worker, tasks):
            total_initial += res_init
            total_day2 += res_day2
            total_topcut += res_top
            total_champ += res_champ
            
            completed_cores += 1
            if progress_callback:
                progress_callback(completed_cores, len(tasks))
            
    results = {}
    with np.errstate(divide='ignore', invalid='ignore'):
        day2_conv = np.where(total_initial > 0, total_day2 / total_initial, 0)
        topcut_conv = np.where(total_initial > 0, total_topcut / total_initial, 0)
        win_conv = np.where(total_initial > 0, total_champ / total_initial, 0)
        
        day2_share = np.where(np.sum(total_day2) > 0, total_day2 / np.sum(total_day2), 0)
        topcut_share = np.where(np.sum(total_topcut) > 0, total_topcut / np.sum(total_topcut), 0)

    for i, deck in enumerate(deck_names):
        if total_initial[i] > 0:
            results[deck] = {
                "day2_conversion": float(day2_conv[i]),
                "top_cut_conversion": float(topcut_conv[i]),
                "win_probability": float(win_conv[i]),
                "day2_share": float(day2_share[i]),
                "top_cut_share": float(topcut_share[i])
            }
            
    return results