#!/usr/bin/env python3
# monte_carlo.py | High-speed static bracket execution
import time
import numpy as np
import multiprocessing as mp
from typing import Dict, List, Tuple, Optional, Callable

def _mc_worker(args: Tuple) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    iterations, players, meta_distribution, win_matrix, d1_rounds, cut_points, d2_rounds, top_cut, seed, use_tie_convergence, global_tie_rate, use_drop_feature = args
    rng = np.random.default_rng(seed)
    n_decks = win_matrix.shape[0]
    
    total_initial = np.zeros(n_decks, dtype=int)
    total_day2 = np.zeros(n_decks, dtype=int)
    total_topcut = np.zeros(n_decks, dtype=int)
    total_champ = np.zeros(n_decks, dtype=int)
    
    match_points = np.zeros(players, dtype=int)
    active = np.arange(players)
    
    for _ in range(iterations):
        field_indices: np.ndarray = np.asarray(rng.choice(n_decks, size=players, p=meta_distribution))
        total_initial += np.bincount(field_indices, minlength=n_decks)

        match_points.fill(0)
        opponents_history: List[List[int]] = [[] for _ in range(players)]
        losses = np.zeros(players, dtype=int) if use_drop_feature else None

        def play_rounds(rounds, current_active):
            for _ in range(rounds):
                if len(current_active) < 2: break
                rng.shuffle(current_active)
                order = np.argsort(-match_points[current_active], kind='mergesort')
                sorted_active = current_active[order]
                
                p1s = sorted_active[0::2]
                p2s = sorted_active[1::2]
                if len(p1s) > len(p2s): p1s = p1s[:-1]
                
                # --- HEURISTIC REMATCH PREVENTION ---
                for idx in range(len(p1s)):
                    p1, p2 = p1s[idx], p2s[idx]
                    if p2 in opponents_history[p1]:
                        if idx + 1 < len(p2s):
                            next_p2 = p2s[idx + 1]
                            if next_p2 not in opponents_history[p1] and p2 not in opponents_history[p1s[idx+1]]:
                                p2s[idx], p2s[idx+1] = p2s[idx+1], p2s[idx]
                
                p1_decks = field_indices[p1s]
                p2_decks = field_indices[p2s]
                win_probs = win_matrix[p1_decks, p2_decks]
                
                # --- TIE CONVERGENCE LOGIC ---
                if use_tie_convergence:
                    tie_probs = global_tie_rate * 4.0 * win_probs * (1.0 - win_probs)
                    rolls = rng.random(len(p1s))
                    
                    p1_win_thresh = win_probs - (tie_probs / 2.0)
                    tie_thresh = p1_win_thresh + tie_probs
                    
                    p1_wins = rolls < p1_win_thresh
                    p2_wins = rolls >= tie_thresh
                    ties = ~(p1_wins | p2_wins)
                    
                    match_points[p1s[p1_wins]] += 3
                    match_points[p2s[p2_wins]] += 3
                    match_points[p1s[ties]] += 1
                    match_points[p2s[ties]] += 1

                    if use_drop_feature and losses is not None:
                        losses[p1s[p2_wins]] += 1
                        losses[p2s[p1_wins]] += 1
                else:
                    p1_basic_wins = rng.random(len(p1s)) < win_probs
                    match_points[p1s[p1_basic_wins]] += 3
                    match_points[p2s[~p1_basic_wins]] += 3
                    if use_drop_feature and losses is not None:
                        losses[p1s[~p1_basic_wins]] += 1
                        losses[p2s[p1_basic_wins]] += 1

                for p1, p2 in zip(p1s, p2s):
                    p1_idx, p2_idx = int(p1), int(p2)
                    opponents_history[p1_idx].append(p2_idx)
                    opponents_history[p2_idx].append(p1_idx)

                # --- X-3 DROP LOGIC ---
                if use_drop_feature and losses is not None:
                    current_active = current_active[losses[current_active] < 3]

        # 1. Play Day 1
        play_rounds(d1_rounds, active)
        
        day2_players = np.where(match_points >= cut_points)[0]
        if len(day2_players) > 0 and d2_rounds > 0:
            total_day2 += np.bincount(field_indices[day2_players], minlength=n_decks)
        
        # 2. Play Day 2
        if d2_rounds > 0 and len(day2_players) > 1:
            play_rounds(d2_rounds, day2_players)
            
        # 3. Calculate OWP (Top Cut sorting)
        top_players = np.array([], dtype=int)
        if top_cut > 0:
            owp = np.zeros(players)
            pool_for_owp = np.array(day2_players if d2_rounds > 0 else active, dtype=int)
            if len(pool_for_owp) > 0:
                for i in pool_for_owp:
                    opps = opponents_history[i]
                    if not opps: continue
                    opp_win_pcts = np.clip(
                        np.array([match_points[o] / (max(1, len(opponents_history[o])) * 3.0) for o in opps],
                                 dtype=float), 0.25, 1.0)
                    owp[i] = np.mean(opp_win_pcts)

                top_order = np.lexsort((-owp[pool_for_owp], -match_points[pool_for_owp]))
                top_players = np.array(pool_for_owp[top_order][:top_cut], dtype=int)
                total_topcut += np.bincount(field_indices[top_players], minlength=n_decks)
            
        # 4. Playoffs
        if len(top_players) > 0:
            standings = top_players.copy()
            while len(standings) > 1:
                half = len(standings) // 2
                tc_p1s = standings[:half]
                tc_p2s = standings[half:][::-1]
                
                unpaired = []
                if len(tc_p2s) > len(tc_p1s):
                    unpaired.append(tc_p2s[-1])
                    tc_p2s = tc_p2s[:-1]
                elif len(tc_p1s) > len(tc_p2s):
                    unpaired.append(tc_p1s[-1])
                    tc_p1s = tc_p1s[:-1]
                
                tc_p1_decks = field_indices[tc_p1s]
                tc_p2_decks = field_indices[tc_p2s]
                tc_p1_wins = rng.random(len(tc_p1s)) < win_matrix[tc_p1_decks, tc_p2_decks]
                
                next_round = []
                for i, p1_won in enumerate(tc_p1_wins):
                    next_round.append(tc_p1s[i] if p1_won else tc_p2s[i])
                
                next_round.extend(unpaired)  # advancing the player with the bye
                standings = np.array(next_round, dtype=int)
                
            if len(standings) > 0:
                total_champ[field_indices[int(standings[0])]] += 1

    return total_initial, total_day2, total_topcut, total_champ, np.zeros(n_decks), np.zeros(n_decks)

def run_monte_carlo_analytics(
    deck_names: List[str],
    win_matrix: np.ndarray,
    meta_distribution: Dict[str, float],
    d1_rounds: int,
    cut_points: int,
    d2_rounds: int,
    top_cut: int,
    players: int = 256,
    iterations: int = 10_000,
    match_format: str = "BO3",
    progress_callback: Optional[Callable[[int, int], None]] = None,
    use_tie_convergence: bool = True,
    global_tie_rate: float = 0.15,
    use_drop_feature: bool = False
) -> Dict[str, Dict[str, float]]:
    
    n_decks = len(deck_names)
    if n_decks == 0:
        return {}
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
            tasks.append((alloc, players, meta_vec, working_matrix, d1_rounds, cut_points, d2_rounds, top_cut, base_seed + i, use_tie_convergence, global_tie_rate, use_drop_feature))
            remaining_iters -= alloc
        
    total_initial = np.zeros(n_decks, dtype=int)
    total_day2 = np.zeros(n_decks, dtype=int)
    total_topcut = np.zeros(n_decks, dtype=int)
    total_champ = np.zeros(n_decks, dtype=int)
    
    completed_cores = 0
    ctx = mp.get_context('spawn')
    with ctx.Pool(n_cores) as pool:
        for res_init, res_day2, res_top, res_champ, _, _ in pool.imap_unordered(_mc_worker, tasks):
            total_initial += res_init
            total_day2 += res_day2
            total_topcut += res_top
            total_champ += res_champ
   
            completed_cores += 1
            if progress_callback:
                progress_callback(completed_cores, len(tasks))
            
    results = {}
    with np.errstate(divide='ignore', invalid='ignore'):
        day2_conv: np.ndarray = np.asarray(np.where(total_initial > 0, total_day2 / total_initial, 0))
        topcut_conv: np.ndarray = np.asarray(np.where(total_initial > 0, total_topcut / total_initial, 0))
        win_conv: np.ndarray = np.asarray(np.where(total_initial > 0, total_champ / total_initial, 0))

        day2_share: np.ndarray = np.asarray(np.where(np.sum(total_day2) > 0, total_day2 / np.sum(total_day2), 0))
        topcut_share: np.ndarray = np.asarray(
            np.where(np.sum(total_topcut) > 0, total_topcut / np.sum(total_topcut), 0))
    for i, deck in enumerate(deck_names):
        if total_initial[i] > 0:
            results[deck] = {
                "day2_conversion": float(day2_conv[i]),
                "top_cut_conversion": float(topcut_conv[i]),
                "win_probability": float(win_conv[i]),
                "day2_share": float(day2_share[i]),
                "top_cut_share": float(topcut_share[i]),
            }
            
    return results