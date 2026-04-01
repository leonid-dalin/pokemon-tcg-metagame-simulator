#!/usr/bin/env python3
# analysis.py | Post-simulation analysis: tier lists, convergence, diagnostics, metrics, cycle detection
from __future__ import annotations

import logging
import numpy as np
from typing import List, Dict, Any, Optional

from src.core.config import (
    STABILITY_THRESHOLD,
    WIN_THRESHOLD,
    CONVERGENCE_WINDOW,
    TIER_ORDER,
    TIER_THRESHOLDS
)

# ----------------------------
# Module-Level Constants (analysis.py)
# ----------------------------
DEFAULT_CONVERGENCE_KWARGS: Dict[str, Any] = {
    "stability_threshold": STABILITY_THRESHOLD,
    "convergence_window": CONVERGENCE_WINDOW,
}

DEFAULT_CYCLE_KWARGS: Dict[str, Any] = {
    "cycle_length": 3,
    "win_threshold": WIN_THRESHOLD,
}

# ----------------------------
# Convergence & Stability Metrics
# ----------------------------
def compute_convergence_metrics(
        history: List[np.ndarray], stability_threshold: float = STABILITY_THRESHOLD
) -> Dict[str, Any]:
    if len(history) < 2:
        return {
            "convergence_generation": None,
            "avg_change_after_convergence": 0.0,
            "time_to_stability": len(history),
            "oscillation_index": 0.0,
            "max_oscillation": 0.0,
        }

    history_arr = np.array(history)
    changes = np.max(np.abs(np.diff(history_arr, axis=0)), axis=1)
    unstable_indices = np.where(changes > stability_threshold)[0]

    if len(unstable_indices) == 0:
        convergence_gen = 1
        post_conv_changes = changes
    else:
        last_unstable_idx = unstable_indices[-1]
        convergence_gen = last_unstable_idx + 2 
        
        if convergence_gen >= len(history):
            convergence_gen = None
            post_conv_changes = changes
        else:
            post_conv_changes = changes[last_unstable_idx + 1:]

    if len(post_conv_changes) > 0:
        avg_post_conv = float(np.mean(post_conv_changes))
        osc_index = float(np.std(post_conv_changes))
        max_oscillation = float(np.max(post_conv_changes))
    else:
        avg_post_conv = osc_index = max_oscillation = 0.0

    metrics = {
        "convergence_generation": convergence_gen,
        "avg_change_after_convergence": avg_post_conv,
        "time_to_stability": convergence_gen if convergence_gen is not None else len(history),
        "oscillation_index": osc_index,
        "max_oscillation": max_oscillation,
    }
    
    logging.info(f"📈 Convergence at gen {convergence_gen if convergence_gen is not None else 'N/A'} | Avg post-conv change: {avg_post_conv:.2e}")
    return metrics

# ----------------------------
# Tier List Generator
# ----------------------------
def generate_final_state_tier_list(
        deck_names: List[str],
        metagame_history: List[np.ndarray],
        win_matrix: np.ndarray
) -> Dict[str, List[Dict[str, Any]]]:
    """Generate tier list based on final metagame state using Meta Score logic."""
    n = len(deck_names)
    
    if win_matrix.shape[0] != n or win_matrix.shape[1] != n:
        raise ValueError(f"win_matrix shape {win_matrix.shape} does not match deck_names length {n}")
        
    if not metagame_history:
        return {tier: [] for tier in TIER_ORDER}
        
    final_freqs = metagame_history[-1]
    if len(final_freqs) != n:
        raise ValueError(f"final_freqs length {len(final_freqs)} does not match deck_names length {n}")
        
    expected_wr = win_matrix.dot(final_freqs)

    max_wr = float(np.max(expected_wr).item())
    min_wr_floor = float(1.0 - max_wr)

    if max_wr > min_wr_floor:
        power_scores: np.ndarray = np.asarray(
            np.clip((expected_wr - min_wr_floor) / (max_wr - min_wr_floor) * 100.0, 0.0, 100.0)
        )
    else:
        power_scores: np.ndarray = np.asarray(np.full(n, 50.0))

    max_freq = float(np.max(final_freqs).item())
    freq_scores: np.ndarray = np.zeros(n, dtype=float)
    if max_freq > 0.0:
        freq_scores = np.asarray((final_freqs / max_freq) * 100.0)

    meta_scores: np.ndarray = np.asarray((power_scores + freq_scores) / 2.0)
    
    tiers = {tier: [] for tier in TIER_ORDER}

    for i in range(n):
        if final_freqs[i] <= 1e-6:
            continue
            
        deck_data = {
            "deck": deck_names[i],
            "score": float(meta_scores[i]), 
            "power_score": float(power_scores[i]),
            "frequency_score": float(freq_scores[i]),
            "win_rate": float(expected_wr[i]),
            "presence": float(final_freqs[i]), 
        }
        
        for tier, threshold in TIER_THRESHOLDS.items():
            if expected_wr[i] >= threshold: 
                tiers[tier].append(deck_data)
                break
            
    for tier in tiers:
        tiers[tier].sort(key=lambda x: x["score"], reverse=True)
        
    logging.info("🏆 Final State Tier List Generated")
    for tier in TIER_ORDER:
        if tiers[tier]:
            top_deck = tiers[tier][0]["deck"]
            logging.info(f"  {tier}-Tier Top: {top_deck} (WR: {tiers[tier][0]['win_rate']:.2%})")
            
    return tiers

# ----------------------------
# Matchup Graph Analysis
# ----------------------------
def compute_matchup_cycles(
        win_matrix: np.ndarray, 
        deck_names: List[str], 
        cycle_length: int = DEFAULT_CYCLE_KWARGS["cycle_length"],
        win_threshold: float = DEFAULT_CYCLE_KWARGS["win_threshold"],
        final_active_mask: Optional[List[bool]] = None
) -> List[List[str]]:
    """Identify unique rock-paper-scissors cycles strictly within the active metagame."""
    n = len(deck_names)
    
    if win_matrix.shape[0] != n or win_matrix.shape[1] != n:
        raise ValueError(f"win_matrix shape {win_matrix.shape} does not match deck_names length {n}")
        
    cycles = [] 
    if cycle_length != 3:
        logging.warning("Only 3-cycles implemented currently.")
        return cycles
        
    active_mask = np.array(final_active_mask) if final_active_mask is not None else np.array([True] * n)
        
    from itertools import combinations
    for i, j, k in combinations(range(n), 3):
        if not (active_mask[i] and active_mask[j] and active_mask[k]):
            continue
            
        if win_matrix[i, j] > win_threshold and win_matrix[j, k] > win_threshold and win_matrix[k, i] > win_threshold:
            cycles.append([deck_names[i], deck_names[j], deck_names[k]])
            
        if win_matrix[i, k] > win_threshold and win_matrix[k, j] > win_threshold and win_matrix[j, i] > win_threshold:
            cycles.append([deck_names[i], deck_names[k], deck_names[j]])
            
    logging.info(f"🌀 Found {len(cycles)} unique RPS-style 3-cycles in the active matchup graph.")
    return cycles

def debug_print_rps_cycles(win_matrix: np.ndarray, deck_names: List[str], final_active_mask: Optional[List[bool]] = None) -> None:
    """Standalone debug function to print all identified RPS cycles with their exact win rates."""
    cycles = compute_matchup_cycles(win_matrix, deck_names, final_active_mask=final_active_mask)
    
    if not cycles:
        logging.info("--- RPS CYCLE DEBUG REPORT: 0 Cycles Found ---")
        return
        
    logging.info(f"--- RPS CYCLE DEBUG REPORT ({len(cycles)} Found) ---")
    for cycle in cycles:
        d1, d2, d3 = cycle
        i, j, k = deck_names.index(d1), deck_names.index(d2), deck_names.index(d3)
        
        wr1 = win_matrix[i, j]
        wr2 = win_matrix[j, k]
        wr3 = win_matrix[k, i]
        
        logging.info(f"  🔄 {d1} ({wr1:.1%}) -> {d2} ({wr2:.1%}) -> {d3} ({wr3:.1%}) -> {d1}")
    logging.info("-------------------------------------------------")

# ----------------------------
# Deck Archetype Similarity
# ----------------------------
def compute_deck_similarity(
        win_matrix: np.ndarray,
        deck_names: List[str],
        extinction_gens: Optional[List[Optional[int]]] = None,
        final_active_mask: Optional[List[bool]] = None,
) -> np.ndarray:
    """Compute pairwise strategic similarity using Pearson Correlation (TCG Accurate)."""
    n = len(deck_names)
    
    if n < 2:
            return np.ones((n, n))

    if final_active_mask is not None and len(final_active_mask) == n:
        active_mask = np.array(final_active_mask)
    elif extinction_gens is not None and len(extinction_gens) == n:
        active_mask = np.array([g is None for g in extinction_gens])
    else:
        active_mask = np.array([True] * n) 

    similarity = np.zeros((n, n))
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.any(active_mask) and len(np.where(active_mask)[0]) >= 2:
            active_indices = np.where(active_mask)[0]
            active_deck_profiles = win_matrix[active_indices, :]

            active_similarity = np.corrcoef(active_deck_profiles)
            active_similarity = np.nan_to_num(active_similarity, nan=0.0)
            
            similarity[np.ix_(active_indices, active_indices)] = active_similarity
            np.fill_diagonal(similarity, 1.0)
        else:
            similarity = np.corrcoef(win_matrix)
            similarity = np.nan_to_num(similarity, nan=0.0)
            np.fill_diagonal(similarity, 1.0)

    pairs = []
    for r in range(n):
        for c in range(r + 1, n): 
            if active_mask[r] and active_mask[c]:
                pairs.append((similarity[r, c], r, c))
                
    pairs.sort(key=lambda x: x[0], reverse=True)
    
    for i, (sim_val, r, c) in enumerate(pairs[:5]):
        # Correlation > 0.70 represents a genuinely strong strategic overlap in TCGs
        logging.info(f"🤝 {deck_names[r]} ≈ {deck_names[c]} (Pearson Correlation: {sim_val:.3f})")
        
    return similarity