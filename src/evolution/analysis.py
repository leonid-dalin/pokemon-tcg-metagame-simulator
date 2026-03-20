#!/usr/bin/env python3
# analysis.py | Post-simulation analysis: tier lists, convergence, diagnostics, metrics, cycle detection
from __future__ import annotations

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from scipy.stats import rankdata
from sklearn.metrics.pairwise import cosine_similarity

from src.core.config import (
    STABILITY_THRESHOLD,
    CONSISTENCY_MEAN_EPSILON,
    CONSISTENCY_STD_EPSILON,
    TIER_S_THRESHOLD,
    TIER_A_THRESHOLD,
    TIER_B_THRESHOLD,
    TIER_C_THRESHOLD,
    COMPOSITE_SCORE_WR_WEIGHT,
    COMPOSITE_SCORE_PRESENCE_WEIGHT,
    COMPOSITE_SCORE_CONSISTENCY_WEIGHT,
    WIN_THRESHOLD,
    CONVERGENCE_WINDOW
)
from src.core.data import cluster_decks_by_matchup_profile

# ----------------------------
# Module-Level Constants (analysis.py)
# ----------------------------
tier_thresholds = {
    "S": TIER_S_THRESHOLD,
    "A": TIER_A_THRESHOLD,
    "B": TIER_B_THRESHOLD,
    "C": TIER_C_THRESHOLD,
}
TIER_ORDER: Tuple[str, ...] = ("S", "A", "B", "C", "D")

COMPOSITE_WEIGHTS: Tuple[float, float, float] = (
    COMPOSITE_SCORE_WR_WEIGHT,
    COMPOSITE_SCORE_PRESENCE_WEIGHT,
    COMPOSITE_SCORE_CONSISTENCY_WEIGHT,
)

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

    # Pure Vectorized History Diff (Instantaneous for 1M+ gens)
    history_arr = np.array(history)
    changes = np.max(np.abs(np.diff(history_arr, axis=0)), axis=1)
    
    # Find all indices where change exceeded the threshold
    unstable_indices = np.where(changes > stability_threshold)[0]

    if len(unstable_indices) == 0:
        # Never unstable
        convergence_gen = 1
        post_conv_changes = changes
    else:
        # Convergence is the generation after the *last* unstable spike
        last_unstable_idx = unstable_indices[-1]
        convergence_gen = last_unstable_idx + 2  # +1 for diff offset, +1 for "next gen"
        
        if convergence_gen >= len(history):
            convergence_gen = None
            post_conv_changes = np.array([])
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
# Tier List Generators
# ----------------------------
def generate_final_state_tier_list(
        deck_names: List[str],
        metagame_history: List[np.ndarray],
        win_matrix: np.ndarray,
        presence_weight: float = 0.4,
        winrate_weight: float = 0.6,
) -> Dict[str, List[Dict[str, Any]]]:
    """Generate tier list based on final metagame state, synced with global thresholds."""
    if not metagame_history:
        return {tier: [] for tier in TIER_ORDER}
        
    final_freqs = metagame_history[-1]
    meta_weighted_win_rate = win_matrix.dot(final_freqs)
    n = len(deck_names)
    
    score = (rankdata(meta_weighted_win_rate) / n * winrate_weight + rankdata(final_freqs) / n * presence_weight)
    
    tiers = {tier: [] for tier in TIER_ORDER}
    
    for i in range(n):
        if final_freqs[i] <= 1e-6:
            continue
            
        deck_data = {
            "deck": deck_names[i],
            "score": float(score[i]),
            "win_rate": float(meta_weighted_win_rate[i]),
            "presence": float(final_freqs[i]), 
        }
        
        assigned = False
        for tier, threshold in tier_thresholds.items():
            if score[i] >= threshold:
                tiers[tier].append(deck_data)
                assigned = True
                break
        if not assigned:
            tiers["D"].append(deck_data)
            
    for tier in tiers:
        tiers[tier].sort(key=lambda x: x["score"], reverse=True)
        
    logging.info("🏆 Final State Tier List Generated")
    for tier in TIER_ORDER:
        if tiers[tier]:
            top_deck = tiers[tier][0]["deck"]
            logging.info(f"  {tier}-Tier Top: {top_deck} (WR: {tiers[tier][0]['win_rate']:.2%})")
            
    return tiers

def generate_all_time_tier_list(
        deck_names: List[str], metagame_history: List[np.ndarray], win_matrix: np.ndarray
) -> Dict[str, List[Dict[str, Any]]]:
    """Generate tier list based on entire simulation history."""
    n = len(deck_names)
    if not metagame_history:
        return {tier: [] for tier in TIER_ORDER}

    freq_history = np.array(metagame_history)
    total_metagame = np.mean(freq_history, axis=0)

    payoffs_over_time = win_matrix @ freq_history.T
    win_rates = np.mean(payoffs_over_time, axis=1)

    mean_share = np.mean(freq_history, axis=0)
    std_dev_share = np.std(freq_history, axis=0)

    raw_consistency = np.divide(
        mean_share,
        std_dev_share + CONSISTENCY_STD_EPSILON,
        out=np.zeros_like(mean_share),
        where=std_dev_share > 0,
    )

    consistency = np.where(mean_share > CONSISTENCY_MEAN_EPSILON, raw_consistency, 0.0)

    normalized_win = rankdata(win_rates) / n
    normalized_presence = rankdata(total_metagame) / n
    normalized_consistency = rankdata(consistency) / n

    composite_score = (
            normalized_win * COMPOSITE_WEIGHTS[0]
            + normalized_presence * COMPOSITE_WEIGHTS[1]
            + normalized_consistency * COMPOSITE_WEIGHTS[2]
    )

    meta_impact = win_rates * total_metagame * (1.0 + np.tanh(consistency - 1.0))

    tiers = {tier: [] for tier in TIER_ORDER}

    for i in range(n):
        deck_data = {
            "deck": deck_names[i],
            "composite_score": float(composite_score[i]),
            "win_rate": float(win_rates[i]),
            "presence": float(total_metagame[i]),
            "consistency": float(consistency[i]),
            "meta_impact": float(meta_impact[i]),
        }
        score = composite_score[i]
        
        assigned = False
        for tier, threshold in tier_thresholds.items():
            if score >= threshold:
                tiers[tier].append(deck_data)
                assigned = True
                break
        if not assigned:
            tiers["D"].append(deck_data)

    for tier in tiers:
        tiers[tier].sort(key=lambda x: x["composite_score"], reverse=True)

    logging.info("🏅 All-Time Tier List Generated")
    return tiers

# ----------------------------
# Matchup Graph Analysis
# ----------------------------
def compute_matchup_cycles(
        win_matrix: np.ndarray, deck_names: List[str], cycle_length: int = 3
) -> List[List[str]]:
    """Identify unique rock-paper-scissors cycles in the metagame."""
    n = len(deck_names)
    cycles = [] 
    if cycle_length != 3:
        logging.warning("Only 3-cycles implemented currently.")
        return cycles
        
    from itertools import combinations
    for i, j, k in combinations(range(n), 3):
        if (win_matrix[i, j] > WIN_THRESHOLD and win_matrix[j, k] > WIN_THRESHOLD and win_matrix[k, i] > WIN_THRESHOLD):
            cycles.append([deck_names[i], deck_names[j], deck_names[k]])
        elif (win_matrix[i, k] > WIN_THRESHOLD and win_matrix[k, j] > WIN_THRESHOLD and win_matrix[j, i] > WIN_THRESHOLD):
            cycles.append([deck_names[i], deck_names[k], deck_names[j]])
            
    logging.info(f"🌀 Found {len(cycles)} unique RPS-style 3-cycles in matchup graph.")
    return cycles

# ----------------------------
# Deck Archetype Similarity
# ----------------------------
def compute_deck_similarity(
        win_matrix: np.ndarray,
        deck_names: List[str],
        extinction_gens: Optional[List[Optional[int]]] = None,
        final_active_mask: Optional[List[bool]] = None,
) -> np.ndarray:
    """Compute pairwise similarity using vectorized NumPy splicing."""
    n = len(deck_names)
    
    if final_active_mask is not None and len(final_active_mask) == n:
        active_mask = np.array(final_active_mask)
    elif extinction_gens is not None and len(extinction_gens) == n:
        active_mask = np.array([g is None for g in extinction_gens])
    else:
        active_mask = np.array([True] * n) 

    similarity = np.zeros((n, n))
    
    if np.any(active_mask) and len(np.where(active_mask)[0]) >= 2:
        active_indices = np.where(active_mask)[0]
        active_deck_names = [deck_names[i] for i in active_indices]
        active_deck_profiles = win_matrix[active_indices, :]

        active_similarity = cosine_similarity(active_deck_profiles)
        
        # Splicing in active similarities
        similarity[np.ix_(active_indices, active_indices)] = active_similarity

        # Universally set diagonal to 1.0
        np.fill_diagonal(similarity, 1.0)
                
        try:
            cluster_decks_by_matchup_profile(
                win_matrix=active_deck_profiles,
                deck_names=active_deck_names,
                n_clusters="auto",
                method="kmeans",
            )
        except Exception as e:
            logging.warning(f"⚠️  Clustering failed: {e}")
    else:
        similarity = cosine_similarity(win_matrix)
        np.fill_diagonal(similarity, 1.0)
        try:
            cluster_decks_by_matchup_profile(
                win_matrix=win_matrix,
                deck_names=deck_names,
                n_clusters="auto",
                method="kmeans",
            )
        except Exception as e:
            logging.warning(f"⚠️  Clustering failed: {e}")

    pairs = []
    for r in range(n):
        for c in range(r + 1, n): 
            if active_mask[r] and active_mask[c]:
                pairs.append((similarity[r, c], r, c))
                
    pairs.sort(key=lambda x: x[0], reverse=True)
    
    for i, (sim_val, r, c) in enumerate(pairs[:5]):
        logging.info(f"🤝 {deck_names[r]} ≈ {deck_names[c]} (similarity: {sim_val:.3f})")
        
    return similarity