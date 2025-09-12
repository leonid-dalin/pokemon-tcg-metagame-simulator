#!/usr/bin/env python3
# analysis.py — Post-simulation analysis: tier lists, convergence, clustering, diagnostics.
from __future__ import annotations
import logging
import numpy as np
from typing import List, Dict, Any, Optional
from scipy.stats import rankdata
from sklearn.metrics.pairwise import cosine_similarity
from .config import STABILITY_THRESHOLD

# ----------------------------
# Convergence & Stability Metrics
# ----------------------------
def compute_convergence_metrics(history: List[np.ndarray], stability_threshold: float = STABILITY_THRESHOLD) -> Dict[str, Any]:
    """Quantify how quickly and stably the metagame converged.
    Args:
        history (List[np.ndarray]): List of frequency vectors over generations.
        stability_threshold (float): Threshold below which change is considered “stable”.
    Returns:
        Dict with:
            - convergence_generation: int or None
            - avg_change_after_convergence: float
            - time_to_stability: int (generations)
            - oscillation_index: float (std of changes post-convergence)
    """
    if len(history) < 2:
        return {
            "convergence_generation": None,
            "avg_change_after_convergence": 0.0,
            "time_to_stability": len(history),
            "oscillation_index": 0.0,
            "max_oscillation": 0.0
        }

    # Calculate the maximum absolute change between consecutive generations
    changes = [
        float(np.max(np.abs(history[i] - history[i-1])))
        for i in range(1, len(history))
    ]

    # Efficiently find convergence by scanning backwards (O(N))
    convergence_gen = None
    for i in range(len(changes) - 1, -1, -1):  # Start from the last change and go backwards
        if changes[i] > stability_threshold:
            # Convergence happened at the generation *after* the last unstable change
            convergence_gen = i + 2  # +2 because `i` is index in `changes`, which starts from gen 1
            break

    # If no change was above the threshold, it converged from the start
    if convergence_gen is None:
        convergence_gen = 1

    # Handle the case where convergence is at or beyond the last generation
    if convergence_gen >= len(history):
        convergence_gen = None  # Mark as not formally converged within the simulation window
        post_conv_changes = []
    else:
        # Get all changes from the convergence generation onwards
        post_conv_changes = changes[convergence_gen - 1:]  # -1 because `changes` index is gen-1

    # Calculate metrics for the post-convergence period
    if len(post_conv_changes) > 0:
        avg_post_conv = float(np.mean(post_conv_changes))
        osc_index = float(np.std(post_conv_changes))
        max_oscillation = float(np.max(post_conv_changes))
    else:
        # If no post-convergence data (e.g., converged at the end), use 0.0
        avg_post_conv = 0.0
        osc_index = 0.0
        max_oscillation = 0.0

    metrics = {
        "convergence_generation": convergence_gen,
        "avg_change_after_convergence": avg_post_conv,
        "time_to_stability": convergence_gen if convergence_gen is not None else len(history),
        "oscillation_index": osc_index,
        "max_oscillation": max_oscillation
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
    winrate_weight: float = 0.6
) -> Dict[str, List[Dict[str, Any]]]:
    """Generate tier list based on final metagame state.
    Prioritises meta-weighted win rate, with presence as secondary factor.
    Mimics tier lists from masterduelmeta.com or similar.
    Args:
        deck_names (List[str]): List of deck names.
        metagame_history (List[np.ndarray]): Evolution history of deck frequencies.
        win_matrix (np.ndarray): Win-rate matrix.
        presence_weight (float): Weight for presence in composite score.
        winrate_weight (float): Weight for win rate in composite score.
    Returns:
        Dict mapping tier (S, A, B, C, D) to list of deck data dicts.
    """
    if not metagame_history:
        return {tier: [] for tier in "SABCD"}

    final_freqs = metagame_history[-1]
    # Calculate meta-weighted win rate using the final frequency vector
    meta_weighted_win_rate = win_matrix.dot(final_freqs)
    n = len(deck_names)

    score = (
        rankdata(meta_weighted_win_rate) / n * winrate_weight +
        rankdata(final_freqs) / n * presence_weight
    )

    tier_thresholds = {'S': 0.95, 'A': 0.85, 'B': 0.70, 'C': 0.50}
    tiers = {tier: [] for tier in "SABCD"}

    for i in range(n):
        if final_freqs[i] <= 1e-6:
            continue  # Skip functionally extinct decks

        deck_data = {
            "deck": deck_names[i],
            "score": float(score[i]),
            "win_rate": float(meta_weighted_win_rate[i]),
            "presence": float(final_freqs[i]) # Use final_freqs for presence
        }

        assigned = False
        for tier, threshold in tier_thresholds.items():
            if score[i] >= threshold:
                tiers[tier].append(deck_data)
                assigned = True
                break
        if not assigned:
            tiers['D'].append(deck_data)

    # Sort tiers by score descending
    for tier in tiers:
        tiers[tier].sort(key=lambda x: x['score'], reverse=True)

    logging.info("🏆 Final State Tier List Generated")
    for tier in "SABCD":
        if tiers[tier]:
            top_deck = tiers[tier][0]["deck"]
            logging.info(f"  {tier}-Tier Top: {top_deck} (WR: {tiers[tier][0]['win_rate']:.2%})")

    # --- Log full S, A, D tiers ---
    for target_tier in ['S', 'A', 'D']:
        logging.info(f"  Full {target_tier}-Tier List:")
        for deck_data in tiers[target_tier]:
            logging.info(f"    - {deck_data['deck']} (Score: {deck_data['score']:.4f}, WR: {deck_data['win_rate']:.2%}, Presence: {deck_data['presence']:.2%})")

    return tiers


def generate_all_time_tier_list(
    deck_names: List[str],
    metagame_history: List[np.ndarray],
    win_matrix: np.ndarray
) -> Dict[str, List[Dict[str, Any]]]:
    """Generate tier list based on entire simulation history — “career performance”.
    Incorporates win rate, presence, consistency, and meta impact.
    Args:
        deck_names (List[str]): List of deck names.
        metagame_history (List[np.ndarray]): Full history of frequencies.
        win_matrix (np.ndarray): Win-rate matrix.
    Returns:
        Dict mapping tier to list of deck data dicts.
    """
    n = len(deck_names)
    num_gens = len(metagame_history)
    if num_gens == 0:
        return {tier: [] for tier in "SABCD"}

    total_metagame = np.zeros(n)
    win_rates = np.zeros(n)
    peak_presence = np.zeros(n)

    for freqs in metagame_history:
        total_metagame += freqs
        win_rates += win_matrix.dot(freqs)
        peak_presence = np.maximum(peak_presence, freqs)

    total_metagame /= num_gens
    win_rates /= num_gens

    # Consistency = mean / std (coefficient of variation inverse)
    freq_history = np.array(metagame_history)
    consistency = np.array([
        mean / (std_val + 1e-9) if (mean := np.mean(freq_history[:, i])) > 1e-6 and (std_val := np.std(freq_history[:, i])) > 0 else 0.0
        for i in range(n)
    ])

    # Composite score
    normalized_win = rankdata(win_rates) / n
    normalized_presence = rankdata(total_metagame) / n
    normalized_consistency = rankdata(consistency) / n

    composite_score = (
        normalized_win * 0.50 +
        normalized_presence * 0.30 +
        normalized_consistency * 0.20
    )

    # Meta impact metric
    meta_impact = win_rates * total_metagame * (1.0 + np.tanh(consistency - 1.0))

    tier_thresholds = {'S': 0.75, 'A': 0.60, 'B': 0.40, 'C': 0.25}
    tiers = {tier: [] for tier in "SABCD"}

    for i in range(n):
        deck_data = {
            "deck": deck_names[i],
            "composite_score": float(composite_score[i]),
            "win_rate": float(win_rates[i]),
            "presence": float(total_metagame[i]),
            "consistency": float(consistency[i]),
            "meta_impact": float(meta_impact[i])
        }

        if composite_score[i] >= tier_thresholds['S']:
            tiers['S'].append(deck_data)
        elif composite_score[i] >= tier_thresholds['A']:
            tiers['A'].append(deck_data)
        elif composite_score[i] >= tier_thresholds['B']:
            tiers['B'].append(deck_data)
        elif composite_score[i] >= tier_thresholds['C']:
            tiers['C'].append(deck_data)
        else:
            tiers['D'].append(deck_data)

    for tier in tiers:
        tiers[tier].sort(key=lambda x: x['composite_score'], reverse=True)

    logging.info("🏅 All-Time Tier List Generated")
    for tier in "SABCD":
        if tiers[tier]:
            top = tiers[tier][0]
            logging.info(f"  {tier}-Tier Leader: {top['deck']} (Impact: {top['meta_impact']:.4f})")

    # --- Log full S, A, D tiers ---
    for target_tier in ['S', 'A', 'D']:
        logging.info(f"  Full {target_tier}-Tier List:")
        for deck_data in tiers[target_tier]:
            logging.info(f"    - {deck_data['deck']} (Score: {deck_data['composite_score']:.4f}, WR: {deck_data['win_rate']:.2%}, Presence: {deck_data['presence']:.2%}, Consistency: {deck_data['consistency']:.4f})")

    return tiers

# ----------------------------
# Matchup Graph Analysis
# ----------------------------
def compute_matchup_cycles(win_matrix: np.ndarray, deck_names: List[str], cycle_length: int = 3) -> List[List[str]]:
    """Identify unique rock-paper-scissors cycles in the metagame.
    Finds unique cycles of decks where A > B > C > A in win rate.
    Uses itertools.combinations to iterate over unique groups of decks, checking both possible cycle directions.
    This avoids redundant checks and eliminates the need for deduplication via sorting and sets.
    Args:
        win_matrix (np.ndarray): Win-rate matrix.
        deck_names (List[str]): Deck names.
        cycle_length (int): Length of cycles to detect (default 3).
    Returns:
        List of unique cycles, each cycle is a list of deck names in order (e.g., [A, B, C] for A->B->C->A).
    """
    n = len(deck_names)
    cycles = []  # List to store unique cycles

    if cycle_length != 3:
        logging.warning("Only 3-cycles implemented currently.")
        return cycles

    from itertools import combinations

    win_threshold = 0.6

    # Iterate over all unique combinations of 3 deck indices
    for i, j, k in combinations(range(n), 3):
        # Check for cycle: i -> j -> k -> i
        if (win_matrix[i, j] > win_threshold and
            win_matrix[j, k] > win_threshold and
            win_matrix[k, i] > win_threshold):
            cycles.append([deck_names[i], deck_names[j], deck_names[k]])

        # Check for the reverse cycle: i -> k -> j -> i
        elif (win_matrix[i, k] > win_threshold and
              win_matrix[k, j] > win_threshold and
              win_matrix[j, i] > win_threshold):
            cycles.append([deck_names[i], deck_names[k], deck_names[j]])

    logging.info(f"🌀 Found {len(cycles)} unique RPS-style 3-cycles in matchup graph.")
    for cycle in cycles[:3]:
        logging.info(f"  → {' → '.join(cycle)} → {cycle[0]}")

    return cycles

# ----------------------------
# Deck Archetype Similarity
# ----------------------------
def compute_deck_similarity(
    win_matrix: np.ndarray,
    deck_names: List[str],
    extinction_gens: Optional[List[Optional[int]]] = None,
    final_active_mask: Optional[List[bool]] = None
) -> np.ndarray:
    """Compute pairwise similarity between decks based on their matchup profiles.
    Uses cosine similarity of win-rate vectors. Optionally filters out extinct decks.
    Performs K-Means clustering on active decks to identify strategic archetypes.
    Args:
        win_matrix (np.ndarray): n x n win-rate matrix.
        deck_names (List[str]): Deck names.
        extinction_gens (Optional[List[Optional[int]]]): List indicating extinction generation for each deck. None or a list of length n.
        final_active_mask (Optional[List[bool]]): List indicating if deck is active in the final state. If provided, overrides extinction_gens for filtering.
    Returns:
        np.ndarray: n x n similarity matrix (0 to 1).
    """
    n = len(deck_names)

    # use final_active_mask if provided, otherwise fall back to extinction_gens
    if final_active_mask is not None and len(final_active_mask) == n:
        active_mask = np.array(final_active_mask)
    elif extinction_gens is not None and len(extinction_gens) == n:
        active_mask = np.array([g is None for g in extinction_gens])
    else:
        active_mask = np.array([True] * n)  # Default: all decks are active

    if np.any(active_mask) and len(np.where(active_mask)[0]) >= 2:
        active_indices = np.where(active_mask)[0]
        # Extract win-rate submatrix for active decks only
        active_win_matrix = win_matrix[np.ix_(active_indices, active_indices)]
        active_deck_names = [deck_names[i] for i in active_indices]

        # Compute similarity on the active subset
        active_similarity = cosine_similarity(active_win_matrix)
        np.fill_diagonal(active_similarity, 1.0)

        # Initialize full similarity matrix
        similarity = np.zeros((n, n))
        # Place active similarities back into the full matrix
        for i, idx_i in enumerate(active_indices):
            for j, idx_j in enumerate(active_indices):
                similarity[idx_i, idx_j] = active_similarity[i, j]

        # Set similarity between extinct decks and others to 0.0 (or handle as needed)
        # Diagonal for extinct decks is set to 1.0
        for i in range(n):
            if not active_mask[i]:
                similarity[i, i] = 1.0

        # --- Perform Clustering on Active Decks Only ---
        try:
            from .data import cluster_decks_by_matchup_profile
            # Determine number of clusters dynamically (min 2, max 5, or sqrt of active decks)
            n_clusters = max(2, min(5, int(np.ceil(np.sqrt(len(active_indices))))))
            cluster_result = cluster_decks_by_matchup_profile(
                win_matrix=active_win_matrix,
                deck_names=active_deck_names,
                n_clusters=n_clusters,
                method="kmeans"
            )
            # The clustering function already logs the clusters, so no need to re-log here.
        except Exception as e:
            logging.warning(f"⚠️  Clustering failed: {e}")
    else:
        # If no active decks or not enough, proceed with all decks (original behavior)
        similarity = cosine_similarity(win_matrix)
        np.fill_diagonal(similarity, 1.0)
        # --- Perform Clustering on All Decks (fallback) ---
        try:
            from .data import cluster_decks_by_matchup_profile
            n_clusters = max(2, min(5, int(np.ceil(np.sqrt(n)))))
            cluster_result = cluster_decks_by_matchup_profile(
                win_matrix=win_matrix,
                deck_names=deck_names,
                n_clusters=n_clusters,
                method="kmeans"
            )
        except Exception as e:
            logging.warning(f"⚠️  Clustering failed: {e}")

    # Log most similar pairs (excluding self, only upper triangle to avoid duplicates)
    # Create list of (similarity, row, col) for upper triangle
    pairs = []
    for r in range(n):
        for c in range(r + 1, n):  # Only upper triangle
            #  Only log pairs if both are active according to active_mask
            if active_mask[r] and active_mask[c]:
                pairs.append((similarity[r, c], r, c))

    # Sort by similarity descending
    pairs.sort(key=lambda x: x[0], reverse=True)

    # Log top 5 *valid* pairs
    logged_count = 0
    for i, (sim_val, r, c) in enumerate(pairs):
        if logged_count >= 5:
            break
        logging.info(f"🤝 {deck_names[r]} ≈ {deck_names[c]} (similarity: {sim_val:.3f})")
        logged_count += 1

    return similarity