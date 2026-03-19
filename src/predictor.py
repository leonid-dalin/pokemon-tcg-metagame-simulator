# src/predictor.py
import os
import math
import numpy as np
from typing import Dict, List, Any, TypedDict, Literal, Union, cast, Tuple
from .config import INPUT_DIR, MIN_GAMES
from .data import load_matchup_data, safe_normalize

# === Input Types ===
class ExactSpec(TypedDict):
    exact: float

class RangeSpec(TypedDict):
    min: float
    max: float

MetaValue = Union[float, ExactSpec, RangeSpec]
UserMetaSpec = Dict[str, MetaValue]
MatchFormat = Literal["BO1", "BO3"]

# === Output Types ===
class DeckRecommendation(TypedDict):
    deck: str
    expected_win_rate: float
    confidence: float
    sample_support: float
    meta_share: float
    is_user_specified: bool
    sos: float
    omw: float
    undefeated_probability: float
    composite_score: float

class PredictionResult(TypedDict):
    recommendations: List[DeckRecommendation]
    avoid: List[DeckRecommendation]
    full_meta: Dict[str, float]
    metrics_per_deck: Dict[str, Any]
    swiss_rounds: int
    total_players: int
    frontrunners: List[str]

def swiss_rounds_from_players(n_players: int) -> int:
    if n_players <= 1:
        return 1
    return min(9, math.ceil(math.log2(n_players)))

def apply_bo3_conversion(win_matrix: np.ndarray) -> np.ndarray:
    """Converts BO1 win probabilities to BO3 using P_bo3 = 3p^2 - 2p^3"""
    return 3 * (win_matrix ** 2) - 2 * (win_matrix ** 3)

def calculate_empirical_baseline(
    deck_names: List[str], 
    matchup_details: Dict[Tuple[str, str], Dict[str, Any]], 
    random_mass_fraction: float = 0.02
) -> np.ndarray:
    """Calculates Live meta share based on real match volume with Laplace smoothing."""
    n = len(deck_names)
    deck_to_idx = {name: i for i, name in enumerate(deck_names)}
    match_counts = np.zeros(n)

    for (d1, d2), details in matchup_details.items():
        if d1 in deck_to_idx:
            match_counts[deck_to_idx[d1]] += details.get("match_count", 0)

    total_matches = np.sum(match_counts)
    if total_matches == 0:
        return np.ones(n) / n

    # Distribute actual matches
    empirical_share = match_counts / total_matches
    
    # Inject random mass to prevent 0% baselines
    smoothed_share = empirical_share * (1.0 - random_mass_fraction) + (random_mass_fraction / n)
    return safe_normalize(smoothed_share)

def resolve_meta_constraints(
    baseline_meta: np.ndarray, 
    user_spec: UserMetaSpec, 
    deck_to_idx: Dict[str, int]
) -> np.ndarray:
    """Vectorized iterative water-filling algorithm to strictly enforce user Min/Max/Exact bounds."""
    n = len(baseline_meta)
    final_meta = np.zeros(n)
    
    min_bounds = np.zeros(n)
    max_bounds = np.ones(n)
    is_locked = np.zeros(n, dtype=bool)

    # 1. Parse Specs (O(N) - Fast enough to keep as a simple loop)
    for deck, spec in user_spec.items():
        if deck not in deck_to_idx:
            continue
        i = deck_to_idx[deck]
        
        if isinstance(spec, (int, float)):
            val = float(spec)
            min_bounds[i] = max_bounds[i] = val
            final_meta[i] = val
            is_locked[i] = True
        elif isinstance(spec, dict):
            if "exact" in spec:
                val = float(spec["exact"])
                min_bounds[i] = max_bounds[i] = val
                final_meta[i] = val
                is_locked[i] = True
            elif "min" in spec and "max" in spec:
                min_bounds[i] = float(spec["min"])
                max_bounds[i] = float(spec["max"])

    # 2. Pure Vectorized Water-Filling
    max_iterations = 100
    
    for iterations in range(max_iterations):
        remaining_mass = 1.0 - np.sum(final_meta[is_locked])
        
        if remaining_mass <= 1e-8:
            break
            
        unlocked_mask = ~is_locked
        if not np.any(unlocked_mask):
            break
            
        # Proportional baseline for unlocked decks
        unlocked_baseline = baseline_meta[unlocked_mask]
        baseline_sum = np.sum(unlocked_baseline)
        
        if baseline_sum == 0:
            unlocked_baseline = np.ones(np.sum(unlocked_mask))
            baseline_sum = np.sum(unlocked_baseline)
            
        proposed_alloc = (unlocked_baseline / baseline_sum) * remaining_mass
        
        # Vectorized bounds checking
        over_mask = proposed_alloc > max_bounds[unlocked_mask]
        under_mask = proposed_alloc < min_bounds[unlocked_mask]
        
        if not (np.any(over_mask) or np.any(under_mask)):
            final_meta[unlocked_mask] = proposed_alloc
            break # All unlocked decks fit beautifully
            
        # Map the local unlocked masks back to global indices to apply locks
        unlocked_indices = np.where(unlocked_mask)[0]
        
        over_global = unlocked_indices[over_mask]
        under_global = unlocked_indices[under_mask]
        
        # Apply the bounds and lock them
        final_meta[over_global] = max_bounds[over_global]
        is_locked[over_global] = True
        
        final_meta[under_global] = min_bounds[under_global]
        is_locked[under_global] = True

    if iterations == max_iterations - 1:
        print("⚠️ Warning: Vectorized water-filling hit max iterations. Precision loss likely.")

    return safe_normalize(final_meta)

def predict_best_decks(
    user_meta_spec: UserMetaSpec,
    total_players: int = 32,
    min_sample_threshold: int = 10,
    match_format: MatchFormat = "BO1"
) -> PredictionResult:
    
    input_path = os.path.join(INPUT_DIR, "ea_input.json")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input data not found: {input_path}")

    deck_names, win_matrix, matchup_details = load_matchup_data(input_path, MIN_GAMES)
    n = len(deck_names)
    deck_to_idx = {name: i for i, name in enumerate(deck_names)}

    # Apply Format Transformation
    if match_format == "BO3":
        win_matrix = apply_bo3_conversion(win_matrix)

    # 1. Calculate Empirical Baseline ("Live" Mode)
    live_baseline = calculate_empirical_baseline(deck_names, matchup_details)

    # 2. Apply Custom Bounds via Water-Filling
    meta_vec = resolve_meta_constraints(live_baseline, user_meta_spec, deck_to_idx)

    # --- Compute performance metrics ---
    sample_matrix = np.full((n, n), 100.0)
    for (d1, d2), details in matchup_details.items():
        if d1 in deck_to_idx and d2 in deck_to_idx:
            sample_matrix[deck_to_idx[d1], deck_to_idx[d2]] = details.get("match_count", 100)

    expected_wr = win_matrix @ meta_vec
    weighted_samples = np.array([np.sum(meta_vec * sample_matrix[i]) for i in range(n)])
    confidence = np.clip(weighted_samples / (weighted_samples + min_sample_threshold), 0.2, 1.0)

    # --- Enhanced Swiss Metrics ---
    sos_values = 1.0 - meta_vec
    total_weighted_wr = np.sum(meta_vec * expected_wr)
    opponent_wr_sums = total_weighted_wr - (meta_vec * expected_wr)
    denominators = 1.0 - meta_vec

    global_avg_wr = np.mean(expected_wr)
    omw_values = np.full_like(meta_vec, global_avg_wr)
    safe_mask = denominators > 1e-6
    omw_values[safe_mask] = opponent_wr_sums[safe_mask] / denominators[safe_mask]

    swiss_rounds = swiss_rounds_from_players(total_players)
    undefeated_probabilities = np.power(expected_wr, swiss_rounds)

    # Composite Score
    raw_composite = (
        0.4 * expected_wr -
        0.1 * sos_values -
        0.1 * omw_values +
        0.4 * undefeated_probabilities
    )

    # Normalize to 0-100 scale to match Tier Thresholds (S: 90, A: 70, B: 50, C: 30)
    raw_min = np.min(raw_composite)
    raw_max = np.max(raw_composite)
    
    if raw_max > raw_min:
        composite_scores = ((raw_composite - raw_min) / (raw_max - raw_min)) * 100.0
    else:
        composite_scores = np.full_like(raw_composite, 50.0)

    metrics_per_deck: Dict[str, Any] = {}
    for i, name in enumerate(deck_names):
        metrics_per_deck[name] = {
            "expected_win_rate": float(expected_wr[i]),
            "confidence": float(confidence[i]),
            "sample_support": float(weighted_samples[i]),
            "meta_share": float(meta_vec[i]),
            "is_user_specified": name in user_meta_spec,
            "sos": float(sos_values[i]),
            "omw": float(omw_values[i]),
            "undefeated_probability": float(undefeated_probabilities[i]),
            "composite_score": float(composite_scores[i]), # Now a 0-100 float
        }

    all_decks_sorted = sorted(deck_names, key=lambda d: metrics_per_deck[d]["composite_score"], reverse=True)

    frontrunners = [
        d for d in all_decks_sorted[:5]
        if composite_scores[deck_to_idx[d]] > 0.6 and metrics_per_deck[d]["meta_share"] > 0.03
    ][:2]

    recommendations = [cast(DeckRecommendation, {**metrics_per_deck[d], "deck": d}) for d in all_decks_sorted]
    avoid = [cast(DeckRecommendation, {**metrics_per_deck[d], "deck": d}) for d in all_decks_sorted[::-1]]
    full_meta = {deck_names[i]: float(meta_vec[i]) for i in range(n)}

    return {
        "recommendations": recommendations,
        "avoid": avoid,
        "full_meta": full_meta,
        "metrics_per_deck": metrics_per_deck,
        "swiss_rounds": swiss_rounds,
        "total_players": total_players,
        "frontrunners": frontrunners,
    }