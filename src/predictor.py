# src/predictor.py
import os
import math
import numpy as np
from typing import Dict, List, Any, TypedDict, Literal, Union, cast, Optional  # <--- Added Optional
from .config import (
    INPUT_DIR,
    MIN_GAMES,
    EXTINCTION_THRESHOLD,
    STABILITY_THRESHOLD,
    MAX_GENERATIONS,
    MIN_GENERATIONS_PROP,
    TOURNAMENT_SIZE,
    NUM_TOURNAMENTS_PER_GEN,
    NUM_ROUNDS,
    USE_BAYESIAN_WINRATES,
    USE_MULTIPROC,
    RNG_SEED,
    DYNAMIC_DECK_INTRO_PROB,
    MUTATION_FLOOR,
    NOISE_SCALE,
    SELECTION_PRESSURE,
    MAX_INACTIVE_GENERATIONS,
    CONVERGENCE_WINDOW,
)
from .data import load_matchup_data, safe_normalize
from .simulation import find_evolutionary_stable_state
from .simulation_config import SimulationConfig

# === Input Types ===
class ExactSpec(TypedDict):
    exact: float


class RangeSpec(TypedDict):
    min: float
    max: float


MetaValue = Union[float, ExactSpec, RangeSpec]
UserMetaSpec = Dict[str, MetaValue]
InferenceMode = Literal["casual", "pro"]


# === Output Types ===
class DeckRecommendation(TypedDict):
    deck: str
    expected_win_rate: float
    confidence: float
    sample_support: float
    meta_share: float
    is_user_specified: bool
    # --- Enhanced Swiss Metrics ---
    sos: float  # Strength of Schedule
    omw: float  # Opponent's Match Win Percentage
    undefeated_probability: float  # Probability of going undefeated in Swiss rounds
    composite_score: float  # Combined score using WR, SoS, OMW


class PredictionResult(TypedDict):
    recommendations: List[DeckRecommendation]
    avoid: List[DeckRecommendation]
    full_meta: Dict[str, float]
    metrics_per_deck: Dict[str, Any]
    swiss_rounds: int
    total_players: int
    frontrunners: List[str]


def swiss_rounds_from_players(n_players: int) -> int:
    """
    Calculates the number of Swiss rounds based on the number of players.
    Uses the standard log base 2 formula.
    """
    if n_players <= 1:
        return 1
    # Cap at 9 rounds (for 257-512 players), standard for large events
    return min(9, math.ceil(math.log2(n_players)))


def predict_best_decks(
        user_meta_spec: UserMetaSpec,
        total_players: int = 32,
        min_sample_threshold: int = 10,
        inference_mode: InferenceMode = "pro",
        # Allow passing a pre-computed "pro" meta to cache simulation
        fallback_meta_pro: Optional[np.ndarray] = None,
) -> PredictionResult:
    """
    Predict best decks given flexible meta constraints.
    Enhanced with metrics inspired by Swiss tournament performance indicators.
    """
    # --- Load data ---
    input_path = os.path.join(INPUT_DIR, "ea_input.json")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input data not found: {input_path}")

    deck_names, win_matrix, matchup_details = load_matchup_data(input_path, MIN_GAMES)
    n = len(deck_names)
    deck_to_idx = {name: i for i, name in enumerate(deck_names)}

    # --- Parse constraints ---
    fixed_meta = np.zeros(n)
    min_bounds = np.zeros(n)
    max_bounds = np.ones(n)

    for deck, spec in user_meta_spec.items():
        if deck not in deck_to_idx:
            continue
        i = deck_to_idx[deck]

        if isinstance(spec, (int, float)):
            val = float(spec)
            if not (0.0 <= val <= 1.0):
                raise ValueError(f"Proportion for {deck} must be in [0,1], got {val}")
            fixed_meta[i] = val
            min_bounds[i] = max_bounds[i] = val
        elif isinstance(spec, dict):
            if "exact" in spec:
                val = float(spec["exact"])
                if not (0.0 <= val <= 1.0):
                    raise ValueError(f"Exact proportion for {deck} must be in [0,1], got {val}")
                fixed_meta[i] = val
                min_bounds[i] = max_bounds[i] = val
            elif "min" in spec and "max" in spec:
                min_val = float(spec["min"])
                max_val = float(spec["max"])
                if not (0.0 <= min_val <= max_val <= 1.0):
                    raise ValueError(f"Invalid min/max for {deck}: [{min_val}, {max_val}]")
                min_bounds[i] = min_val
                max_bounds[i] = max_val
            else:
                raise ValueError(f"Invalid spec for {deck}: {spec}")
        else:
            raise TypeError(f"Unsupported spec type for {deck}: {type(spec)}")

    if fixed_meta.sum() > 1.0:
        raise ValueError("Sum of exact proportions exceeds 1.0")

    # --- Get fallback meta based on mode ---
    if inference_mode == "pro":
        # Use pre-computed meta if provided
        if fallback_meta_pro is not None:
            fallback_meta = fallback_meta_pro
        else:
            # Run the simulation (expensive)
            sim_config = SimulationConfig(
                mode="replicator",
                max_generations=MAX_GENERATIONS,
                min_generations=int(MAX_GENERATIONS * MIN_GENERATIONS_PROP),
                extinction_threshold=EXTINCTION_THRESHOLD,
                stability_threshold=STABILITY_THRESHOLD,
                convergence_window=CONVERGENCE_WINDOW,
                max_inactive_generations=MAX_INACTIVE_GENERATIONS,
                use_bayesian_winrates=USE_BAYESIAN_WINRATES,
                tournament_size=TOURNAMENT_SIZE,
                num_tournaments_per_gen=NUM_TOURNAMENTS_PER_GEN,
                num_rounds=NUM_ROUNDS,
                use_multiproc=USE_MULTIPROC,
                seed=RNG_SEED,
                dynamic_deck_intro_prob=DYNAMIC_DECK_INTRO_PROB,
                mutation_floor=MUTATION_FLOOR,
                noise_scale=NOISE_SCALE,
                selection_pressure=SELECTION_PRESSURE,
            )
            results, _, _ = find_evolutionary_stable_state(
                deck_names=deck_names,
                win_matrix=win_matrix,
                matchup_details=matchup_details,
                config=sim_config,
            )
            fallback_meta = np.array([r["frequency"] for r in results])
    else:  # "casual"
        fallback_meta = np.ones(n) / n

    fallback_meta = safe_normalize(fallback_meta)

    # --- Construct plausible meta ---
    meta_vec = fixed_meta.copy()
    remaining = 1.0 - meta_vec.sum()
    if remaining > 1e-8:
        # Get mask of decks that are *not* fixed
        free_mask = (min_bounds != max_bounds)
        fallback_free = fallback_meta.copy()
        # Zero out the fixed decks
        fallback_free[~free_mask] = 0.0

        if fallback_free.sum() > 0:
            fallback_free = safe_normalize(fallback_free)
        else:
            # Fallback if pro meta is 0 for all free decks
            fallback_free = free_mask.astype(float)
            fallback_free = safe_normalize(fallback_free)

        # Apply bounds and re-normalize the free portion
        fallback_free = np.clip(fallback_free, min_bounds, max_bounds)
        # Distribute remaining share according to bounded fallback
        fallback_free = safe_normalize(fallback_free) * remaining
        meta_vec += fallback_free

    meta_vec = safe_normalize(meta_vec)

    # --- Compute performance metrics ---
    sample_matrix = np.full((n, n), 100.0)
    for (d1, d2), details in matchup_details.items():
        if d1 in deck_to_idx and d2 in deck_to_idx:
            i, j = deck_to_idx[d1], deck_to_idx[d2]
            sample_matrix[i, j] = details.get("match_count", 100)

    expected_wr = win_matrix @ meta_vec
    weighted_samples = np.array([np.sum(meta_vec * sample_matrix[i]) for i in range(n)])
    confidence = np.clip(weighted_samples / (weighted_samples + min_sample_threshold), 0.2, 1.0)

    # --- Enhanced Swiss Metrics ---
    # SoS: Strength of Schedule. 1.0 - meta_vec[i]
    sos_values = 1.0 - meta_vec

    # OMW: Opponent's Match Win Percentage
    # <--- OPTIMIZATION: Vectorized OMW calculation (removed O(N^2) loop)
    total_weighted_wr = np.sum(meta_vec * expected_wr)
    # (scalar) - (vector) -> (vector of sums for all opponents)
    opponent_wr_sums = total_weighted_wr - (meta_vec * expected_wr)
    denominators = 1.0 - meta_vec

    # Use global average WR as a safe fallback for division by zero
    global_avg_wr = np.mean(expected_wr)
    omw_values = np.full_like(meta_vec, global_avg_wr)

    safe_mask = denominators > 1e-6
    omw_values[safe_mask] = opponent_wr_sums[safe_mask] / denominators[safe_mask]
    # --- End of Vectorization ---

    # Undefeated Probability: A proxy metric.
    swiss_rounds = swiss_rounds_from_players(total_players)
    undefeated_probabilities = np.power(expected_wr, swiss_rounds)

    # Composite Score: Combine WR, SoS, OMW, Undefeated Prob
    wr_weight = 0.4
    sos_weight = 0.1  # Slightly penalize high SoS (harder field)
    omw_weight = 0.1  # Slightly penalize high OMW (faced stronger decks)
    undef_weight = 0.4  # Heavily weight Undefeated Prob

    # Note: SoS and OMW are *subtracted* as higher values mean a harder field
    composite_scores = (
            wr_weight * expected_wr -
            sos_weight * sos_values -
            omw_weight * omw_values +
            undef_weight * undefeated_probabilities
    )

    # --- Assemble results with enhanced metrics ---
    metrics_per_deck: Dict[str, Any] = {}
    for i, name in enumerate(deck_names):
        metrics_per_deck[name] = {
            "expected_win_rate": float(expected_wr[i]),
            "confidence": float(confidence[i]),
            "sample_support": float(weighted_samples[i]),
            "meta_share": float(meta_vec[i]),
            "is_user_specified": name in user_meta_spec,
            # --- Enhanced Metrics ---
            "sos": float(sos_values[i]),
            "omw": float(omw_values[i]),
            "undefeated_probability": float(undefeated_probabilities[i]),
            "composite_score": float(composite_scores[i]),
        }

    # Sort decks by the NEW composite score for recommendations
    all_decks_sorted_by_composite = sorted(
        deck_names,
        key=lambda d: metrics_per_deck[d]["composite_score"],
        reverse=True,
    )

    # Frontrunners: high composite score AND high meta share
    frontrunners = [
                       d for d in all_decks_sorted_by_composite[:5]
                       # <--- FIX: Use composite_scores[deck_to_idx[d]] for check
                       if composite_scores[deck_to_idx[d]] > 0.6 and metrics_per_deck[d]["meta_share"] > 0.03
                   ][:2]

    # Recommendations and Avoid lists based on composite score
    recommendations = [
        cast(DeckRecommendation, {**metrics_per_deck[d], "deck": d})
        for d in all_decks_sorted_by_composite[:3]
    ]

    # Get "avoid" list from the same sorted list
    # Get last 3 decks, then reverse them (so worst is first)
    avoid = [
        cast(DeckRecommendation, {**metrics_per_deck[d], "deck": d})
        for d in all_decks_sorted_by_composite[-3:][::-1]
    ]

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