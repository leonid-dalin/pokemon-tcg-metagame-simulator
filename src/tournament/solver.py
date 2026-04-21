# solver.py | Water-filling constraints & Base Meta Scoring
import bisect
import math
import numpy as np
import os
import structlog
from typing import Dict, List, Any, Tuple

from src.api.models import DeckRecommendation, PredictionRequest, PredictionResult
from src.core.config import INPUT_DATA, MIN_GAMES
import src.core.config as core_config
from src.core.data import load_matchup_data, safe_normalize
from src.core.telemetry import tracer

logger = structlog.get_logger()

# ==========================================
# Helper Funcs
# ==========================================
def get_variant_5_structure(players: int) -> Tuple[int, int, int, int]:
    """Returns: (Day1_Rounds, Match_Point_Cutoff, Day2_Rounds, Top_Cut) based on official handbook."""
    _STRUCTURE_THRESHOLDS = getattr(core_config, "_STRUCTURE_THRESHOLDS")
    _STRUCTURE_RESULTS = getattr(core_config, "_STRUCTURE_RESULTS")
    index = bisect.bisect_left(_STRUCTURE_THRESHOLDS, players)
    index = min(index, len(_STRUCTURE_RESULTS) - 1)
    return _STRUCTURE_RESULTS[index]

def swiss_rounds_from_players(n_players: int) -> int:
    if n_players < 4:
        return 0
    natural_swiss = math.ceil(math.log2(n_players))
    if 16 < n_players <= 64:
        return natural_swiss + 1
    elif n_players > 64:
        return natural_swiss + 2
    return natural_swiss

def apply_bo3_conversion(win_matrix: np.ndarray) -> np.ndarray:
    """Converts BO1 win probabilities to BO3 using P_bo3 = 3p^2 - 2p^3"""
    return np.asarray(3 * (win_matrix ** 2) - 2 * (win_matrix ** 3), dtype=float)

# ==========================================
# Meta Analysis & Prediction Engine
# ==========================================
def calculate_empirical_baseline(deck_names: List[str],
                                 matchup_details: Dict[Tuple[str, str], Dict[str, Any]]) -> np.ndarray:
    """Calculates a baseline field distribution from historical match volumes."""
    n = len(deck_names)
    counts = np.zeros(n, dtype=float)
    deck_to_idx = {name: i for i, name in enumerate(deck_names)}

    for (d1, d2), details in matchup_details.items():
        if d1 in deck_to_idx:
            counts[deck_to_idx[d1]] += float(details.get("match_count", 0.0))

    return safe_normalize(counts)

def resolve_meta_constraints(
        live_baseline: np.ndarray,
        user_meta_spec: Dict[str, Any],
        deck_to_idx: Dict[str, int]
) -> np.ndarray:
    """
    Vectorized water-filling algorithm to resolve field constraints.
    Enforces exact, minimum, and maximum boundaries from the user configuration.
    """
    n = len(live_baseline)
    final_meta = np.zeros(n, dtype=float)
    exact_mask = np.zeros(n, dtype=bool)

    for deck, spec in user_meta_spec.items():
        if deck in deck_to_idx:
            idx = deck_to_idx[deck]
            val = 0.0

            if isinstance(spec, float) or isinstance(spec, int):
                val = float(spec)
            elif isinstance(spec, dict) and "exact" in spec:
                val = float(spec["exact"])
            elif hasattr(spec, "exact"):
                val = float(getattr(spec, "exact"))

            final_meta[idx] = val
            exact_mask[idx] = True

    remaining = 1.0 - float(np.sum(final_meta[exact_mask]))
    if remaining <= 0.0:
        return safe_normalize(final_meta)

    unfixed_mask = ~exact_mask
    if not bool(np.any(unfixed_mask)):
        return final_meta

    unfixed_baseline = safe_normalize(live_baseline[unfixed_mask])
    final_meta[unfixed_mask] = unfixed_baseline * remaining

    return safe_normalize(final_meta)

def predict_best_decks(request: PredictionRequest) -> dict:
    """
    Orchestrates static recommendations.
    Natively respects PredictionRequest while preserving all original mathematical scaling.
    """
    with tracer.start_as_current_span("predict_best_decks_orchestration"):
        user_meta_spec = request.user_meta_spec
        total_players = request.total_players
        min_sample_threshold = request.min_sample_threshold
        match_format = request.match_format

        input_path = os.path.join(INPUT_DATA)
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input data not found: {input_path}")

        # Load from disk specifically to get matchup_details (match counts for confidence)
        with tracer.start_as_current_span("load_disk_matchup_data"):
            disk_deck_names, disk_win_matrix, matchup_details = load_matchup_data(str(input_path), MIN_GAMES)

        # Use the matrix from the payload if it exists, otherwise fallback to disk
        if request.deck_names and request.matchup_matrix:
            deck_names = request.deck_names
            win_matrix = np.array(request.matchup_matrix, dtype=float)
        else:
            deck_names = disk_deck_names
            win_matrix = disk_win_matrix

        n = len(deck_names)
        deck_to_idx = {name: i for i, name in enumerate(deck_names)}

        if match_format == "BO3":
            win_matrix = apply_bo3_conversion(win_matrix)

        # Resolve baselines
        with tracer.start_as_current_span("resolve_water_filling_constraints"):
            live_baseline = calculate_empirical_baseline(deck_names, matchup_details)
            meta_vec = resolve_meta_constraints(live_baseline, user_meta_spec, deck_to_idx)

        # --- Compute performance metrics ---
        sample_matrix = np.full((n, n), 100.0, dtype=float)
        for (d1, d2), details in matchup_details.items():
            if d1 in deck_to_idx and d2 in deck_to_idx:
                sample_matrix[deck_to_idx[d1], deck_to_idx[d2]] = float(details.get("match_count", 100.0))

        expected_wr: np.ndarray = np.asarray(win_matrix @ meta_vec, dtype=float)
        weighted_samples: np.ndarray = np.asarray([np.sum(meta_vec * sample_matrix[i]) for i in range(n)], dtype=float)
        confidence: np.ndarray = np.asarray(np.clip(weighted_samples / (weighted_samples + min_sample_threshold), 0.2, 1.0),
                                            dtype=float)
        swiss_rounds = swiss_rounds_from_players(total_players)

        # --- Meta Score Analytics ---
        # Power Score
        max_wr = float(np.max(expected_wr))
        min_wr_floor = 1.0 - max_wr

        if max_wr > min_wr_floor:
            power_scores: np.ndarray = np.asarray(
                np.minimum((expected_wr - min_wr_floor) / (max_wr - min_wr_floor) * 100.0, 100.0), dtype=float)
        else:
            power_scores: np.ndarray = np.full(n, 50.0, dtype=float)

        # Frequency Score (0-100)
        max_freq = float(np.max(meta_vec))
        freq_scores: np.ndarray = np.zeros(n, dtype=float)
        if max_freq > 0.0:
            freq_scores = np.asarray((meta_vec / max_freq) * 100.0, dtype=float)

        # Base Meta Score
        base_meta_scores: np.ndarray = np.asarray((power_scores + freq_scores) / 2.0, dtype=float)

        metrics_per_deck: Dict[str, Any] = {}
        for i, name in enumerate(deck_names):
            metrics_per_deck[name] = {
                "expected_win_rate": float(expected_wr[i]),
                "confidence": float(confidence[i]),
                "sample_support": float(weighted_samples[i]),
                "meta_share": float(meta_vec[i]),
                "is_user_specified": name in user_meta_spec,
                "power_score": float(power_scores[i]),
                "frequency_score": float(freq_scores[i]),
                "base_meta_score": float(base_meta_scores[i]),
            }

        all_decks_sorted = sorted(deck_names, key=lambda d: metrics_per_deck[d]["base_meta_score"], reverse=True)
        logger.info("solver_recommendations_generated", total_decks=len(all_decks_sorted))
        frontrunners = [
            d for d in all_decks_sorted[:5]
            if metrics_per_deck[d]["base_meta_score"] > 60.0 and metrics_per_deck[d]["meta_share"] > 0.03
        ][:2]

        recommendations = [
            DeckRecommendation(deck=str(d), **metrics_per_deck[d])
            for d in all_decks_sorted
        ]

        avoid = [
            DeckRecommendation(deck=str(d), **metrics_per_deck[d])
            for d in reversed(all_decks_sorted)
        ]

        full_meta: Dict[str, float] = {str(deck_names[i]): float(meta_vec[i]) for i in range(n)}
        result = PredictionResult(
            recommendations=recommendations,
            avoid=avoid,
            full_meta=full_meta,
            metrics_per_deck=metrics_per_deck,
            swiss_rounds=swiss_rounds,
            total_players=total_players,
            frontrunners=frontrunners,
        )

        return result.model_dump()