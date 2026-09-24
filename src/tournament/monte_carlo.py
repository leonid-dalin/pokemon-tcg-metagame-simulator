#!/usr/bin/env python3
# monte_carlo.py | High-speed static bracket execution (Rust-Powered)
import multiprocessing
import os

import structlog
import numpy as np
import tcg_engine
from scipy.stats import beta as beta_distribution
from typing import Any, Dict, List, Optional, Callable, Tuple

from src.core.runtime import get_container_cores
from src.api.models import GLOBAL_TIE_RATE
from src.core.telemetry import tracer
from src.core.config import (
    BDIF_COVERAGE_RATIO,
    BDIF_MIN_MATCHES,
    BDIF_PAIR_MIN_GAMES,
    BDIF_POSTERIOR_DRAWS,
    BDIF_PRIOR_STRENGTH,
    BDIF_PANEL_DECKS,
)

logger = structlog.get_logger()
safe_cores = max(1, get_container_cores())
os.environ["RAYON_NUM_THREADS"] = str(safe_cores)


def build_hierarchical_beta_posteriors(
        deck_names: List[str],
        win_matrix: np.ndarray,
        matchup_details: Dict[Tuple[str, str], Dict[str, Any]],
        prior_strength: float = BDIF_PRIOR_STRENGTH,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Build matchup posteriors and identify decks with insufficient evidence."""
    n_decks = len(deck_names)
    index = {name: i for i, name in enumerate(deck_names)}
    totals = np.zeros(n_decks, dtype=float)
    wins = np.zeros(n_decks, dtype=float)
    covered = np.zeros(n_decks, dtype=float)

    for (deck, opponent), details in matchup_details.items():
        i = index.get(deck)
        if i is None:
            continue
        matches = max(0, int(details.get("match_count", 0)))
        totals[i] += matches
        wins[i] += float(details.get("win_rate", 0.5)) * matches
        if matches >= BDIF_PAIR_MIN_GAMES:
            covered[i] += matches

    field_wr = np.divide(wins, totals, out=np.full(n_decks, 0.5), where=totals > 0)
    coverage = np.divide(covered, totals, out=np.zeros(n_decks), where=totals > 0)
    insufficient = [
        deck for i, deck in enumerate(deck_names)
        if totals[i] < BDIF_MIN_MATCHES or coverage[i] < BDIF_COVERAGE_RATIO
    ]

    alpha = np.zeros((n_decks, n_decks), dtype=float)
    beta = np.zeros((n_decks, n_decks), dtype=float)
    for i, deck in enumerate(deck_names):
        alpha[i, i] = beta[i, i] = 1.0
        for j in range(i + 1, n_decks):
            opponent = deck_names[j]
            forward = matchup_details.get((deck, opponent), {})
            reverse = matchup_details.get((opponent, deck), {})
            forward_matches = max(0, int(forward.get("match_count", 0)))
            reverse_matches = max(0, int(reverse.get("match_count", 0)))
            forward_wr = float(forward.get("win_rate", win_matrix[i, j]))
            reverse_wr = float(reverse.get("win_rate", 1.0 - win_matrix[i, j]))
            observed_total = forward_matches + reverse_matches
            if observed_total:
                observed_wr = (
                    forward_wr * forward_matches
                    + (1.0 - reverse_wr) * reverse_matches
                ) / observed_total
                effective_matches = max(forward_matches, reverse_matches)
            else:
                observed_wr = float(win_matrix[i, j])
                effective_matches = 0
            prior = (float(field_wr[i]) + float(field_wr[j])) / 2.0
            pair_alpha = observed_wr * effective_matches + prior_strength * prior
            pair_beta = (1.0 - observed_wr) * effective_matches + prior_strength * (1.0 - prior)
            pair_alpha = max(pair_alpha, np.finfo(float).eps)
            pair_beta = max(pair_beta, np.finfo(float).eps)
            alpha[i, j], beta[i, j] = pair_alpha, pair_beta
            alpha[j, i], beta[j, i] = pair_beta, pair_alpha
    return alpha, beta, insufficient


def build_matchup_panel(
        deck_names: List[str],
        meta_vector: np.ndarray,
        alpha: np.ndarray,
        beta: np.ndarray,
        matchup_details: Dict[Tuple[str, str], Dict[str, Any]],
        panel_decks: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Build analytical posterior rows for configured decks against the field."""
    requested = list(panel_decks if panel_decks is not None else BDIF_PANEL_DECKS)
    index = {deck: i for i, deck in enumerate(deck_names)}
    top_indices = np.argsort(-meta_vector, kind="stable")[:6]
    top_decks = [deck_names[i] for i in top_indices]
    rows: Dict[str, List[Dict[str, Any]]] = {}
    unmatched: List[str] = []

    for deck in requested:
        if deck not in index:
            unmatched.append(deck)
            continue
        deck_index = index[deck]
        opponents = [opponent for opponent in top_decks if opponent != deck]
        if deck in top_decks:
            opponents = opponents[:5]
            opponents.append(deck)
        deck_rows = []
        for opponent in opponents:
            if opponent == deck:
                deck_rows.append({
                    "opponent": opponent,
                    "mean": 0.5,
                    "lower": 0.5,
                    "upper": 0.5,
                    "match_count": 0,
                    "reliable": True,
                    "mirror": True,
                })
                continue
            opponent_index = index[opponent]
            posterior_alpha = float(alpha[deck_index, opponent_index])
            posterior_beta = float(beta[deck_index, opponent_index])
            forward = matchup_details.get((deck, opponent), {})
            reverse = matchup_details.get((opponent, deck), {})
            match_count = max(
                int(forward.get("match_count", 0)),
                int(reverse.get("match_count", 0)),
            )
            deck_rows.append({
                "opponent": opponent,
                "mean": posterior_alpha / (posterior_alpha + posterior_beta),
                "lower": float(beta_distribution.ppf(0.025, posterior_alpha, posterior_beta)),
                "upper": float(beta_distribution.ppf(0.975, posterior_alpha, posterior_beta)),
                "match_count": match_count,
                "reliable": match_count >= BDIF_PAIR_MIN_GAMES,
                "mirror": False,
            })
        rows[deck] = deck_rows

    return {"rows": rows, "unmatched": unmatched, "opponents": top_decks}


def run_monte_carlo_analytics(
        deck_names: List[str],
        win_matrix: np.ndarray,
        meta_distribution: Dict[str, float],
        d1_rounds: int,
        cut_points: int,
        d2_rounds: int,
        top_cut: int,
        players: int = 256,
        iterations: int = 25_000,
        match_format: str = "BO3",
        progress_callback: Optional[Callable[[int, int], None]] = None,
        use_tie_convergence: bool = True,
        global_tie_rate: float = GLOBAL_TIE_RATE,
        use_drop_feature: bool = False,
        seed: int = 0,
        matchup_details: Optional[Dict[Tuple[str, str], Dict[str, Any]]] = None,
        posterior_draws: int = BDIF_POSTERIOR_DRAWS,
        panel_decks: Optional[List[str]] = None,
) -> Dict[str, Any]:
    if not hasattr(run_monte_carlo_analytics, "_rayon_initialized"):
        try:
            tcg_engine.initialize_rayon(safe_cores)
            run_monte_carlo_analytics._rayon_initialized = True
        except Exception:
            pass

    n_decks = len(deck_names)
    if n_decks == 0:
        return {
            "metrics": {},
            "ranked_metrics": {},
            "insufficient_data": [],
            "matchup_panel": {
                "rows": {},
                "unmatched": list(panel_decks if panel_decks is not None else BDIF_PANEL_DECKS),
                "opponents": [],
            },
        }

    meta_vec = np.zeros(n_decks)
    for i, name in enumerate(deck_names):
        meta_vec[i] = meta_distribution.get(name, 0.0)

    meta_sum = np.sum(meta_vec)
    if meta_sum > 0:
        meta_vec = meta_vec / meta_sum
    else:
        logger.warning("empty_meta_distribution_using_uniform_field", deck_count=n_decks)
        meta_vec.fill(1.0 / n_decks)

    use_posterior = matchup_details is not None
    if use_posterior:
        alpha, beta, insufficient_data = build_hierarchical_beta_posteriors(
            deck_names, win_matrix, matchup_details
        )
        base_iterations, remainder = divmod(iterations, posterior_draws)
        draw_sizes = [
            base_iterations + (1 if i < remainder else 0)
            for i in range(posterior_draws)
        ]
        draw_count = posterior_draws

        def matrix_sampler(draw_index: int) -> np.ndarray:
            rng = np.random.default_rng(seed + draw_index)
            working_matrix = np.zeros((n_decks, n_decks), dtype=float)
            for i in range(n_decks):
                working_matrix[i, i] = 0.5
                for j in range(i + 1, n_decks):
                    sample = float(rng.beta(alpha[i, j], beta[i, j]))
                    working_matrix[i, j] = sample
                    working_matrix[j, i] = 1.0 - sample
            return working_matrix
    else:
        insufficient_data = []
        base_chunk_size = 10000 if iterations >= 10000 else iterations
        chunks = max(1, iterations // base_chunk_size)
        remainder = iterations % base_chunk_size
        draw_sizes = [
            base_chunk_size + (remainder if i == chunks - 1 else 0)
            for i in range(chunks)
        ]
        draw_count = chunks

        def matrix_sampler(draw_index: int) -> np.ndarray:
            return win_matrix.copy()

    draw_specs = [(size, i) for i, size in enumerate(draw_sizes)]
    draw_starts = np.cumsum([0, *draw_sizes[:-1]]).tolist()

    # Init empty tracking arrays for the aggregated totals
    total_initial = np.zeros(n_decks, dtype=int)
    total_day2 = np.zeros(n_decks, dtype=int)
    total_topcut = np.zeros(n_decks, dtype=int)
    total_champ = np.zeros(n_decks, dtype=int)

    draw_metrics = []
    for current_chunk, draw_index in draw_specs:
        working_matrix = matrix_sampler(draw_index)
        if match_format == "BO3":
            working_matrix = 3 * (working_matrix ** 2) - 2 * (working_matrix ** 3)

        if current_chunk == 0:
            continue

        # Ensure a unique, deterministic seed per tournament
        base_seed = (seed + int(draw_starts[draw_index])) % (1 << 32)

        with tracer.start_as_current_span("rust_tcg_engine_batch") as rust_span:
            rust_span.set_attribute("chunk.size", current_chunk)
            rust_span.set_attribute("chunk.index", draw_index)

            res_init, res_day2, res_top, res_champ = tcg_engine.run_parallel_monte_carlo(
                current_chunk,
                players,
                meta_vec.tolist(),
                working_matrix.tolist(),
                d1_rounds,
                cut_points,
                d2_rounds,
                top_cut,
                base_seed,
                use_tie_convergence,
                global_tie_rate,
                use_drop_feature
            )
        logger.debug("chunk_processed", chunk_index=draw_index, size=current_chunk)
        total_initial += np.array(res_init, dtype=int)
        total_day2 += np.array(res_day2, dtype=int)
        total_topcut += np.array(res_top, dtype=int)
        total_champ += np.array(res_champ, dtype=int)

        # Fire progress state back to Huey
        with np.errstate(divide="ignore", invalid="ignore"):
            draw_initial = np.array(res_init, dtype=float)
            draw_day2 = np.array(res_day2, dtype=float)
            draw_top = np.array(res_top, dtype=float)
            draw_champ = np.array(res_champ, dtype=float)
            draw_metrics.append({
                "day2_share": np.divide(draw_day2, draw_day2.sum(), out=np.zeros(n_decks), where=draw_day2.sum() > 0),
                "top_cut_share": np.divide(draw_top, draw_top.sum(), out=np.zeros(n_decks), where=draw_top.sum() > 0),
                "win_probability": np.divide(draw_champ, draw_initial, out=np.zeros(n_decks), where=draw_initial > 0),
            })

        if progress_callback:
            progress_callback(draw_index + 1, draw_count)

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

    draw_array = {key: np.array([draw[key] for draw in draw_metrics]) for key in ("day2_share", "top_cut_share", "win_probability")}
    if use_posterior and draw_metrics:
        for i, deck in enumerate(deck_names):
            if deck not in results:
                continue
            for metric, values in draw_array.items():
                results[deck][f"{metric}_lower"] = float(np.quantile(values[:, i], 0.025))
                results[deck][f"{metric}_upper"] = float(np.quantile(values[:, i], 0.975))
    ranked_metrics = {
        deck: metrics for deck, metrics in results.items()
        if deck not in insufficient_data
    }
    return {
        "metrics": results,
        "ranked_metrics": ranked_metrics,
        "insufficient_data": insufficient_data,
        "matchup_panel": build_matchup_panel(
            deck_names,
            meta_vec,
            alpha,
            beta,
            matchup_details or {},
            panel_decks=panel_decks,
        ) if use_posterior else {
            "rows": {},
            "unmatched": list(panel_decks if panel_decks is not None else BDIF_PANEL_DECKS),
            "opponents": [],
        },
    }
