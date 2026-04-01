#!/usr/bin/env python3
# monte_carlo.py | High-speed static bracket execution (Rust-Powered)
import time
import numpy as np
import tcg_engine
from typing import Dict, List, Optional, Callable


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

    # 1. Flatten the meta distribution to match the matrix indices
    meta_vec = np.zeros(n_decks)
    for i, name in enumerate(deck_names):
        meta_vec[i] = meta_distribution.get(name, 0.0)

    meta_sum = np.sum(meta_vec)
    if meta_sum > 0:
        meta_vec = meta_vec / meta_sum

        # 2. Convert BO1 win rates to BO3 realities mathematically
    working_matrix = win_matrix.copy()
    if match_format == "BO3":
        working_matrix = 3 * (working_matrix ** 2) - 2 * (working_matrix ** 3)

    base_seed = int(time.time() * 1000) % (1 << 32)

    if progress_callback:
        # start signal
        progress_callback(0, 1)

    # 3. Call the Rust Engine
    res_init, res_day2, res_top, res_champ = tcg_engine.run_parallel_monte_carlo(
        iterations,
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

    if progress_callback:
        # fire completion signal
        progress_callback(1, 1)

    # 4. Convert Rust vectors (Vec<usize>) back to NumPy arrays
    total_initial = np.array(res_init, dtype=int)
    total_day2 = np.array(res_day2, dtype=int)
    total_topcut = np.array(res_top, dtype=int)
    total_champ = np.array(res_champ, dtype=int)

    # 5. Calculate Conversion Rates & Metagame Shares safely
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