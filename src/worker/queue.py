from huey import SqliteHuey
from src.core.config import INPUT_DATA, MIN_GAMES
from src.core.data import load_matchup_data
from src.tournament.solver import predict_best_decks, get_variant_5_structure, swiss_rounds_from_players
from src.tournament.monte_carlo import run_monte_carlo_analytics

huey = SqliteHuey(filename='tcg_tasks.db')

@huey.task()
def execute_simulation_job(payload: dict):
    """
    This runs in a background worker process, completely freeing up the API
    """
    deck_names, win_matrix, _ = load_matchup_data(INPUT_DATA, MIN_GAMES)

    # 1. Run Solver (Water-filling & Baseline Constraints)
    solver_res = predict_best_decks(
        user_meta_spec=payload["user_meta_spec"],
        total_players=payload["total_players"],
        min_sample_threshold=payload["min_sample_threshold"],
        match_format=payload["match_format"]
    )

    # 2. Determine Tournament Structure
    players = payload["total_players"]
    if payload.get("tournament_style", "championship_series") == "championship_series":
        d1_rounds, cut_points, d2_rounds, top_cut = get_variant_5_structure(players)
    else:
        d1_rounds = swiss_rounds_from_players(players)
        cut_points, d2_rounds, top_cut = 99, 0, (8 if players >= 8 else 0)

    # 3. Run Monte Carlo Brackets
    mc_res = run_monte_carlo_analytics(
        deck_names=deck_names,
        win_matrix=win_matrix,
        meta_distribution=solver_res["full_meta"],
        d1_rounds=d1_rounds,
        cut_points=cut_points,
        d2_rounds=d2_rounds,
        top_cut=top_cut,
        players=players,
        iterations=payload["mc_iterations"],
        match_format=payload["match_format"],
        use_tie_convergence=payload["use_tie_convergence"],
        global_tie_rate=payload["global_tie_rate"],
        use_drop_feature=payload["use_drop_feature"]
    )

    return {
        "solver_results": solver_res,
        "mc_results": mc_res
    }