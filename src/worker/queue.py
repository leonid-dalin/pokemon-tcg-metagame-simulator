import json
from huey import SqliteHuey, crontab
from src.api.models import ScrapedMatrix
from src.scraper import fetch_live_matchup_data, build_complete_matchup_matrix
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

@huey.periodic_task(crontab(minute='0', hour='2'))
def automated_daily_pipeline():
    print("Starting automated daily Limitless TCG data scrape...")

    from src.core.urls import ASC_URLS, POR_URLS
    TARGET_URLS = ASC_URLS

    try:
        canonical_map = {}
        raw_matchups = fetch_live_matchup_data(TARGET_URLS, canonical_map)

        if not raw_matchups:
            raise ValueError("Scraper returned zero matchups. Limitless HTML structure may have changed.")

        matrix_data = build_complete_matchup_matrix(raw_matchups)

        payload_for_validation = {
            "format_name": "Standard",
            "archetypes": [
                {
                    "archetype_name": arch,
                    "matchups": matrix_data["matchup_matrix"][arch]
                }
                for arch in matrix_data["archetypes"]
            ]
        }

        validated_data = ScrapedMatrix(**payload_for_validation)

        output_path = INPUT_DATA

        dumped_data = validated_data.model_dump()

        final_json_structure = {
            "archetypes": matrix_data["archetypes"],
            "win_rate_matrix": {
                arch["archetype_name"]: arch["matchups"]
                for arch in dumped_data["archetypes"]
            }
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(final_json_structure, f, indent=2)

        print(f"Pipeline successful! Matrix updated and thermodynamic purity verified.")

    except ValueError as ve:
        print(f"DATA VALIDATION FAILED. Aborting update. Reason: {ve}")
    except Exception as e:
        print(f"CRITICAL PIPELINE FAILURE: {e}")