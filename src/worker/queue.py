import json
import os

from huey import RedisHuey, crontab

from src.api.models import ScrapedMatrix, TIER_MAPPING, PredictionRequest
from src.core.config import INPUT_DATA, MIN_GAMES
from src.core.data import load_matchup_data
from src.core.scraper import fetch_live_matchup_data, build_complete_matchup_matrix
from src.tournament.monte_carlo import run_monte_carlo_analytics
from src.tournament.solver import predict_best_decks, get_variant_5_structure, swiss_rounds_from_players

redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/?db=0")
huey = RedisHuey('tcg_tasks', url=redis_url)

@huey.task()
def execute_simulation_job(payload: dict):
    """
    This runs in a background worker process, completely freeing up the API
    """
    # 1. Instantiate and Validate Request First
    request = PredictionRequest(**payload)
    job_id = request.job_id

    deck_names, win_matrix, _ = load_matchup_data(INPUT_DATA, MIN_GAMES)

    # Extract clean, validated parameters
    iterations = TIER_MAPPING.get(request.precision_tier, 25_000)
    players = request.total_players

    # 2. Run Solver (Water-filling & Baseline Constraints)
    solver_res = predict_best_decks(request)

    # 3. Determine Tournament Structure
    if request.tournament_style == "championship_series":
        d1_rounds, cut_points, d2_rounds, top_cut = get_variant_5_structure(players)
    else:
        d1_rounds = swiss_rounds_from_players(players)
        cut_points, d2_rounds, top_cut = 99, 0, (8 if players >= 8 else 0)

    def _progress_handler(current_chunk: int, total_chunks: int):
        pct = int((current_chunk / total_chunks) * 100)
        huey.storage.put_data(f"prog_{job_id}", str(pct).encode('utf-8'))

    # 4. Run Monte Carlo Brackets
    mc_res = run_monte_carlo_analytics(
        deck_names=deck_names,
        win_matrix=win_matrix,
        meta_distribution=solver_res["full_meta"],
        d1_rounds=d1_rounds,
        cut_points=cut_points,
        d2_rounds=d2_rounds,
        top_cut=top_cut,
        players=players,
        iterations=iterations,
        match_format=request.match_format,
        use_tie_convergence=request.use_tie_convergence,
        global_tie_rate=request.global_tie_rate,
        use_drop_feature=request.use_drop_feature,
        progress_callback=_progress_handler
    )

    return {
        "solver_results": solver_res,
        "mc_results": mc_res
    }

@huey.periodic_task(crontab(minute='0', hour='2'))
def automated_daily_pipeline():
    print("Starting automated daily Limitless TCG data scrape...")

    from src.core.urls import POR_URLS
    target_urls = POR_URLS

    try:
        canonical_map = {}
        raw_matchups = fetch_live_matchup_data(target_urls, canonical_map)

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
        dumped_data = validated_data.model_dump()

        final_json_structure = {
            "archetypes": matrix_data["archetypes"],
            "win_rate_matrix": {
                arch["archetype_name"]: arch["matchups"]
                for arch in dumped_data["archetypes"]
            }
        }

        with open(INPUT_DATA, "w", encoding="utf-8") as f:
            json.dump(final_json_structure, f, indent=2)

        print(f"Pipeline successful! Matrix updated and thermodynamic purity verified.")

    except ValueError as ve:
        print(f"DATA VALIDATION FAILED. Aborting update. Reason: {ve}")
    except Exception as e:
        print(f"CRITICAL PIPELINE FAILURE: {e}")