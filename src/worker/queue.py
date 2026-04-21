import json
import os
import structlog
from opentelemetry import trace
from opentelemetry.instrumentation.redis import RedisInstrumentor
from huey import RedisHuey, crontab

from src.api.models import ScrapedMatrix, TIER_MAPPING, PredictionRequest
from src.core.config import INPUT_DATA, MIN_GAMES
from src.core.data import load_matchup_data
from src.core.scraper import fetch_live_matchup_data, build_complete_matchup_matrix
from src.core.telemetry import tracer
from src.tournament.monte_carlo import run_monte_carlo_analytics
from src.tournament.solver import predict_best_decks, get_variant_5_structure, swiss_rounds_from_players

q_logger = structlog.get_logger()
RedisInstrumentor().instrument()
redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/?db=0")
huey = RedisHuey('tcg_tasks', url=redis_url)


@huey.task()
def execute_simulation_job(payload: dict):
    """
    Background worker process with distributed tracing and structured logging.
    """
    with tracer.start_as_current_span("execute_simulation_job") as span:
        # 1. Instantiate and Validate Request (Tracing Pydantic overhead)
        with tracer.start_as_current_span("pydantic_validation"):
            request = PredictionRequest(**payload)
            job_id = request.job_id
            span.set_attribute("job.id", job_id)
            span.set_attribute("precision_tier", request.precision_tier)

        # Contextual logging
        log = q_logger.bind(job_id=job_id, task="simulation")
        log.info("starting_simulation_job", players=request.total_players)

        try:
            deck_names, win_matrix, _ = load_matchup_data(INPUT_DATA, MIN_GAMES)

            iterations = TIER_MAPPING.get(request.precision_tier, 25_000)
            players = request.total_players

            # 2. Run Solver (Water-filling & Baseline Constraints)
            with tracer.start_as_current_span("solver_prediction"):
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

            # 4. Run Monte Carlo Brackets (Tracing Rust Engine execution)
            with tracer.start_as_current_span("monte_carlo_analytics") as mc_span:
                mc_span.set_attribute("iterations", iterations)
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

            log.info("simulation_job_complete", status="success")
            return {
                "solver_results": solver_res,
                "mc_results": mc_res
            }
        except Exception as e:
            log.error("simulation_job_failed", error=str(e), exc_info=True)
            span.record_exception(e)
            span.set_status(trace.Status(trace.StatusCode.ERROR))
            raise


@huey.periodic_task(crontab(minute='0', hour='4'))
def automated_daily_pipeline():
    """
    Asynchronous periodic task to refresh metagame data with full observability.
    """
    # Start a root span for the daily ingestion process
    with tracer.start_as_current_span("automated_daily_pipeline") as span:
        log = q_logger.bind(task="daily_pipeline", schedule="02:00")
        log.info("starting_daily_scrape")  # Initialise the structured log entry

        from src.core.urls import POR_URLS
        target_urls = POR_URLS

        try:
            canonical_map = {}

            # Trace the HTTP overhead of fetching data from Limitless TCG
            with tracer.start_as_current_span("fetch_live_data"):
                raw_matchups = fetch_live_matchup_data(target_urls, canonical_map)

            if not raw_matchups:
                log.error("scraper_returned_no_data")
                raise ValueError("Scraper returned zero matchups. Limitless HTML structure may have changed.")

            # Trace the matrix reconstruction logic
            with tracer.start_as_current_span("build_matchup_matrix"):
                matrix_data = build_complete_matchup_matrix(raw_matchups)

            # Trace thermodynamic purity validation
            with tracer.start_as_current_span("pydantic_matrix_validation"):
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

            log.info("pipeline_successful", deck_count=len(matrix_data["archetypes"]))  # Log success with metadata

        except ValueError as ve:
            log.warn("data_validation_failed", reason=str(ve))  # Log warnings for non-critical integrity issues
            span.record_exception(ve)
            span.set_status(trace.Status(trace.StatusCode.ERROR))
        except Exception as e:
            log.error("critical_pipeline_failure", error=str(e), exc_info=True)  # Log critical errors with stack traces
            span.record_exception(e)
            span.set_status(trace.Status(trace.StatusCode.ERROR))
