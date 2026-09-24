import json
import os

import requests
import structlog
from opentelemetry import trace
from opentelemetry.instrumentation.redis import RedisInstrumentor
from huey import RedisHuey, crontab


from src.api.models import ScrapedMatrix, TIER_MAPPING, PredictionRequest
from src.core.config import INPUT_DATA
from src.bdif.settings import BdifSettings
from src.bdif import service as bdif_service
from src.core.files import write_json_atomic
from src.ingestion.model import select_model_cards
from src.core.scraper import (
    build_complete_matchup_matrix,
    discover_live_matchup_urls,
    fetch_live_matchup_data,
)
from src.core.telemetry import tracer


q_logger = structlog.get_logger()

RedisInstrumentor().instrument()
redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/?db=0")
huey = RedisHuey('tcg_tasks', url=redis_url)



@huey.task()
def execute_simulation_job(payload: dict):
    """Validate a queued request and delegate simulation to the BDIF service."""
    with tracer.start_as_current_span("execute_simulation_job") as span:
        with tracer.start_as_current_span("pydantic_validation"):
            request = PredictionRequest(**payload)
            job_id = request.job_id
            span.set_attribute("job.id", job_id)
            span.set_attribute("precision_tier", request.precision_tier)

        # Contextual logging
        log = q_logger.bind(job_id=job_id, task="simulation")
        log.info("starting_simulation_job", players=request.total_players)

        try:
            def progress_callback(current_chunk: int, total_chunks: int):
                pct = int((current_chunk / total_chunks) * 100)
                message = json.dumps({"status": "processing", "progress": pct})
                pipe = huey.storage.conn.pipeline()
                pipe.setex(f"task:progress:{job_id}", 3600, message)
                pipe.publish(f"channel:progress:{job_id}", message)
                pipe.execute()

            with tracer.start_as_current_span("bdif_prediction"):
                result = bdif_service.run_prediction(
                    request, progress_callback, BdifSettings.from_environment(), log
                )
            log.info("simulation_job_complete", status="success")
            return result
        except Exception as exc:
            log.error("simulation_job_failed", error=str(exc), exc_info=True)
            span.record_exception(exc)
            span.set_status(trace.Status(trace.StatusCode.ERROR))
            raise


@huey.task()
def ingest_limitless_results():
    return bdif_service.run_ingestion(BdifSettings.from_environment())


@huey.periodic_task(crontab(minute='0', hour='*/2'))
def automated_daily_pipeline():
    """
    Periodic task to refresh metagame data with full observability.
    Runs every two hours; also triggered once on API startup under a lock.
    """
    # Start a root span for the daily ingestion process
    with tracer.start_as_current_span("automated_daily_pipeline") as span:
        log = q_logger.bind(task="daily_pipeline", schedule="0 */2 * * *")
        log.info("starting_daily_scrape")  # Initialise the structured log entry

        try:
            canonical_map = {}
            with requests.Session() as session:
                target_urls = discover_live_matchup_urls(session)
            log.info("discovered_pbl_matchup_urls", count=len(target_urls))

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

            write_json_atomic(final_json_structure, INPUT_DATA)

            log.info("pipeline_successful", deck_count=len(matrix_data["archetypes"]))  # Log success with metadata

        except ValueError as ve:
            log.warn("data_validation_failed", reason=str(ve))  # Log warnings for non-critical integrity issues
            span.record_exception(ve)
            span.set_status(trace.Status(trace.StatusCode.ERROR))
            raise
        except Exception as e:
            log.error("critical_pipeline_failure", error=str(e), exc_info=True)  # Log critical errors with stack traces
            span.record_exception(e)
            span.set_status(trace.Status(trace.StatusCode.ERROR))
            raise