import json
import os
from copy import deepcopy
import requests
import structlog
from opentelemetry import trace
from opentelemetry.instrumentation.redis import RedisInstrumentor
from huey import RedisHuey, crontab
from typing import Sequence

from src.api.models import ScrapedMatrix, TIER_MAPPING, PredictionRequest
from src.core.config import (
    BDIF_PANEL_DECKS,
    BDIF_PANEL_SHARE_THRESHOLD,
    BDIF_USE_CARD_MODEL,
    INPUT_DATA,
    LIMITLESS_BACKFILL_TOURNAMENTS,
    LIMITLESS_INGESTION_ENABLED,
    MIN_GAMES,
    RNG_SEED,
)
from src.ingestion.model import PlayerObservation, select_model_cards
from src.core.data import load_matchup_data
from src.core.scraper import (
    build_complete_matchup_matrix,
    discover_live_matchup_urls,
    fetch_live_matchup_data,
    normalize_archetype,
)
from src.core.telemetry import tracer
from src.tournament.monte_carlo import run_monte_carlo_analytics
from src.tournament.reporting import build_bdif_report
from src.tournament.solver import predict_best_decks, get_variant_5_structure, swiss_rounds_from_players

q_logger = structlog.get_logger()
_BDIF_MODEL_CACHE: dict[tuple[str, float], tuple[dict, dict]] = {}
RedisInstrumentor().instrument()
redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/?db=0")
huey = RedisHuey('tcg_tasks', url=redis_url)


def _write_json_atomic(payload: dict, path: str) -> None:
    temp_path = f"{path}.tmp"
    target_mode = os.stat(path).st_mode & 0o777 if os.path.exists(path) else None
    try:
        with open(temp_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        if target_mode is not None:
            os.chmod(temp_path, target_mode)
        os.replace(temp_path, path)
    except Exception:
        try:
            os.remove(temp_path)
        except FileNotFoundError:
            pass
        raise


def _has_complete_observations(observations: Sequence[PlayerObservation]) -> bool:
    return len(observations) >= 4 and len({row.result for row in observations}) == 2


def _map_panel_decks_to_matrix(panel_decks: list[str], deck_names: list[str]) -> list[str]:
    by_normalised_name: dict[str, list[str]] = {}
    for name in deck_names:
        by_normalised_name.setdefault(normalize_archetype(name), []).append(name)
    mapped = []
    dropped = []
    for deck in panel_decks:
        tokens = normalize_archetype(deck.replace("-", " ")).split()
        matrix_name = None
        for end in range(len(tokens), 0, -1):
            matches = by_normalised_name.get(" ".join(tokens[:end]), [])
            if len(matches) == 1:
                matrix_name = matches[0]
                break
            if len(matches) > 1:
                break
        if matrix_name:
            mapped.append(matrix_name)
        else:
            dropped.append(deck)
    if dropped:
        q_logger.warning("bdif_panel_decks_dropped", deck_ids=dropped)
    return mapped


def _panel_decks_for_report(deck_names: list[str] | None = None, store=None) -> list[str]:
    if not BDIF_USE_CARD_MODEL:
        return list(BDIF_PANEL_DECKS)
    db_path = os.path.join("data", "limitless.db")
    if not os.path.exists(db_path):
        return list(BDIF_PANEL_DECKS)
    from src.ingestion.model import select_panel_decks
    if store is None:
        from src.ingestion.store import LimitlessStore
        store = LimitlessStore(db_path)
    selected = select_panel_decks(
        store.deck_weights(),
        threshold=BDIF_PANEL_SHARE_THRESHOLD,
    )
    return _map_panel_decks_to_matrix(selected, deck_names or [])


def _build_bdif_report_addons(store=None) -> tuple[dict, dict]:
    if not BDIF_USE_CARD_MODEL:
        return {}, {}

    from src.ingestion.model import Best60Request, CardModelNotIdentifiable, fit_card_model, select_model_cards, fit_h1_misty_variant, h1_observations, recommend_best60, select_panel_decks
    from src.ingestion.store import LimitlessStore

    db_path = os.path.join("data", "limitless.db")
    if not os.path.exists(db_path):
        return {}, {}
    if store is None:
        store = LimitlessStore(db_path)
    store.prepare_for_read()
    cache_key = (os.path.abspath(db_path), float(os.path.getmtime(db_path)))
    if cache_key in _BDIF_MODEL_CACHE:
        recommendations, h1 = _BDIF_MODEL_CACHE[cache_key]
        return deepcopy(recommendations), deepcopy(h1)
    _BDIF_MODEL_CACHE.clear()
    deck_weights = store.deck_weights()
    if not deck_weights:
        return {}, {}
    top_decks = select_panel_decks(deck_weights, threshold=BDIF_PANEL_SHARE_THRESHOLD)
    observations = store.player_observations()
    if not _has_complete_observations(observations):
        return {deck: {"status": "insufficient stored observations"} for deck in top_decks}, {}
    try:
        fitted = fit_card_model(observations, select_model_cards(observations))
    except CardModelNotIdentifiable as exc:
        return {deck: {"status": "not identifiable", "reason": str(exc)} for deck in top_decks}, {}
    inclusion = fitted.inclusion
    coefficients, intervals = fitted.coefficient_report()
    candidates = sorted({card for values in inclusion.values() for card in values})
    recommendations = {}
    for deck in top_decks:
        recommendations[deck] = recommend_best60(Best60Request(
            archetype=deck,
            candidates=candidates,
            coefficients=coefficients,
            coefficient_intervals=intervals,
            inclusion=inclusion,
            meta_weights=deck_weights,
            playable_cards=store.observed_cards(deck),
            skeleton=store.observed_skeleton(deck),
        ))
    h1_rows = store.pairings_with_decklists("%alakazam%")
    h1_data = h1_observations(h1_rows)
    h1 = fit_h1_misty_variant(h1_data) if h1_data else {}
    result = (recommendations, h1)
    _BDIF_MODEL_CACHE[cache_key] = deepcopy(result)
    return deepcopy(result[0]), deepcopy(result[1])


def _simulation_input_path() -> str:
    if BDIF_USE_CARD_MODEL:
        candidate = os.path.join("data", "input", "limitless_model_input.json")
        if os.path.exists(candidate):
            return candidate
        q_logger.warning("card_model_artifact_missing", path=candidate)
    return INPUT_DATA


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
            deck_names, win_matrix, matchup_details = load_matchup_data(_simulation_input_path(), MIN_GAMES)

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

                msg_payload = json.dumps({"status": "processing", "progress": pct})

                pipe = huey.storage.conn.pipeline()
                pipe.setex(f"task:progress:{job_id}", 3600, msg_payload)
                pipe.publish(f"channel:progress:{job_id}", msg_payload)
                pipe.execute()

            bdif_store = None
            if BDIF_USE_CARD_MODEL and os.path.exists(os.path.join("data", "limitless.db")):
                from src.ingestion.store import LimitlessStore
                bdif_store = LimitlessStore(os.path.join("data", "limitless.db"))

            try:
                panel_decks = _panel_decks_for_report(deck_names, bdif_store)
            except Exception as exc:
                log.warning("bdif_panel_selection_failed", error=str(exc), exc_info=True)
                panel_decks = list(BDIF_PANEL_DECKS)

            try:
                best60_recommendations, h1_report = _build_bdif_report_addons(bdif_store)
            except Exception as exc:
                log.warning("bdif_report_addons_failed", error=str(exc), exc_info=True)
                best60_recommendations, h1_report = {}, {}

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
                    seed=RNG_SEED,
                    progress_callback=_progress_handler,
                    matchup_details=matchup_details,
                    panel_decks=panel_decks,
                )

            log.info("simulation_job_complete", status="success")
            return {
                "solver_results": solver_res,
                "mc_results": build_bdif_report(mc_res, best60_recommendations, h1_report)
            }
        except Exception as e:
            log.error("simulation_job_failed", error=str(e), exc_info=True)
            span.record_exception(e)
            span.set_status(trace.Status(trace.StatusCode.ERROR))
            raise


@huey.task()
def ingest_limitless_results():
    if not LIMITLESS_INGESTION_ENABLED:
        return {"status": "disabled"}

    from src.ingestion.aggregate import build_artifact
    from src.ingestion.client import LimitlessClient
    from src.ingestion.model import CardModelNotIdentifiable, fit_card_model, model_artifact
    from src.ingestion.store import LimitlessStore

    client = LimitlessClient.from_environment()
    store = LimitlessStore(os.path.join("data", "limitless.db"))
    store.ensure_schema()
    deck_names = {
        str(deck.get("identifier") or deck.get("id")): str(deck["name"])
        for deck in client.game_decks()
        if deck.get("name") and (deck.get("identifier") or deck.get("id"))
    }
    store.backfill_deck_names(deck_names)
    events = client.tournaments(
        game="PTCG",
        format="STANDARD",
        limit=LIMITLESS_BACKFILL_TOURNAMENTS,
    )
    failed_events = []
    for event in events:
        event_id = str(event["id"])
        try:
            details, standings, pairings = client.fetch_event_bundle(event_id)
            store.upsert_tournament(event, details)
            if deck_names:
                store.upsert_standings(event_id, standings, deck_names)
            else:
                store.upsert_standings(event_id, standings)
            store.upsert_pairings(event_id, pairings)
        except Exception as exc:
            failed_events.append({"id": event_id, "error": str(exc)})
            q_logger.warning("limitless_event_failed", event_id=event_id, error=str(exc))

    artifact = build_artifact(store)
    artifact_path = os.path.join("data", "input", "limitless_input.json")
    os.makedirs(os.path.dirname(artifact_path), exist_ok=True)
    _write_json_atomic(artifact, artifact_path)
    observations = store.player_observations()
    model_path = os.path.join("data", "input", "limitless_model_input.json")
    model_status = "insufficient observations"
    if _has_complete_observations(observations):
        try:
            fitted = fit_card_model(observations, select_model_cards(observations))
        except CardModelNotIdentifiable:
            model_status = "not identifiable"
        else:
            _write_json_atomic(model_artifact(fitted), model_path)
            model_status = "complete"
    return {"status": "complete", "events": len(events), "failed_events": failed_events, "path": artifact_path, "model_status": model_status}


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

            _write_json_atomic(final_json_structure, INPUT_DATA)

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