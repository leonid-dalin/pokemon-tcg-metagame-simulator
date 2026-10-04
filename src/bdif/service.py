from pathlib import Path
import os
from typing import Sequence

import numpy as np

from src.api.models import PredictionRequest, TIER_MAPPING
from src.bdif.settings import BdifSettings, simulation_input_path
from src.core import config
from src.core.logger import logger
from src.core.data import load_matchup_data
from src.core.scraper import normalize_archetype
from src.ingestion.model import PlayerObservation, select_panel_decks
from src.tournament.monte_carlo import run_monte_carlo_analytics
from src.tournament.reporting import build_bdif_report
from src.tournament.solver import predict_best_decks, get_variant_5_structure, swiss_rounds_from_players

_MODEL_CACHE = {}


def _file_sha256(path: str) -> str | None:
    if not os.path.isfile(path):
        return None
    import hashlib
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
def _has_complete_observations(observations: Sequence[PlayerObservation]) -> bool:
    return len(observations) >= 4 and len({row.result for row in observations}) == 2


def map_panel_decks_to_matrix(panel_decks: list[str], deck_names: list[str]) -> list[str]:
    by_name: dict[str, list[str]] = {}
    for name in deck_names:
        by_name.setdefault(normalize_archetype(name), []).append(name)
    mapped, dropped = [], []
    for deck in panel_decks:
        tokens = normalize_archetype(deck.replace("-", " ")).split()
        match = None
        for end in range(len(tokens), 0, -1):
            candidates = by_name.get(" ".join(tokens[:end]), [])
            if len(candidates) == 1:
                match = candidates[0]
                break
            if len(candidates) > 1:
                break
        if match:
            mapped.append(match)
        else:
            dropped.append(deck)
    if dropped:
        logger.warning("bdif_panel_decks_dropped", deck_ids=dropped)
    return mapped


def open_store(settings: BdifSettings | None = None):
    settings = settings or BdifSettings.from_environment()
    if not os.path.exists(settings.db_path):
        return None
    from src.ingestion.store import LimitlessStore
    return LimitlessStore(settings.db_path)


def panel_decks_for_report(deck_names: list[str] | None = None, store=None, settings: BdifSettings | None = None) -> list[str]:
    settings = settings or BdifSettings.from_environment()
    if not settings.use_card_model or not os.path.exists(settings.db_path):
        return list(settings.fallback_panel_decks)
    from src.ingestion.model import select_panel_decks
    store = store or open_store(settings)
    selected = select_panel_decks(
        store.deck_weights(),
        threshold=settings.panel_share_threshold,
        max_decks=settings.panel_max_decks,
    )
    return map_panel_decks_to_matrix(selected, deck_names or [])


def _build_report_addons(
    store=None,
    settings: BdifSettings | None = None,
    requested_archetypes: Sequence[str] | None = None,
) -> tuple[dict, dict, dict]:
    settings = settings or BdifSettings.from_environment()
    if not settings.use_card_model or not os.path.exists(settings.db_path):
        return {}, {}, {"model_status": "disabled"}
    from src.ingestion.model_cache import build_model_addons
    store = store or open_store(settings)
    store.prepare_for_read()
    database_sha256 = _file_sha256(settings.db_path)
    if database_sha256 is None:
        return {deck: {"status": "missing"} for deck in requested}, {}, {"model_status": "missing"}
    if settings.model_cache_path:
        cache_path = Path(settings.model_cache_path)
    else:
        cache_path = Path(settings.db_path).with_suffix(".model.json")
    weights = store.deck_weights()
    requested = tuple(requested_archetypes or ())
    if not weights:
        return {}, {}, {"database_sha256": database_sha256, "model_status": "insufficient observations"}
    top = [deck for deck in requested if deck in weights] if requested else select_panel_decks(weights, threshold=settings.panel_share_threshold)
    unknown = {deck: {"status": "unknown archetype"} for deck in requested if deck not in weights}
    if not top:
        return unknown, {}, {"database_sha256": database_sha256}
    observations = store.player_observations()
    if not _has_complete_observations(observations):
        return {**unknown, **{deck: {"status": "insufficient stored observations"} for deck in top}}, {}, {
            "database_sha256": database_sha256,
            "model_status": "insufficient observations",
        }
    import src.ingestion.model_cache as model_cache
    recommendations, h1, model_provenance = model_cache.build_model_addons(
        observations,
        weights,
        top,
        database_sha256,
        cache_path,
        store,
    )
    if not isinstance(recommendations, dict):
        recommendations = {deck: {"status": "complete"} for deck in top}
    return {**unknown, **recommendations}, h1, model_provenance


def build_report_addons(
    store=None,
    settings: BdifSettings | None = None,
    requested_archetypes: Sequence[str] | None = None,
) -> tuple[dict, dict]:
    recommendations, h1, _ = _build_report_addons(store, settings, requested_archetypes)
    return recommendations, h1


def request_bdif_report(
    archetype: str,
    additional_archetypes: Sequence[str] | None = None,
    settings: BdifSettings | None = None,
    store=None,
) -> dict:
    settings = settings or BdifSettings.from_environment()
    requested = [archetype, *(additional_archetypes or ())]
    if not settings.use_card_model:
        return {
            "status": "unavailable",
            "best60_recommendations": {deck: {"status": "disabled"} for deck in requested},
            "provenance": {"model_status": "disabled"},
        }
    recommendations, h1, provenance = _build_report_addons(store, settings, requested)
    statuses = [recommendation.get("status", "complete") for recommendation in recommendations.values()]
    status = "complete" if statuses and all(value == "complete" for value in statuses) else "partial"
    return {
        "status": status,
        "best60_recommendations": recommendations,
        "h1_report": h1,
        "provenance": provenance,
    }


def refit_card_model(settings: BdifSettings | None = None) -> dict:
    settings = settings or BdifSettings.from_environment()
    if not os.path.exists(settings.db_path):
        return {"status": "missing", "db_path": settings.db_path}
    from src.ingestion.model import CardModelNotIdentifiable, model_artifact
    from src.ingestion.store import LimitlessStore
    from src.ingestion.model_cache import load_or_fit_card_model
    store = LimitlessStore(settings.db_path)
    observations = store.player_observations()
    if not _has_complete_observations(observations):
        return {"status": "insufficient observations"}
    database_sha256 = _file_sha256(settings.db_path)
    try:
        fitted, reused = load_or_fit_card_model(observations, database_sha256, settings.model_cache_path)
    except CardModelNotIdentifiable:
        return {"status": "not identifiable"}
    os.makedirs(os.path.dirname(settings.model_input_path), exist_ok=True)
    from src.core.files import write_json_atomic
    write_json_atomic(model_artifact(fitted), settings.model_input_path)
    return {
        "status": "complete",
        "path": settings.model_input_path,
        "model_cache_path": settings.model_cache_path,
        "database_sha256": database_sha256,
        "model_cache_reused": reused,
        "model_max_iter": 10_000,
        "card_packages": list(fitted.packages),
        "not_identified": fitted.not_identified,
    }


def run_ingestion(settings: BdifSettings | None = None, limit: int | None = None) -> dict:
    settings = settings or BdifSettings.from_environment()
    if not settings.ingestion_enabled:
        return {"status": "disabled"}
    from src.ingestion.aggregate import build_artifact
    from src.ingestion.client import LimitlessClient
    from src.ingestion.model import CardModelNotIdentifiable, fit_card_model, model_artifact, select_model_cards
    from src.ingestion.store import LimitlessStore
    from src.core.files import write_json_atomic
    client = LimitlessClient.from_environment()
    store = LimitlessStore(settings.db_path)
    store.ensure_schema()
    names = {str(row.get("identifier") or row.get("id")): str(row["name"]) for row in client.game_decks() if row.get("name") and (row.get("identifier") or row.get("id"))}
    store.backfill_deck_names(names)
    events = list(client.iter_tournaments(game="PTCG", format="STANDARD", limit=limit if limit is not None else settings.backfill_limit))
    existing_ids = store.existing_tournament_ids()
    failed = []
    skipped_events = 0
    for event in events:
        event_id = str(event["id"])
        if event_id in existing_ids:
            skipped_events += 1
            continue
        try:
            details, standings, pairings = client.fetch_event_bundle(event_id)
            store.upsert_tournament(event, details)
            store.upsert_standings(event_id, standings, names) if names else store.upsert_standings(event_id, standings)
            store.upsert_pairings(event_id, pairings)
            existing_ids.add(event_id)
        except Exception as exc:
            failed.append({"id": event_id, "error": str(exc)})
            logger.warning("limitless_event_failed", event_id=event_id, error=str(exc))
    artifact_path = settings.ingestion_input_path
    os.makedirs(os.path.dirname(artifact_path), exist_ok=True)
    write_json_atomic(build_artifact(store), artifact_path)
    unmapped_deck_ids = store.unmapped_deck_ids()
    if unmapped_deck_ids:
        logger.warning("ingest_unmapped_archetypes", deck_ids=unmapped_deck_ids)
    observations = store.player_observations()
    status = "insufficient observations"
    if _has_complete_observations(observations):
        try:
            fitted = fit_card_model(observations, select_model_cards(observations))
        except CardModelNotIdentifiable:
            status = "not identifiable"
        else:
            write_json_atomic(model_artifact(fitted), settings.model_input_path)
            status = "complete"
    if events and len(failed) == len(events):
        outcome = "failed"
    elif failed:
        outcome = "partial"
    else:
        outcome = "complete"
    return {"status": outcome, "events": len(events), "skipped_events": skipped_events, "failed_events": failed, "unmapped_deck_ids": unmapped_deck_ids, "path": artifact_path, "model_status": status}


def bdif_status(settings: BdifSettings | None = None) -> dict:
    settings = settings or BdifSettings.from_environment()
    base = {
        "db_path": settings.db_path,
        "use_card_model": settings.use_card_model,
        "ingestion_enabled": settings.ingestion_enabled,
        "model_artifact": os.path.exists(settings.model_input_path),
    }
    if not os.path.exists(settings.db_path):
        return {"status": "missing", **base}
    from src.ingestion.store import LimitlessStore
    return {"status": "available", **base, **LimitlessStore(settings.db_path).summary()}


def run_prediction(request: PredictionRequest, progress_callback=None, settings: BdifSettings | None = None, logger=None, seed: int | None = None, input_path: str | None = None) -> dict:
    settings = settings or BdifSettings.from_environment()
    logger = logger or __import__("structlog").get_logger()
    source = input_path or simulation_input_path(settings)
    input_sha256 = _file_sha256(source)
    deck_names, matrix, details = load_matchup_data(source, config.MIN_GAMES)
    requested = np.asarray(request.matchup_matrix, dtype=float)
    if list(request.deck_names) != list(deck_names) or requested.shape != matrix.shape or not np.allclose(requested, matrix):
        raise ValueError(f"request matrix does not match the simulation input {source}")
    solver = predict_best_decks(request)
    players = request.total_players
    if request.tournament_style == "championship_series":
        d1, cut, d2, top_cut = get_variant_5_structure(players)
    else:
        d1, cut, d2, top_cut = swiss_rounds_from_players(players), 99, 0, (8 if players >= 8 else 0)
    store = open_store(settings) if settings.use_card_model else None
    if request.bdif_panel_decks:
        panels = list(request.bdif_panel_decks)
    else:
        try:
            panels = panel_decks_for_report(deck_names, store, settings)
        except Exception as exc:
            logger.warning("bdif_panel_selection_failed", error=str(exc), exc_info=True)
            panels = list(settings.fallback_panel_decks)
    try:
        recommendations, h1 = build_report_addons(store, settings)
    except Exception as exc:
        logger.warning("bdif_report_addons_failed", error=str(exc), exc_info=True)
        recommendations, h1 = {}, {}
    run_seed = config.RNG_SEED if seed is None else seed
    iterations = TIER_MAPPING.get(request.precision_tier, 25_000)
    result = run_monte_carlo_analytics(
        deck_names=deck_names, win_matrix=matrix, meta_distribution=solver["full_meta"],
        d1_rounds=d1, cut_points=cut, d2_rounds=d2, top_cut=top_cut, players=players,
        iterations=iterations, match_format=request.match_format,
        use_tie_convergence=request.use_tie_convergence, global_tie_rate=request.global_tie_rate,
        use_drop_feature=request.use_drop_feature, seed=run_seed,
        progress_callback=progress_callback, matchup_details=details, panel_decks=panels,
    )
    provenance = {
        "input_path": source,
        "input_sha256": input_sha256,
        "seed": run_seed,
        "iterations": iterations,
        "posterior_draws": (result.get("posterior") or {}).get("draws"),
        "use_card_model": settings.use_card_model,
        "panel_decks": list(panels),
    }
    return {"solver_results": solver, "mc_results": build_bdif_report(result, recommendations, h1, provenance)}
