import json
import os
from copy import deepcopy
from typing import Sequence

from src.api.models import PredictionRequest, TIER_MAPPING
from src.bdif.settings import BdifSettings
from src.core import config
from src.core.logger import logger
from src.core.data import load_matchup_data
from src.core.scraper import normalize_archetype
from src.ingestion.model import PlayerObservation
from src.tournament.monte_carlo import run_monte_carlo_analytics
from src.tournament.reporting import build_bdif_report
from src.tournament.solver import predict_best_decks, get_variant_5_structure, swiss_rounds_from_players

_MODEL_CACHE: dict[tuple[str, float], tuple[dict, dict]] = {}


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


def build_report_addons(store=None, settings: BdifSettings | None = None) -> tuple[dict, dict]:
    settings = settings or BdifSettings.from_environment()
    if not settings.use_card_model or not os.path.exists(settings.db_path):
        return {}, {}
    from src.ingestion.model import Best60Request, CardModelNotIdentifiable, fit_card_model, select_model_cards, fit_h1_misty_variant, h1_observations, recommend_best60, select_panel_decks
    store = store or open_store(settings)
    store.prepare_for_read()
    key = (os.path.abspath(settings.db_path), float(os.path.getmtime(settings.db_path)))
    if key in _MODEL_CACHE:
        result = _MODEL_CACHE[key]
        return deepcopy(result[0]), deepcopy(result[1])
    _MODEL_CACHE.clear()
    weights = store.deck_weights()
    if not weights:
        return {}, {}
    top = select_panel_decks(weights, threshold=settings.panel_share_threshold)
    observations = store.player_observations()
    if not _has_complete_observations(observations):
        return {deck: {"status": "insufficient stored observations"} for deck in top}, {}
    try:
        fitted = fit_card_model(observations, select_model_cards(observations))
    except CardModelNotIdentifiable as exc:
        return {deck: {"status": "not identifiable", "reason": str(exc)} for deck in top}, {}
    coefficients, intervals = fitted.coefficient_report()
    inclusion = fitted.inclusion
    candidates = sorted({card for cards in inclusion.values() for card in cards})
    recommendations = {
        deck: recommend_best60(Best60Request(
            archetype=deck, candidates=candidates, coefficients=coefficients,
            coefficient_intervals=intervals, inclusion=inclusion, meta_weights=weights,
            playable_cards=store.observed_cards(deck), skeleton=store.observed_skeleton(deck),
        )) for deck in top
    }
    rows = store.pairings_with_decklists("%alakazam%")
    h1_data = h1_observations(rows)
    result = (recommendations, fit_h1_misty_variant(h1_data) if h1_data else {})
    _MODEL_CACHE[key] = deepcopy(result)
    return deepcopy(result[0]), deepcopy(result[1])


def simulation_input_path(settings: BdifSettings | None = None) -> str:
    settings = settings or BdifSettings.from_environment()
    if settings.use_card_model and os.path.exists(settings.model_input_path):
        return settings.model_input_path
    if settings.use_card_model:
        logger.warning("card_model_artifact_missing", path=settings.model_input_path)
    return settings.baseline_input_path


def refit_card_model(settings: BdifSettings | None = None) -> dict:
    settings = settings or BdifSettings.from_environment()
    from src.ingestion.model import CardModelNotIdentifiable, fit_card_model, model_artifact, select_model_cards
    from src.ingestion.store import LimitlessStore
    store = LimitlessStore(settings.db_path)
    observations = store.player_observations()
    if not _has_complete_observations(observations):
        return {"status": "insufficient observations"}
    try:
        fitted = fit_card_model(observations, select_model_cards(observations))
    except CardModelNotIdentifiable:
        return {"status": "not identifiable"}
    os.makedirs(os.path.dirname(settings.model_input_path), exist_ok=True)
    from src.core.files import write_json_atomic
    write_json_atomic(model_artifact(fitted), settings.model_input_path)
    return {"status": "complete", "path": settings.model_input_path}


def run_ingestion(settings: BdifSettings | None = None) -> dict:
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
    events = client.tournaments(game="PTCG", format="STANDARD", limit=settings.backfill_limit)
    failed = []
    for event in events:
        event_id = str(event["id"])
        try:
            details, standings, pairings = client.fetch_event_bundle(event_id)
            store.upsert_tournament(event, details)
            store.upsert_standings(event_id, standings, names) if names else store.upsert_standings(event_id, standings)
            store.upsert_pairings(event_id, pairings)
        except Exception as exc:
            failed.append({"id": event_id, "error": str(exc)})
            logger.warning("limitless_event_failed", event_id=event_id, error=str(exc))
    artifact_path = settings.ingestion_input_path
    os.makedirs(os.path.dirname(artifact_path), exist_ok=True)
    write_json_atomic(build_artifact(store), artifact_path)
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
    return {"status": "complete", "events": len(events), "failed_events": failed, "path": artifact_path, "model_status": status}


def bdif_status(settings: BdifSettings | None = None) -> dict:
    settings = settings or BdifSettings.from_environment()
    if not os.path.exists(settings.db_path):
        return {"status": "missing", "db_path": settings.db_path}
    store = open_store(settings)
    store.prepare_for_read()
    return {"status": "available", "db_path": settings.db_path, "decks": len(store.deck_weights()), "observations": len(store.player_observations())}


def run_prediction(request: PredictionRequest, progress_callback=None, settings: BdifSettings | None = None, logger=None, seed: int | None = None) -> dict:
    settings = settings or BdifSettings.from_environment()
    logger = logger or __import__("structlog").get_logger()
    deck_names, matrix, details = load_matchup_data(simulation_input_path(settings), config.MIN_GAMES)
    solver = predict_best_decks(request)
    players = request.total_players
    if request.tournament_style == "championship_series":
        d1, cut, d2, top_cut = get_variant_5_structure(players)
    else:
        d1, cut, d2, top_cut = swiss_rounds_from_players(players), 99, 0, (8 if players >= 8 else 0)
    store = open_store(settings) if settings.use_card_model else None
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
    result = run_monte_carlo_analytics(
        deck_names=deck_names, win_matrix=matrix, meta_distribution=solver["full_meta"],
        d1_rounds=d1, cut_points=cut, d2_rounds=d2, top_cut=top_cut, players=players,
        iterations=TIER_MAPPING.get(request.precision_tier, 25_000), match_format=request.match_format,
        use_tie_convergence=request.use_tie_convergence, global_tie_rate=request.global_tie_rate,
        use_drop_feature=request.use_drop_feature, seed=config.RNG_SEED if seed is None else seed,
        progress_callback=progress_callback, matchup_details=details, panel_decks=panels,
    )
    return {"solver_results": solver, "mc_results": build_bdif_report(result, recommendations, h1)}
