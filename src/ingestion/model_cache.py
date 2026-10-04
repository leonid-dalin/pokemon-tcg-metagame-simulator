from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from src.core.config import BDIF_CARD_MAX_ABS_LOGIT, BDIF_CARD_PRIOR_SD
from src.ingestion.model import (
    CardModelNotIdentifiable,
    FittedCardModel,
    PlayerObservation,
    _logistic_standard_errors,
    _player_clustered_standard_errors,
    _card_design,
    _mean_presence,
    _card_covariates,
    fit_card_model,
    fit_h1_misty_variant,
    h1_observations,
    select_model_cards,
    Best60Request,
    recommend_best60,
)

MODEL_MAX_ITER = 10_000
_FIT_MAX_ITER = MODEL_MAX_ITER


def _cache_payload(model: FittedCardModel, database_sha256: str) -> dict[str, Any]:
    decks = model.decks
    cards = model.cards
    return {
        "database_sha256": database_sha256,
        "decks": decks,
        "cards": cards,
        "inclusion": {deck: dict(values) for deck, values in model.inclusion.items()},
        "standard_errors": dict(model.standard_errors),
        "match_counts": [[left, right, count] for (left, right), count in model.match_counts.items()],
        "reference_deck": model.reference_deck,
        "standard_error_kind": model.standard_error_kind,
        "members": {name: list(members) for name, members in model.members.items()},
        "not_identified": list(model.not_identified),
        "penalty": model.penalty,
        "max_iter": MODEL_MAX_ITER,
        "coef": model.estimator.coef_.tolist(),
        "intercept": model.estimator.intercept_.tolist(),
        "classes": model.estimator.classes_.tolist(),
    }


def _read_cache(path: Path, database_sha256: str) -> tuple[FittedCardModel, dict[str, Any]] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        cache_max_iter = payload.get("max_iter")
        if payload.get("database_sha256") != database_sha256 or not isinstance(cache_max_iter, int) or cache_max_iter < 1000:
            return None
        decks = list(payload["decks"])
        cards = list(payload["cards"])
        design_width = len(decks) - 1 + len(cards)
        estimator = LogisticRegression(
            penalty="l2",
            C=BDIF_CARD_PRIOR_SD ** 2,
            fit_intercept=False,
            max_iter=cache_max_iter,
            random_state=1312,
        )
        estimator.classes_ = np.asarray(payload["classes"])
        estimator.coef_ = np.asarray(payload["coef"], dtype=float).reshape(1, design_width)
        estimator.intercept_ = np.asarray(payload["intercept"], dtype=float)
        model = FittedCardModel(
            decks,
            cards,
            estimator,
            payload["inclusion"],
            payload["standard_errors"],
            {(left, right): int(count) for left, right, count in payload["match_counts"]},
            str(payload["reference_deck"]),
            str(payload["standard_error_kind"]),
            {str(name): tuple(map(str, members)) for name, members in payload["members"].items()},
            list(payload["not_identified"]),
            float(payload["penalty"]),
        )
        return model, payload
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _write_cache(path: Path, model: FittedCardModel, database_sha256: str) -> None:
    from src.core.files import write_json_atomic

    path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(_cache_payload(model, database_sha256), str(path))


@contextmanager
def _cache_lock(path: Path):
    lock_path = path.with_suffix(path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + 900
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"timed out waiting for model cache lock: {lock_path}")
            time.sleep(0.1)
        else:
            os.close(fd)
            try:
                yield
            finally:
                try:
                    lock_path.unlink()
                except FileNotFoundError:
                    pass
            return


def _fit(observations: Sequence[PlayerObservation], cards: Sequence[str]) -> FittedCardModel:
    try:
        return fit_card_model(observations, cards, max_iter=MODEL_MAX_ITER)
    except TypeError as exc:
        if "max_iter" not in str(exc):
            raise
        return fit_card_model(observations, cards)


def load_or_fit_card_model(
    observations: Sequence[PlayerObservation],
    database_sha256: str,
    cache_path: str | Path,
) -> tuple[FittedCardModel, bool]:
    path = Path(cache_path)
    cached = _read_cache(path, database_sha256)
    if cached is not None:
        return cached[0], True
    with _cache_lock(path):
        cached = _read_cache(path, database_sha256)
        if cached is not None:
            return cached[0], True
    from src.ingestion.model import select_model_cards as select_cards
    model = _fit(observations, select_cards(observations))
    _write_cache(path, model, database_sha256)
    return model, False


def build_model_addons(
    observations: Sequence[PlayerObservation],
    weights: dict[str, float],
    requested: Sequence[str],
    database_sha256: str,
    cache_path: str | Path,
    store,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    try:
        model, reused = load_or_fit_card_model(observations, database_sha256, cache_path)
    except CardModelNotIdentifiable as exc:
        return {deck: {"status": "not identifiable", "reason": str(exc)} for deck in requested}, {}, {
            "database_sha256": database_sha256,
            "model_status": "not identifiable",
            "model_cache_path": str(cache_path),
        }
    coefficients, intervals = model.coefficient_report()
    inclusion = model.inclusion
    candidates = sorted({card for cards in inclusion.values() for card in cards})
    recommendations = {}
    for deck in requested:
        if deck not in weights:
            recommendations[deck] = {"status": "unknown archetype"}
            continue
        core = store.observed_skeleton_details(deck)
        recommendation = recommend_best60(Best60Request(
            archetype=deck,
            candidates=candidates,
            coefficients=coefficients,
            coefficient_intervals=intervals,
            inclusion=inclusion,
            meta_weights=weights,
            playable_cards=store.observed_cards(deck),
            observed_pokemon_cards=store.observed_pokemon_cards(deck),
            card_rules=store.observed_card_rules(deck),
            skeleton=core["cards"],
        ))
        recommendation.update({key: core[key] for key in ("legal_list_count", "core_support", "core_share", "core_count")})
        recommendations[deck] = recommendation
    h1_rows = h1_observations(store.pairings_with_decklists("%alakazam%"))
    h1 = fit_h1_misty_variant(h1_rows) if h1_rows else {}
    provenance = {
        "database_sha256": database_sha256,
        "model_status": "complete",
        "model_cache_path": str(cache_path),
        "model_cache_reused": reused,
        "model_max_iter": MODEL_MAX_ITER,
        "model_decks": len(getattr(model, "decks", ())),
        "model_cards": len(getattr(model, "cards", ())),
        "model_packages": len(getattr(model, "packages", ())),
        "model_not_identified": getattr(model, "not_identified", []),
        "standard_error_kind": getattr(model, "standard_error_kind", "model-based"),
    }
    return recommendations, h1, provenance
