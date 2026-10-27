"""Chronological model selection, exact-list recommendations and held-out evaluation."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import time
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import sparse
from scipy.special import expit

from src.ingestion.best60 import ArchetypeList, consensus_sixty
from src.ingestion.best60_research.evaluation import (
    balanced_h1_summary,
    decide_h1,
    player_aware_interval,
    policy_value,
    ranking_agreement,
)
from src.ingestion.best60_research.estimators import FitBudgetExceeded, FittedModel, fit_model
from src.ingestion.best60_research.selection import field_scores, select_catalogue
from src.ingestion.best60_research.uncertainty import RefitFailure, fixed_list_interval
from src.ingestion.model import validate_recommendation


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"cannot serialise {type(value).__name__}")


def _atomic_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, default=_json_default, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(payload + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _remaining(deadline: float | None) -> float | None:
    return None if deadline is None else max(0.001, deadline - time.monotonic())


def _date_value(value: Any) -> date | None:
    if value is None or not str(value).strip():
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        try:
            return date.fromisoformat(str(value)[:10])
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).date()


def _event_block(context: Mapping[str, Any], row: Mapping[str, Any]) -> str:
    event = str(row.get("event") or "unknown")
    configured = context.get("event_block_ids", {}).get(event)
    if configured is not None:
        return str(configured)
    day = _date_value(row.get("date"))
    return f"utc-date:{day.isoformat()}" if day is not None else event


def _object_array(values: Sequence[Any]) -> np.ndarray:
    array = np.empty(len(values), dtype=object)
    array[:] = list(values)
    return array


def _category_value(row: Mapping[str, Any], field: str) -> str:
    if field == "month":
        parsed = _date_value(row.get("date"))
        return parsed.strftime("%Y-%m") if parsed else "unknown"
    if field == "is_online":
        value = row.get(field)
        if value is None:
            return "unknown"
        return "true" if bool(value) else "false"
    if field == "matchup":
        return json.dumps(
            [str(row.get("archetype_id") or "unknown"), str(row.get("opponent_archetype_id") or "unknown")],
            separators=(",", ":"),
        )
    return str(row.get(field) or "unknown")


_CATEGORIES = (
    "archetype_id",
    "opponent_archetype_id",
    "matchup",
    "source",
    "platform",
    "delivery_mode",
    "match_mode",
    "format",
    "game",
    "is_online",
    "month",
)


def _design(
    rows: Sequence[Mapping[str, Any]],
    context: Mapping[str, Any],
    *,
    training: bool,
    schema: Mapping[str, Any] | None = None,
) -> tuple[Any, np.ndarray, Any, np.ndarray, Any]:
    """Encode copy slots and frozen categorical controls as sparse matrices."""
    slot_names = tuple(tuple(slot) for slot in context["slot_names"])
    slot_ids = {slot: index for index, slot in enumerate(slot_names)}
    row_ids: list[int] = []
    col_ids: list[int] = []
    for index, row in enumerate(rows):
        for card, count in row.get("counts", {}).items():
            for copy_number in range(1, int(count) + 1):
                column = slot_ids.get((str(card), copy_number))
                if column is not None:
                    row_ids.append(index)
                    col_ids.append(column)
    X = sparse.csr_matrix(
        (np.ones(len(row_ids), dtype=float), (row_ids, col_ids)), shape=(len(rows), len(slot_names))
    )
    if schema is None:
        vocab = {field: sorted({_category_value(row, field) for row in rows} | {"unknown"}) for field in _CATEGORIES}
        schema = {"categories": _CATEGORIES, "vocabulary": vocab}
    vocabulary = schema["vocabulary"]
    unknown_counts = {field: 0 for field in _CATEGORIES}
    column_specs: list[tuple[str, str]] = [("base", "intercept"), ("base", "skill_gap")]
    for field in _CATEGORIES:
        levels = vocabulary[field]
        for level in levels[1:]:
            column_specs.append((field, level))
    column_by_spec = {spec: index for index, spec in enumerate(column_specs)}
    nuisance_rows: list[int] = []
    nuisance_cols: list[int] = []
    nuisance_values: list[float] = []
    y = np.asarray([float(row.get("won", 0.0)) for row in rows], dtype=float)
    groups = _object_array([str(row.get("archetype_id") or "unknown") for row in rows])
    strength_map = context.get("strength", {}) if training else context.get("strength_at_cutoff", {})
    for index, row in enumerate(rows):
        player, opponent, event = row.get("player"), row.get("opponent"), row.get("event")
        if training:
            own_strength = float(strength_map.get((event, player), 0.0))
            opponent_strength = float(strength_map.get((event, opponent), 0.0))
        else:
            own_strength = float(strength_map.get(player, 0.0))
            opponent_strength = float(strength_map.get(opponent, 0.0))
        nuisance_rows.extend((index, index))
        nuisance_cols.extend((column_by_spec[("base", "intercept")], column_by_spec[("base", "skill_gap")]))
        nuisance_values.extend((1.0, own_strength - opponent_strength))
        for field in _CATEGORIES:
            value = _category_value(row, field)
            if value not in vocabulary[field]:
                unknown_counts[field] += 1
                value = "unknown"
            column = column_by_spec.get((field, value))
            if column is not None:
                nuisance_rows.append(index)
                nuisance_cols.append(column)
                nuisance_values.append(1.0)
    nuisance = sparse.csr_matrix(
        (nuisance_values, (nuisance_rows, nuisance_cols)), shape=(len(rows), len(column_specs))
    )
    if training:
        return X, y, nuisance, groups, dict(schema)
    return X, y, nuisance, groups, {"unknown_categories": unknown_counts, "nuisance_columns": column_specs}


def _inner_validation_context(
    context: Mapping[str, Any],
    validation_rows: Sequence[Mapping[str, Any]],
    *,
    embargo_days: int,
) -> tuple[dict[str, Any], str | None]:
    """Freeze validation skill controls before the first date in an inner fold."""
    validation_dates = [_date_value(row.get("date")) for row in validation_rows]
    first_validation_date = min((value for value in validation_dates if value is not None), default=None)
    inner = dict(context)
    if first_validation_date is None:
        inner["strength_at_cutoff"] = {}
        return inner, None

    boundary = first_validation_date - timedelta(days=embargo_days)
    event_dates = {
        str(event): _date_value(value)
        for event, value in context.get("training_event_dates", {}).items()
    }
    for event, block in context.get("event_block_ids", {}).items():
        event_dates.setdefault(str(event), _date_value(str(block).rsplit(":", 1)[-1]))

    totals: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for record in context.get("eligible_records", ()):
        event = str(record.get("event") or "")
        record_date = event_dates.get(event)
        if record_date is None or record_date >= boundary:
            continue
        player = str(record.get("player") or "")
        if not player:
            continue
        totals[player][0] += int(record.get("wins") or 0)
        totals[player][1] += int(record.get("losses") or 0)

    from src.ingestion.best60_research.context import STRENGTH_PRIOR_GAMES

    prior = float(context.get("strength_prior_games", STRENGTH_PRIOR_GAMES))
    inner["strength_at_cutoff"] = {
        player: math.log((wins + prior) / (losses + prior))
        for player, (wins, losses) in totals.items()
    }
    return inner, boundary.isoformat()


def _features_for_lists(counts_rows: Sequence[Mapping[str, int]], slot_names: Sequence[tuple[str, int]]) -> Any:
    slot_ids = {tuple(slot): index for index, slot in enumerate(slot_names)}
    row_ids: list[int] = []
    col_ids: list[int] = []
    for index, counts in enumerate(counts_rows):
        for card, count in counts.items():
            for copy_number in range(1, int(count) + 1):
                column = slot_ids.get((str(card), copy_number))
                if column is not None:
                    row_ids.append(index)
                    col_ids.append(column)
    return sparse.csr_matrix(
        (np.ones(len(row_ids), dtype=float), (row_ids, col_ids)), shape=(len(counts_rows), len(slot_ids))
    )


def _loss(logits: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
    if len(y) == 0 or not np.isfinite(logits).all() or weights.sum() <= 0:
        return math.inf
    return float(np.average(np.logaddexp(0.0, logits) - y * logits, weights=weights))


def _view_weights(rows: Sequence[Mapping[str, Any]], *, focal: bool) -> np.ndarray:
    weights = np.asarray([float(row.get("weight", 0.5)) for row in rows], dtype=float)
    if not focal:
        return weights
    positions: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        positions[tuple(row.get("pairing_key", ()))].append(index)
    for indices in positions.values():
        total = float(weights[indices].sum())
        if total > 0:
            weights[indices] /= total
        else:
            weights[indices] = 1.0 / len(indices)
    return weights


def _unit_pair_view_weights(
    rows: Sequence[Mapping[str, Any]], weights: np.ndarray
) -> np.ndarray:
    """Normalise each represented underlying pairing to one view-weight unit."""
    result = np.asarray(weights, dtype=float).copy()
    positions: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        positions[tuple(row.get("pairing_key", ()))].append(index)
    for indices in positions.values():
        total = float(result[indices].sum())
        if total > 0:
            result[indices] /= total
        else:
            result[indices] = 1.0 / len(indices)
    return result


def _groups(scope: str, labels: np.ndarray) -> np.ndarray:
    if scope == "archetype":
        return labels
    return np.full(len(labels), "pooled", dtype=object)


def _effective_penalty(config: Mapping[str, Any], weights: np.ndarray) -> float:
    if "prior_sd" in config:
        return 1.0 / (float(weights.sum()) * float(config["prior_sd"]) ** 2)
    # Nuisance-only fits have no regularised card component.
    return float(config.get("penalty", 0.0))


def _params(config: Mapping[str, Any], weights: np.ndarray) -> dict[str, Any]:
    params = {
        "kind": str(config["kind"]),
        "rank": int(config.get("rank", 2)),
        "seed": int(config.get("seed", 1312)),
        "penalty": _effective_penalty(config, weights),
        "group_scope": str(config.get("group_scope", "pooled")),
    }
    return params


def _max_iterations(protocol: Mapping[str, Any], config: Mapping[str, Any]) -> int:
    estimators = protocol["estimators"]
    kind = str(config.get("kind", ""))
    if kind == "nuisance":
        model_name = "nuisance"
    elif kind == "additive" and config.get("group_scope") == "archetype":
        model_name = "per_archetype_additive"
    elif kind == "additive":
        model_name = "pooled_additive"
    elif kind == "pairs":
        model_name = "regularised_selected_pairs"
    elif kind == "fm":
        model_name = "pooled_factorisation_machine"
    else:
        raise ValueError(f"unknown model kind: {kind}")
    return int(estimators.get("max_iterations_by_model", {}).get(model_name, estimators.get("max_iterations", 5000)))


def _fit(
    X: Any,
    y: np.ndarray,
    nuisance: Any,
    labels: np.ndarray,
    weights: np.ndarray,
    config: Mapping[str, Any],
    *,
    max_iterations: int,
    max_seconds: float | None,
) -> FittedModel:
    params = _params(config, weights)
    return fit_model(
        X,
        y,
        nuisance=nuisance,
        groups=_groups(params["group_scope"], labels),
        kind=params["kind"],
        rank=params["rank"],
        seed=params["seed"],
        penalty=params["penalty"],
        main_effect_penalty=(
            float(config["main_effect_penalty"]) if config.get("main_effect_penalty") is not None else None
        ),
        max_iter=max_iterations,
        sample_weight=weights,
        max_seconds=max_seconds,
    )


def _config_key(config: Mapping[str, Any]) -> str:
    return json.dumps(dict(config), sort_keys=True, separators=(",", ":"))


def _load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"schema_version": 1, "fit_records": {}, "pilot": None}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or not isinstance(payload.get("fit_records"), dict):
        raise ValueError("validation checkpoint has an unsupported schema")
    return payload


def _save_fit(path: Path, checkpoint: dict[str, Any]) -> None:
    _atomic_json(path, checkpoint)


def _event_rows(rows: Sequence[Mapping[str, Any]], event_ids: set[str]) -> list[Mapping[str, Any]]:
    return [row for row in rows if str(row.get("event")) in event_ids]


def _folds(
    context: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], fold_count: int = 3
) -> list[dict[str, Any]]:
    event_blocks = context.get("event_block_ids", {})
    event_to_block: dict[str, str] = {}
    for row in rows:
        event = str(row.get("event"))
        event_to_block[event] = str(event_blocks.get(event, _event_block(context, row)))
    unique_blocks = sorted(set(event_to_block.values()), key=lambda value: value.rsplit(":", 1)[-1])
    if fold_count < 1 or len(unique_blocks) < fold_count + 1:
        return []
    chunks = [list(chunk) for chunk in np.array_split(np.asarray(unique_blocks, dtype=object), fold_count + 1)]
    result = []
    for index in range(1, fold_count + 1):
        validation_blocks = set(chunks[index])
        first_validation_date = _date_value(sorted(validation_blocks, key=str)[0].rsplit(":", 1)[-1])
        if first_validation_date is not None:
            boundary = first_validation_date - timedelta(days=7)
            training_blocks = {
                block for chunk in chunks[:index] for block in chunk
                if (_date_value(str(block).rsplit(":", 1)[-1]) or date.min) < boundary
            }
        else:
            training_blocks = {block for chunk in chunks[:index] for block in chunk}
        if not training_blocks:
            continue
        training_events = {event for event, block in event_to_block.items() if block in training_blocks}
        validation_events = {event for event, block in event_to_block.items() if block in validation_blocks}
        if training_events and validation_events:
            result.append({
                "fold": index,
                "training_event_ids": sorted(training_events),
                "validation_event_ids": sorted(validation_events),
            })
    return result


def _training_slots(context: Mapping[str, Any], events: set[str], minimum: int) -> list[tuple[str, int]]:
    lists = [row for row in context.get("training_lists", ()) if str(row.get("event")) in events]
    support: Counter[tuple[str, int]] = Counter()
    for row in lists:
        for card, copies in row.get("counts", {}).items():
            for copy_number in range(1, int(copies) + 1):
                support[(str(card), copy_number)] += 1
    return sorted(
        slot for slot, count in support.items()
        if count >= minimum and len(lists) - count >= minimum
    )


def _benchmark_rows(
    context: Mapping[str, Any], validation_views: Sequence[Mapping[str, Any]], archetype_id: str
) -> dict[str, tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]]:
    training = list(context.get("eligible_views", ()))
    validation = list(validation_views)
    return {
        "pooled": (training, validation),
        archetype_id: (
            [row for row in training if str(row.get("archetype_id")) == archetype_id],
            [row for row in validation if str(row.get("archetype_id")) == archetype_id],
        ),
    }


def _benchmark_order(
    benchmark_rows: Mapping[str, Any], archetype_id: str, protocol: Mapping[str, Any]
) -> list[str]:
    configured = protocol.get("features", {}).get("focal_benchmarks")
    if isinstance(configured, Sequence) and not isinstance(configured, (str, bytes)):
        order = [str(value) for value in configured if str(value) in benchmark_rows]
        order.extend(value for value in benchmark_rows if value not in order)
        return order
    return [value for value in (archetype_id, "pooled") if value in benchmark_rows]


def _model_configs(
    protocol: Mapping[str, Any],
    *,
    group_scope: str | None = None,
    main_effect_penalty: float | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    estimators = protocol["estimators"]
    max_iterations = int(estimators.get("max_iterations", 5000))
    penalties = [float(value) for value in estimators.get("interaction_penalties", [1e-5, 1e-4, 1e-3])]
    ranks = [int(value) for value in estimators.get("fm_ranks", [2, 4, 8])]
    seeds = [int(value) for value in estimators.get("fm_seeds", [1312, 1313, 1314])]
    if not penalties or not ranks or not seeds:
        raise ValueError("the frozen estimator grid must contain penalties, FM ranks and seeds")
    scope = group_scope or "pooled"
    pair_configs = [
        {"kind": "pairs", "penalty": penalty, "group_scope": scope,
         "main_effect_penalty": main_effect_penalty}
        for penalty in penalties
    ]
    fm_configs = [
        {"kind": "fm", "penalty": penalty, "rank": rank, "seed": seed,
         "group_scope": scope, "main_effect_penalty": main_effect_penalty}
        for rank in ranks for seed in seeds for penalty in penalties
    ]
    return pair_configs, fm_configs


def _additive_configs() -> list[dict[str, Any]]:
    from src.core.config import BDIF_BEST60_PRIOR_GRID

    return [
        {"kind": "additive", "prior_sd": float(prior), "group_scope": scope}
        for scope in ("pooled", "archetype") for prior in BDIF_BEST60_PRIOR_GRID
    ]


def _score_config(
    X: Any,
    y: np.ndarray,
    nuisance: Any,
    labels: np.ndarray,
    weights: np.ndarray,
    validation_X: Any,
    validation_y: np.ndarray,
    validation_nuisance: Any,
    validation_labels: np.ndarray,
    validation_weights: np.ndarray,
    config: Mapping[str, Any],
    *,
    max_iterations: int,
    max_seconds: float | None,
) -> dict[str, Any]:
    model = _fit(
        X, y, nuisance, labels, weights, config,
        max_iterations=max_iterations, max_seconds=max_seconds,
    )
    prediction_groups = _groups(str(config.get("group_scope", "pooled")), validation_labels)
    logits = model.predict_logit(validation_X, nuisance=validation_nuisance, groups=prediction_groups)
    loss = _loss(logits, validation_y, validation_weights)
    return {
        "status": "complete" if model.converged and np.isfinite(loss) else "fit_failed",
        "converged": bool(model.converged),
        "optimizer_status": int(model.status),
        "optimizer_message": str(model.message),
        "iterations": int(model.iterations),
        "gradient_norm": float(model.gradient_norm),
        "parameter_count": int(model.parameter_count),
        "validation_log_loss": loss if np.isfinite(loss) else None,
        "config": dict(config),
        "effective_penalty": _effective_penalty(config, weights),
        "effective_main_effect_penalty": (
            float(config["main_effect_penalty"])
            if config.get("main_effect_penalty") is not None
            else _effective_penalty(config, weights)
        ),
    }


def _cached_score(
    cache_id: str,
    X: Any,
    y: np.ndarray,
    nuisance: Any,
    labels: np.ndarray,
    weights: np.ndarray,
    validation_X: Any,
    validation_y: np.ndarray,
    validation_nuisance: Any,
    validation_labels: np.ndarray,
    validation_weights: np.ndarray,
    config: Mapping[str, Any],
    *,
    checkpoint: dict[str, Any],
    checkpoint_path: Path,
    max_iterations: int,
    max_seconds: float | None,
    candidate_fit_seconds: float | None = None,
) -> dict[str, Any]:
    records = checkpoint["fit_records"]
    if cache_id in records:
        cached = records[cache_id]
        if (
            cached.get("status") != "fit_timeout"
            or cached.get("candidate_fit_seconds") == candidate_fit_seconds
        ):
            return cached
        del records[cache_id]
    budgets = [value for value in (max_seconds, candidate_fit_seconds) if value is not None]
    fit_seconds = min(budgets) if budgets else None
    fit_started = time.monotonic()
    try:
        value = _score_config(
            X, y, nuisance, labels, weights, validation_X, validation_y,
            validation_nuisance, validation_labels, validation_weights, config,
            max_iterations=max_iterations, max_seconds=fit_seconds,
        )
    except FitBudgetExceeded as exc:
        stage_budget_remains = (
            max_seconds is None or time.monotonic() - fit_started < max_seconds
        )
        candidate_is_tighter = (
            candidate_fit_seconds is not None
            and (max_seconds is None or candidate_fit_seconds < max_seconds)
            and stage_budget_remains
        )
        if candidate_is_tighter:
            value = {
                "status": "fit_timeout",
                "converged": False,
                "timeout_scope": "candidate_budget",
                "candidate_fit_seconds": float(candidate_fit_seconds),
                "fit_seconds": fit_seconds,
                "reason": str(exc),
                "config": dict(config),
                "effective_penalty": _effective_penalty(config, weights),
            }
        else:
            # Do not save this result. The incomplete candidate remains pending
            # and can be retried when the stage resumes with a fresh budget.
            return {
                "status": "stage_timeout",
                "converged": False,
                "timeout_scope": "stage_budget",
                "fit_seconds": fit_seconds,
                "reason": str(exc),
                "config": dict(config),
            }
    records[cache_id] = value
    _save_fit(checkpoint_path, checkpoint)
    return value


def _model_order(record: Mapping[str, Any]) -> tuple[float, int, str]:
    loss = record.get("validation_log_loss")
    complexity = int(record.get("parameter_count", 10**12))
    return (float(loss) if loss is not None else math.inf, complexity, _config_key(record.get("config", {})))


def _best(records: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    eligible = [dict(record) for record in records if record.get("status") == "complete"]
    return min(eligible, key=_model_order) if eligible else None


def _pilot(
    X: Any, y: np.ndarray, nuisance: Any, groups: np.ndarray, weights: np.ndarray,
    *, max_iterations: int, max_seconds: float | None, rank: int, seed: int, penalty: float,
) -> dict[str, Any]:
    started = time.monotonic()
    try:
        model = fit_model(
            X, y, nuisance=nuisance, groups=_groups("pooled", groups), kind="fm", rank=rank,
            seed=seed, penalty=penalty, max_iter=min(max_iterations, 500), sample_weight=weights,
            max_seconds=max_seconds,
        )
    except FitBudgetExceeded as exc:
        return {
            "status": "fit_timeout", "seconds": time.monotonic() - started,
            "rows": int(len(y)), "features": int(X.shape[1]), "converged": False,
            "rank": rank, "seed": seed, "penalty": penalty, "reason": str(exc),
        }
    return {
        "status": "complete" if model.converged else "fit_failed",
        "seconds": time.monotonic() - started,
        "rows": int(len(y)),
        "features": int(X.shape[1]),
        "iterations": int(model.iterations),
        "converged": bool(model.converged),
        "rank": rank,
        "seed": seed,
        "penalty": penalty,
    }


def validate_experiment(
    context: dict[str, Any],
    validation_views: Sequence[Mapping[str, Any]],
    protocol: dict[str, Any],
    *,
    archetype_id: str,
    checkpoint_path: str | Path,
    max_seconds: int | None,
) -> dict[str, Any]:
    """Tune on development and validation rows only; never inspect test outcomes."""
    checkpoint_file = Path(checkpoint_path)
    checkpoint = _load_checkpoint(checkpoint_file)
    maximum = None if max_seconds is None else max(0.0, float(max_seconds))
    started = time.monotonic()
    maximum_iterations = int(protocol["estimators"].get("max_iterations", 5000))
    candidate_fit_seconds = protocol["estimators"].get("candidate_fit_seconds")
    if candidate_fit_seconds is not None:
        if isinstance(candidate_fit_seconds, bool):
            raise ValueError("estimators.candidate_fit_seconds must be numeric, not boolean")
        candidate_fit_seconds = float(candidate_fit_seconds)
        if not np.isfinite(candidate_fit_seconds) or candidate_fit_seconds <= 0:
            raise ValueError("estimators.candidate_fit_seconds must be finite and positive when set")
    min_views = 20
    benchmarks: dict[str, Any] = {}
    benchmark_rows = _benchmark_rows(context, validation_views, archetype_id)
    benchmark_order = _benchmark_order(benchmark_rows, archetype_id, protocol)
    for benchmark in benchmark_order:
        train_rows, valid_rows = benchmark_rows[benchmark]
        if len(train_rows) < min_views or len(valid_rows) < min_views:
            benchmarks[benchmark] = {
                "status": "inconclusive",
                "reason": "fewer_than_20_training_or_validation_views",
                "training_views": len(train_rows),
                "validation_views": len(valid_rows),
            }
            continue
        X, y, nuisance, groups, schema = _design(train_rows, context, training=True)
        valid_X, valid_y, valid_nuisance, valid_groups, diagnostics = _design(
            valid_rows, context, training=False, schema=schema
        )
        focal_benchmark = benchmark != "pooled"
        weights = _view_weights(train_rows, focal=focal_benchmark)
        valid_weights = _view_weights(valid_rows, focal=focal_benchmark)
        if maximum is None or time.monotonic() - started < maximum:
            if checkpoint.get("pilot") is None:
                pilot_records = benchmark_rows["pooled"][0]
                pilot_X, pilot_y, pilot_nuisance, pilot_groups, _ = _design(pilot_records, context, training=True)
                pilot_weights = _view_weights(pilot_records, focal=False)
                checkpoint["pilot"] = _pilot(
                    pilot_X, pilot_y, pilot_nuisance, pilot_groups, pilot_weights,
                    max_iterations=_max_iterations(protocol, {"kind": "fm"}),
                    max_seconds=min(
                        [value for value in (
                            candidate_fit_seconds,
                            None if maximum is None else max(0.001, maximum - (time.monotonic() - started)),
                        ) if value is not None], default=None,
                    ),
                    rank=int(protocol.get("simulation", {}).get("quick_rank", 2)),
                    seed=int(protocol.get("simulation", {}).get("quick_seed", 1312)),
                    penalty=float(protocol.get("simulation", {}).get("quick_penalty", 0.001)),
                )
                _save_fit(checkpoint_file, checkpoint)
        rolling = []
        stage_expired = False
        for fold in _folds(
            context, train_rows, int(protocol.get("estimators", {}).get("rolling_folds", 3))
        ):
            train_fold = _event_rows(train_rows, set(fold["training_event_ids"]))
            valid_fold = _event_rows(train_rows, set(fold["validation_event_ids"]))
            if len(train_fold) < min_views or len(valid_fold) < min_views:
                rolling.append({**fold, "status": "inconclusive", "reason": "fewer_than_20_views"})
                continue
            fold_context = dict(context)
            fold_context["slot_names"] = _training_slots(
                context, set(fold["training_event_ids"]), int(protocol["features"].get("slot_minimum_lists", 30))
            )
            fold_X, fold_y, fold_nuisance, fold_groups, fold_schema = _design(
                train_fold, fold_context, training=True
            )
            inner_context, inner_strength_boundary = _inner_validation_context(
                fold_context,
                valid_fold,
                embargo_days=int(context.get("embargo_days", 7)),
            )
            inner_X, inner_y, inner_nuisance, inner_groups, _ = _design(
                valid_fold, inner_context, training=False, schema=fold_schema
            )
            fold_weights = _view_weights(train_fold, focal=focal_benchmark)
            inner_weights = _view_weights(valid_fold, focal=focal_benchmark)
            configs = _additive_configs()
            fold_results = []
            for config in configs:
                cache_id = ":".join((benchmark, "rolling", str(fold["fold"]), _config_key(config)))
                result = _cached_score(
                    cache_id, fold_X, fold_y, fold_nuisance, fold_groups, fold_weights,
                    inner_X, inner_y, inner_nuisance, inner_groups, inner_weights, config,
                    checkpoint=checkpoint, checkpoint_path=checkpoint_file,
                    max_iterations=_max_iterations(protocol, config),
                    max_seconds=None if maximum is None else max(0.0, maximum - (time.monotonic() - started)),
                    candidate_fit_seconds=candidate_fit_seconds,
                )
                fold_results.append(result)
                if result.get("status") == "stage_timeout":
                    stage_expired = True
                    break
            fold_reference = _best(fold_results)
            if fold_reference:
                fold_main_penalty = _effective_penalty(fold_reference["config"], fold_weights)
                fold_scope = str(fold_reference["config"].get("group_scope", "pooled"))
                middle_penalty = float(protocol["estimators"].get("interaction_penalties", [1e-4])[min(
                    1, len(protocol["estimators"].get("interaction_penalties", [1e-4])) - 1
                )])
                representative = [
                    {"kind": "pairs", "penalty": middle_penalty, "group_scope": fold_scope,
                     "main_effect_penalty": fold_main_penalty},
                    {"kind": "fm", "rank": int(protocol["estimators"].get("fm_ranks", [2])[0]),
                     "seed": int(protocol["estimators"].get("fm_seeds", [1312])[0]),
                     "penalty": middle_penalty, "group_scope": fold_scope,
                     "main_effect_penalty": fold_main_penalty},
                ]
                for config in representative:
                    cache_id = ":".join((benchmark, "rolling", str(fold["fold"]), _config_key(config)))
                    result = _cached_score(
                        cache_id, fold_X, fold_y, fold_nuisance, fold_groups, fold_weights,
                        inner_X, inner_y, inner_nuisance, inner_groups, inner_weights, config,
                        checkpoint=checkpoint, checkpoint_path=checkpoint_file,
                        max_iterations=_max_iterations(protocol, config),
                        max_seconds=None if maximum is None else max(0.0, maximum - (time.monotonic() - started)),
                        candidate_fit_seconds=candidate_fit_seconds,
                    )
                    fold_results.append(result)
                    if result.get("status") == "stage_timeout":
                        stage_expired = True
                        break
            rolling.append({
                **fold,
                "strength_frozen_before": inner_strength_boundary,
                "status": (
                    "incomplete" if stage_expired
                    else "limited" if any(result.get("status") in {"fit_timeout", "fit_failed"} for result in fold_results)
                    else "complete" if fold_results and all(result.get("status") == "complete" for result in fold_results)
                    else "inconclusive"
                ),
                "fit_results": fold_results,
            })
            if stage_expired:
                break
        if stage_expired:
            benchmarks[benchmark] = {
                "status": "incomplete",
                "reason": "stage_budget_exhausted_during_rolling_folds",
                "rolling_development_folds": rolling,
                "validation_model_results": [],
                "selected_models": {"primary_additive": None, "best_interaction": None},
            }
            break
        additive_configs = _additive_configs()
        validation_results = []
        baseline_configs = [{"kind": "nuisance", "group_scope": "pooled"}, *additive_configs]
        for config in baseline_configs:
            cache_id = ":".join((benchmark, "validation", _config_key(config)))
            record = _cached_score(
                cache_id, X, y, nuisance, groups, weights, valid_X, valid_y,
                valid_nuisance, valid_groups, valid_weights, config,
                checkpoint=checkpoint, checkpoint_path=checkpoint_file,
                max_iterations=_max_iterations(protocol, config),
                max_seconds=None if maximum is None else max(0.0, maximum - (time.monotonic() - started)),
                candidate_fit_seconds=candidate_fit_seconds,
            )
            validation_results.append(record)
            if record.get("status") == "stage_timeout":
                stage_expired = True
                break
        if stage_expired:
            benchmarks[benchmark] = {
                "status": "incomplete",
                "reason": "stage_budget_exhausted_during_additive_validation_fits",
                "rolling_development_folds": rolling,
                "validation_model_results": validation_results,
                "selected_models": {"primary_additive": None, "best_interaction": None},
            }
            break
        additive = [record for record in validation_results if record["config"]["kind"] == "additive"]
        reference = _best(additive)
        if reference:
            main_effect_penalty = _effective_penalty(reference["config"], weights)
            group_scope = str(reference["config"].get("group_scope", "pooled"))
        else:
            main_effect_penalty, group_scope = None, "pooled"
        pair_configs, fm_configs = _model_configs(
            protocol, group_scope=group_scope, main_effect_penalty=main_effect_penalty
        )
        for config in [*pair_configs, *fm_configs]:
            cache_id = ":".join((benchmark, "validation", _config_key(config)))
            record = _cached_score(
                cache_id, X, y, nuisance, groups, weights, valid_X, valid_y,
                valid_nuisance, valid_groups, valid_weights, config,
                checkpoint=checkpoint, checkpoint_path=checkpoint_file,
                max_iterations=_max_iterations(protocol, config),
                max_seconds=None if maximum is None else max(0.0, maximum - (time.monotonic() - started)),
                candidate_fit_seconds=candidate_fit_seconds,
            )
            validation_results.append(record)
            if record.get("status") == "stage_timeout":
                stage_expired = True
                break
        if stage_expired:
            benchmarks[benchmark] = {
                "status": "incomplete",
                "reason": "stage_budget_exhausted_during_interaction_validation_fits",
                "rolling_development_folds": rolling,
                "validation_model_results": validation_results,
                "primary_reference": _best([
                    row for row in validation_results if row["config"]["kind"] == "additive"
                ]),
                "selected_models": {"primary_additive": None, "best_interaction": None},
            }
            break
        pair_best = _best([record for record in validation_results if record["config"]["kind"] == "pairs"])
        fm_best = _best([record for record in validation_results if record["config"]["kind"] == "fm"])
        interaction = _best([record for record in (pair_best, fm_best) if record is not None])
        candidate_statuses = {"complete", "fit_timeout", "fit_failed"}
        completed = all(record.get("status") in candidate_statuses for record in validation_results)
        rolling_complete = all(fold.get("status") in {"complete", "limited", "inconclusive"} for fold in rolling)
        rolling_results = [
            record for fold in rolling for record in fold.get("fit_results", ())
        ]
        limited_records = [
            record for record in [*validation_results, *rolling_results]
            if record.get("status") in {"fit_timeout", "fit_failed"}
        ]
        limited_search = bool(limited_records)
        benchmarks[benchmark] = {
            "status": (
                "complete" if completed and rolling_complete and reference and interaction
                else "inconclusive"
            ),
            "training_views": len(train_rows),
            "validation_views": len(valid_rows),
            "training_pairings": len({tuple(row.get("pairing_key", ())) for row in train_rows}),
            "validation_pairings": len({tuple(row.get("pairing_key", ())) for row in valid_rows}),
            "unknown_validation_categories": diagnostics["unknown_categories"],
            "rolling_development_folds": rolling,
            "validation_model_results": validation_results,
            "primary_reference": reference,
            "best_interaction": interaction,
            "selected_models": {
                "primary_additive": reference.get("config") if reference else None,
                "best_interaction": interaction.get("config") if interaction else None,
            },
            "search_completeness": "limited_by_candidate_fit_timeouts_or_failures" if limited_search else "complete_grid",
            "candidate_fit_timeout_count": sum(record.get("status") == "fit_timeout" for record in limited_records),
            "candidate_fit_failure_count": sum(record.get("status") == "fit_failed" for record in limited_records),
            "selection_rule": "lowest validation log loss, then fewer parameters, then lexical config",
        }
    statuses = [row["status"] for row in benchmarks.values()]
    simulation_pilot = None
    quick_replicates = int(protocol.get("simulation", {}).get("quick_replicates_per_effect", 0))
    if quick_replicates > 0 and statuses and all(value in {"complete", "inconclusive"} for value in statuses):
        from .simulation import run_simulation

        simulation_context = prepare_simulation_context(context)
        simulation_settings = protocol.get("simulation", {})
        simulation_pilot = run_simulation(
            simulation_context,
            replicates=quick_replicates,
            base_seed=int(simulation_settings.get("quick_seed", 1312)),
            effect=float(simulation_settings.get("quick_effect", 0.2)),
            mode="low_rank",
            checkpoint=checkpoint_file.with_name(checkpoint_file.stem + "-simulation.json"),
            max_seconds=None if maximum is None else _remaining(started + maximum),
            rank_grid=(int(simulation_settings.get("quick_rank", 2)),),
            interaction_penalties=(float(simulation_settings.get("quick_penalty", 0.001)),),
            seed_grid=(int(simulation_settings.get("quick_seed", 1312)),),
            max_iter=min(maximum_iterations, 500),
        )
    status = (
        "complete" if statuses and all(value in {"complete", "inconclusive", "not_identified"} for value in statuses)
        else "incomplete"
    )
    if simulation_pilot and simulation_pilot.get("status") == "incomplete":
        status = "incomplete"
    return {
        "status": status,
        "benchmarks": benchmarks,
        "pilot": checkpoint.get("pilot"),
        "simulation_pilot": simulation_pilot,
        "test_outcomes_read": False,
        "estimation_selection_separation": {
            "rolling_folds": "expanding chronological development date blocks with seven-day embargo",
            "final_selection": "validation log loss only",
            "test_use": "none",
        },
        "primary_reference_rule": "stronger additive variant by validation log loss",
    }


def prepare_simulation_context(
    context: Mapping[str, Any], archetype_id: str | None = None
) -> dict[str, Any]:
    """Encode the same frozen training views for the outcome-injection simulator."""
    rows = list(context.get("eligible_views", ()))
    if archetype_id is not None:
        rows = [row for row in rows if str(row.get("archetype_id")) == archetype_id]
    X, y, nuisance, groups, _ = _design(rows, context, training=True)
    pairing_ids = _object_array([tuple(row.get("pairing_key", ())) for row in rows])
    event_ids = _object_array([str(row.get("event") or "unknown") for row in rows])
    player_ids = _object_array([str(row.get("player") or "unknown") for row in rows])
    opponent_ids = _object_array([str(row.get("opponent") or "unknown") for row in rows])
    side = np.asarray([
        -1.0 if row.get("side") in (2, -1, "2", "-1") else 1.0 for row in rows
    ])
    block_ids = _object_array([_event_block(context, row) for row in rows])
    weights = _view_weights(rows, focal=archetype_id is not None)
    slot_names = [tuple(slot) for slot in context.get("slot_names", ())]
    cooccur = (X.T @ X).tocoo()
    feature_pair = None
    for i, j, count in sorted(zip(cooccur.row, cooccur.col, cooccur.data))[:5000]:
        if i >= j or count <= 0 or slot_names[i][0] == slot_names[j][0]:
            continue
        pair_matrix = X[:, [int(i), int(j)]].toarray() > 0
        if len(np.unique(pair_matrix, axis=0)) == 4:
            feature_pair = (int(i), int(j))
            break
    arrays = {
        "X": X,
        "y": y,
        "base_y": y,
        "nuisance": nuisance,
        "groups": groups,
        "pairing_ids": pairing_ids,
        "event_ids": event_ids,
        "player_ids": player_ids,
        "opponent_ids": opponent_ids,
        "list_ids": _object_array([
            (str(row.get("event") or "unknown"), str(row.get("player") or "unknown")) for row in rows
        ]),
        "sample_weight": weights,
        "side": side,
        "block_ids": block_ids,
        "event_block_ids": dict(context.get("event_block_ids", {})),
        "pair_base": np.zeros(len(rows), dtype=float),
        "feature_pair": feature_pair,
        "feature_names": [slot[0] for slot in slot_names],
        "feature_pair_status": "identified_non_aliased" if feature_pair else "not_identified",
        "simulation_scope": archetype_id or "pooled",
        "synthetic_design": False,
    }
    result = dict(context)
    result.update(arrays)
    return result


def _target_rows(rows: Sequence[Mapping[str, Any]], archetype_id: str | None) -> list[Mapping[str, Any]]:
    if archetype_id is None or archetype_id == "pooled":
        return list(rows)
    return [row for row in rows if str(row.get("archetype_id")) == archetype_id]


def _score_count_mapping(
    counts: Mapping[str, int],
    model: FittedModel,
    schema: Mapping[str, Any],
    context: Mapping[str, Any],
    archetype_id: str,
    field: Mapping[str, float],
    group_scope: str,
) -> float:
    opponent_ids = sorted(str(key) for key, value in field.items() if float(value) > 0)
    if not opponent_ids:
        raise ValueError("recommendation field has no positive mass")
    synthetic = []
    for opponent in opponent_ids:
        row = {
            "counts": counts,
            "won": 0,
            "archetype_id": archetype_id,
            "opponent_archetype_id": opponent,
            "player": "__recommendation__",
            "opponent": "__opponent__",
            "event": "__recommendation__",
            "date": context.get("cutoff"),
            "source": "unknown",
            "platform": "unknown",
            "delivery_mode": "unknown",
            "match_mode": "unknown",
            "format": "unknown",
            "game": "unknown",
            "is_online": None,
        }
        synthetic.append(row)
    X = _features_for_lists([counts for _ in synthetic], context["slot_names"])
    _, _, nuisance, labels, _ = _design(synthetic, context, training=False, schema=schema)
    logits = model.predict_logit(X, nuisance=nuisance, groups=_groups(group_scope, labels))
    weights = np.asarray([float(field.get(row["opponent_archetype_id"], 0.0)) for row in synthetic])
    return float(field_scores(logits, weights))


def _vote_list(context: Mapping[str, Any], archetype_id: str) -> dict[str, int] | None:
    source = [row for row in context.get("training_lists", ()) if str(row.get("archetype_id")) == archetype_id]
    entries = []
    for row in source:
        entries.append(ArchetypeList(
            event=str(row.get("event") or ""),
            player=str(row.get("player") or ""),
            deck_id=None if row.get("deck_id") is None else str(row.get("deck_id")),
            date=str(row.get("date") or ""),
            players=int(row.get("players") or 0),
            placing=None if row.get("placing") is None else int(row["placing"]),
            wins=int(row.get("wins") or 0),
            losses=int(row.get("losses") or 0),
            counts={str(name): int(count) for name, count in row.get("counts", {}).items()},
            groups={str(name): str(group) for name, group in row.get("groups", {}).items()},
        ))
    if not entries:
        return None
    vote = dict(consensus_sixty(entries))
    try:
        validate_recommendation([{"card": card, "copies": copies} for card, copies in vote.items()])
    except ValueError:
        return None
    return vote


def _model_for_config(
    train_rows: Sequence[Mapping[str, Any]],
    context: Mapping[str, Any],
    config: Mapping[str, Any],
    max_iterations: int,
    max_seconds: float | None,
    schema: Mapping[str, Any] | None = None,
    focal: bool = False,
) -> tuple[FittedModel, Mapping[str, Any]]:
    X, y, nuisance, groups, schema = _design(train_rows, context, training=True, schema=schema)
    weights = _view_weights(train_rows, focal=focal)
    model = _fit(
        X, y, nuisance, groups, weights, config,
        max_iterations=max_iterations, max_seconds=max_seconds,
    )
    return model, schema


def _model_cache_identity(
    X: Any,
    y: np.ndarray,
    nuisance: Any,
    groups: np.ndarray,
    weights: np.ndarray,
    schema: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    benchmark: str,
    role: str,
) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(
        {"benchmark": benchmark, "role": role, "config": dict(config), "schema": dict(schema)},
        default=_json_default, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8"))
    for matrix in (X, nuisance):
        csr = sparse.csr_matrix(matrix)
        digest.update(np.asarray(csr.shape, dtype=np.int64).tobytes())
        for array in (csr.data, csr.indices, csr.indptr):
            digest.update(np.asarray(array).tobytes())
    for array in (y, groups.astype(str), weights):
        digest.update(np.asarray(array).tobytes())
    return digest.hexdigest()


def _write_fitted_model(path: Path, model: FittedModel, schema: Mapping[str, Any], cache_key: str) -> None:
    metadata = {
        "cache_key": cache_key,
        "schema": dict(schema),
        "kind": model.kind,
        "pair_indices": [list(pair) for pair in model.pair_indices],
        "feature_count": int(model.feature_count),
        "group_levels": list(model.group_levels),
        "reference_group": model.reference_group,
        "rank": int(model.rank),
        "penalty": float(model.penalty),
        "main_effect_penalty": float(model.main_effect_penalty),
        "seed": int(model.seed),
        "converged": bool(model.converged),
        "status": int(model.status),
        "message": str(model.message),
        "objective": float(model.objective),
        "iterations": int(model.iterations),
        "gradient_norm": float(model.gradient_norm),
        "parameter_count": int(model.parameter_count),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npz", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(
                stream,
                metadata=np.asarray(json.dumps(metadata, default=_json_default, sort_keys=True)),
                nuisance_coef=model.nuisance_coef,
                main_coef=model.main_coef,
                interaction_coef=model.interaction_coef,
                factors=model.factors,
                deviation_coef=model.deviation_coef,
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _read_fitted_model(path: Path, cache_key: str) -> tuple[FittedModel, Mapping[str, Any]] | None:
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"].item()))
        if metadata.get("cache_key") != cache_key:
            return None
        arrays = {name: np.asarray(archive[name]).copy() for name in (
            "nuisance_coef", "main_coef", "interaction_coef", "factors", "deviation_coef",
        )}
    schema = dict(metadata["schema"])
    schema["categories"] = tuple(schema.get("categories", ()))
    model = FittedModel(
        kind=str(metadata["kind"]),
        nuisance_coef=arrays["nuisance_coef"],
        main_coef=arrays["main_coef"],
        interaction_coef=arrays["interaction_coef"],
        pair_indices=tuple(tuple(int(index) for index in pair) for pair in metadata["pair_indices"]),
        factors=arrays["factors"],
        feature_count=int(metadata["feature_count"]),
        group_levels=tuple(metadata["group_levels"]),
        reference_group=metadata["reference_group"],
        deviation_coef=arrays["deviation_coef"],
        rank=int(metadata["rank"]),
        penalty=float(metadata["penalty"]),
        main_effect_penalty=float(metadata["main_effect_penalty"]),
        seed=int(metadata["seed"]),
        converged=bool(metadata["converged"]),
        status=int(metadata["status"]),
        message=str(metadata["message"]),
        objective=float(metadata["objective"]),
        iterations=int(metadata["iterations"]),
        gradient_norm=float(metadata["gradient_norm"]),
        parameter_count=int(metadata["parameter_count"]),
    )
    return model, schema


def _evaluation_model(
    train_rows: Sequence[Mapping[str, Any]],
    context: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    benchmark: str,
    role: str,
    cache_dir: Path,
    max_iterations: int,
    max_seconds: float | None,
) -> tuple[FittedModel, Mapping[str, Any], bool]:
    X, y, nuisance, groups, schema = _design(train_rows, context, training=True)
    weights = _view_weights(train_rows, focal=benchmark != "pooled")
    cache_key = _model_cache_identity(
        X, y, nuisance, groups, weights, schema, config, benchmark=benchmark, role=role,
    )
    cache_token = hashlib.sha256(f"{benchmark}\0{role}".encode("utf-8")).hexdigest()[:20]
    cache_path = cache_dir / f"{cache_token}.npz"
    cached = _read_fitted_model(cache_path, cache_key)
    if cached is not None:
        model, cached_schema = cached
        return model, cached_schema, True
    model = _fit(
        X, y, nuisance, groups, weights, config,
        max_iterations=max_iterations, max_seconds=max_seconds,
    )
    _write_fitted_model(cache_path, model, schema, cache_key)
    return model, schema, False


def _fixed_interval(
    context: dict[str, Any],
    counts: Mapping[str, int],
    model_config: Mapping[str, Any],
    field: Mapping[str, float],
    archetype_id: str,
    schema: Mapping[str, Any],
    protocol: Mapping[str, Any],
    *,
    max_seconds: float | None,
    checkpoint_path: Path,
    benchmark: str,
) -> dict[str, Any]:
    original_rows = list(context.get("eligible_views", ()))
    lookup = defaultdict(list)
    for row in original_rows:
        lookup[str(row.get("event"))].append(row)
    started = time.monotonic()

    def fit_and_score(sample: Mapping[str, Any], fixed_counts: Mapping[str, int]) -> float:
        sampled_rows = [row for event in sample["resample_event_ids"] for row in lookup.get(str(event), ())]
        if len(sampled_rows) < 20:
            raise RefitFailure("fewer than 20 sampled training views")
        remaining = None if max_seconds is None else max(0.001, max_seconds - (time.monotonic() - started))
        model, _ = _model_for_config(
            sampled_rows, context, model_config,
            _max_iterations(protocol, model_config), remaining,
            schema=schema, focal=True,
        )
        if not model.converged:
            raise RefitFailure("model did not converge")
        return _score_count_mapping(
            fixed_counts, model, schema, context, archetype_id, field,
            str(model_config.get("group_scope", "pooled")),
        )

    draws = int(protocol.get("h4", {}).get("bootstrap_draws", 200))
    seed = int(protocol.get("estimators", {}).get("fm_seeds", [1312])[0])
    X, y, nuisance, groups, _ = _design(original_rows, context, training=True)
    training_weights = _view_weights(original_rows, focal=True)
    model_key = _model_cache_identity(
        X, y, nuisance, groups, training_weights, schema, model_config,
        benchmark=benchmark, role="h4_fixed_list",
    )
    interval_key = hashlib.sha256(json.dumps({
        "model_key": model_key,
        "counts": dict(counts),
        "field": dict(field),
        "archetype_id": archetype_id,
    }, default=_json_default, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return fixed_list_interval(
        context, counts, fit_and_score=fit_and_score, draws=draws, seed=seed,
        width_target=float(protocol.get("h4", {}).get("mean_interval_width_target", 0.1)),
        checkpoint_path=checkpoint_path,
        context_key=interval_key,
        max_seconds=max_seconds,
    )


def _rank_lists(
    rows: Sequence[Mapping[str, Any]],
    nuisance_logits: np.ndarray,
    weights: np.ndarray,
    score_list: Any,
    *,
    deadline: float | None = None,
) -> dict[str, list[Any]]:
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        list_id = str(row.get("player") or row.get("deck_id") or "unknown")
        groups[(str(row.get("event") or "unknown"), list_id)].append(index)
    data = {key: [] for key in ("model", "baseline", "proxy", "matches", "events", "archetypes")}
    for (event, list_id), indices in groups.items():
        if deadline is not None and time.monotonic() >= deadline:
            raise FitBudgetExceeded("evaluation budget exhausted while scoring observed lists")
        local_weights = weights[indices]
        total = float(local_weights.sum())
        if total <= 0:
            continue
        row = rows[indices[0]]
        model_score, baseline_score = score_list(dict(row.get("counts", {})), row)
        data["model"].append(float(model_score))
        data["baseline"].append(float(baseline_score))
        data["proxy"].append(float(np.average(
            np.asarray([float(rows[i].get("won", 0.0)) for i in indices]) - expit(nuisance_logits[indices]),
            weights=local_weights,
        )))
        data["matches"].append(len({tuple(rows[i].get("pairing_key", ())) for i in indices}))
        data["events"].append(str(row.get("date_block_id") or _event_block({}, row)))
        data["archetypes"].append(str(row.get("archetype_id") or "unknown"))
    return data


def _weighted_h1_summary(
    gains: np.ndarray,
    event_ids: Sequence[Any],
    archetype_ids: Sequence[Any],
    weights: np.ndarray,
) -> dict[str, Any]:
    events = np.asarray(event_ids, dtype=object)
    archetypes = np.asarray(archetype_ids, dtype=object)
    def mean_by(labels: np.ndarray) -> float:
        values = [
            float(np.average(gains[labels == label], weights=weights[labels == label]))
            for label in np.unique(labels)
            if weights[labels == label].sum() > 0
        ]
        return float(np.mean(values)) if values else math.nan

    return {
        "pairing_weighted": float(np.average(gains, weights=weights)),
        "event_balanced": mean_by(events),
        "archetype_balanced": mean_by(archetypes),
        "event_count": int(np.unique(events).size),
        "archetype_count": int(np.unique(archetypes).size),
        "pairing_count": int(len(gains)),
        "view_weighting": "0.5 per pooled orientation; focal weights renormalised within underlying pairing",
    }


def _ranking_gain_interval(
    ranking_data: Mapping[str, Sequence[Any]],
    *,
    min_matches: int,
    seed: int,
    draws: int,
    deadline: float | None = None,
) -> dict[str, Any]:
    events = np.asarray(ranking_data["events"], dtype=object)
    clusters = np.unique(events)
    model = ranking_agreement(
        ranking_data["model"], ranking_data["proxy"], ranking_data["matches"],
        events, ranking_data["archetypes"], min_matches=min_matches,
    )
    baseline = ranking_agreement(
        ranking_data["baseline"], ranking_data["proxy"], ranking_data["matches"],
        events, ranking_data["archetypes"], min_matches=min_matches,
    )
    if model["weighted_agreement"] is None or baseline["weighted_agreement"] is None:
        return {"status": "inconclusive_sparse_support", "estimate": None, "interval": None,
                "interaction_agreement": model, "additive_agreement": baseline,
                "cluster_count": int(len(clusters))}
    estimate = float(model["weighted_agreement"] - baseline["weighted_agreement"])
    result = {"status": "inconclusive", "estimate": estimate, "interval": None,
              "interaction_agreement": model, "additive_agreement": baseline,
              "cluster_count": int(len(clusters)), "resampling_unit": "utc_date_block"}
    if len(clusters) < 20:
        result["warning"] = "fewer_than_20_independent_clusters"
        return result
    by_event = {event: np.flatnonzero(events == event) for event in clusters}
    rng = np.random.default_rng(seed)
    estimates = []
    exhausted = False
    for _ in range(draws):
        if deadline is not None and time.monotonic() >= deadline:
            exhausted = True
            break
        selected = rng.choice(clusters, size=len(clusters), replace=True)
        fields = {key: [] for key in ranking_data}
        for replicate, event in enumerate(selected):
            for key, values in ranking_data.items():
                fields[key].extend(values[index] for index in by_event[event])
            # Duplicate sampled blocks receive distinct IDs in this bootstrap draw.
            start = len(fields["events"]) - len(by_event[event])
            fields["events"][start:] = [f"bootstrap-{replicate}" for _ in by_event[event]]
        model_draw = ranking_agreement(
            fields["model"], fields["proxy"], fields["matches"], fields["events"],
            fields["archetypes"], min_matches=min_matches,
        )
        base_draw = ranking_agreement(
            fields["baseline"], fields["proxy"], fields["matches"], fields["events"],
            fields["archetypes"], min_matches=min_matches,
        )
        if model_draw["weighted_agreement"] is not None and base_draw["weighted_agreement"] is not None:
            estimates.append(model_draw["weighted_agreement"] - base_draw["weighted_agreement"])
    if exhausted:
        result["status"] = "incomplete"
        result["valid_draws"] = len(estimates)
        result["failed_draws"] = draws - len(estimates)
        result["reason"] = "evaluation_budget_exhausted_during_ranking_bootstrap"
        return result
    if len(estimates) < 0.8 * draws:
        result["status"] = "inconclusive"
        result["valid_draws"] = len(estimates)
        result["failed_draws"] = draws - len(estimates)
        return result
    interval = tuple(float(value) for value in np.quantile(estimates, [0.025, 0.975]))
    result["interval"] = interval
    result["status"] = "accepted" if interval[0] > 0 else "rejected" if interval[1] <= 0 else "inconclusive"
    result["valid_draws"] = len(estimates)
    return result


def _predictive_coverage(
    interval: Mapping[str, Any] | None,
    block_ids: Sequence[str],
    logits: Sequence[float],
    outcomes: Sequence[float],
    *,
    folds: int = 5,
) -> dict[str, Any]:
    """H4 coverage is not estimated from the fixed-list interval in this run.

    The model-conditional interval covers the fixed list's expected win rate, not a held-out pairing's
    outcome, and the simulator's injected truth is not the observed list's mean. Rather than report a
    number that answers neither question, the result is inconclusive with the measured row count.
    """
    bounds = None if not interval else interval.get("interval")
    reason = "no_fixed_list_interval" if not bounds else "coverage_not_estimated_for_this_interval"
    return {
        "status": "inconclusive",
        "reason": reason,
        "held_out_rows": len(outcomes),
        "held_out_blocks": len({str(block) for block in block_ids}),
        "latent_mean_coverage": None,
        "future_aggregate_coverage": None,
    }


def _evaluate_benchmark(
    context: dict[str, Any],
    test_views: Sequence[Mapping[str, Any]],
    validation_result: Mapping[str, Any],
    protocol: Mapping[str, Any],
    benchmark: str,
    *,
    archetype_id: str,
    field_override: Mapping[str, float] | None,
    max_seconds: float | None,
    output_path: Path,
) -> dict[str, Any]:
    benchmark_started = time.monotonic()
    deadline = None if max_seconds is None else benchmark_started + max_seconds
    benchmark_result = validation_result.get("benchmarks", {}).get(benchmark)
    if not benchmark_result or benchmark_result.get("status") != "complete":
        return {"status": "inconclusive", "reason": "no_complete_validation_selection"}
    train_rows = _target_rows(context.get("eligible_views", ()), None if benchmark == "pooled" else benchmark)
    measured_rows = _target_rows(test_views, None if benchmark == "pooled" else benchmark)
    if len(train_rows) < 20 or len(measured_rows) < 20:
        return {
            "status": "inconclusive",
            "reason": "fewer_than_20_training_or_test_views",
            "training_views": len(train_rows),
            "test_views": len(measured_rows),
        }
    selected = benchmark_result.get("selected_models", {})
    ref_config = selected.get("primary_additive")
    validation_interaction_config = selected.get("best_interaction")
    if not ref_config or not validation_interaction_config:
        return {"status": "inconclusive", "reason": "validation_did_not_select_both_models"}
    # Match the selected additive model's prior strength at the final refit
    # sample size. Validation stores the numeric penalty for its own training
    # sample; carrying it forward unchanged would weaken or strengthen the FM
    # main effects relative to the selected additive comparator.
    interaction_config = dict(validation_interaction_config)
    final_training_weights = _view_weights(train_rows, focal=benchmark != "pooled")
    interaction_config["main_effect_penalty"] = _effective_penalty(ref_config, final_training_weights)
    fitted: dict[str, tuple[FittedModel, Mapping[str, Any], Mapping[str, Any]]] = {}
    cache_dir = output_path.with_name(f"{output_path.stem}-model-cache")
    cache_usage: dict[str, bool] = {}
    for role, config in (("primary_additive", ref_config), ("best_interaction", interaction_config)):
        try:
            model, schema, cache_hit = _evaluation_model(
                train_rows, context, config,
                benchmark=benchmark,
                role=role,
                cache_dir=cache_dir,
                max_iterations=_max_iterations(protocol, config),
                max_seconds=_remaining(deadline),
            )
        except FitBudgetExceeded:
            return {
                "status": "incomplete",
                "reason": "fit_budget_exceeded",
                "completed_model_roles": sorted(fitted),
                "model_cache_dir": str(cache_dir),
                "test_outcomes_read": False,
            }
        cache_usage[role] = cache_hit
        if not model.converged:
            return {
                "status": "inconclusive",
                "reason": "frozen_model_refit_did_not_converge",
                "failed_model_role": role,
                "completed_model_roles": sorted(fitted),
                "converged": False,
                "model_cache_dir": str(cache_dir),
                "model_cache_hits": cache_usage,
                "test_outcomes_read": False,
            }
        test_X, test_y, test_nuisance, test_groups, diagnostics = _design(
            measured_rows, context, training=False, schema=schema
        )
        predicted_groups = _groups(str(config.get("group_scope", "pooled")), test_groups)
        logits = model.predict_logit(test_X, nuisance=test_nuisance, groups=predicted_groups)
        weights = _view_weights(measured_rows, focal=benchmark != "pooled")
        loss = _loss(logits, test_y, weights)
        fitted[role] = (model, schema, {"logits": logits, "loss": loss, "weights": weights, "diagnostics": diagnostics})
    if deadline is not None and time.monotonic() >= deadline:
        return {
            "status": "incomplete",
            "reason": "evaluation_budget_exhausted_after_model_refits",
            "completed_model_roles": sorted(fitted),
            "model_cache_dir": str(cache_dir),
            "model_cache_hits": cache_usage,
            "test_outcomes_read": False,
        }
    ref_model, ref_schema, ref_test = fitted["primary_additive"]
    int_model, int_schema, int_test = fitted["best_interaction"]
    ref_logit = np.asarray(ref_test["logits"], dtype=float)
    int_logit = np.asarray(int_test["logits"], dtype=float)
    test_y = np.asarray([float(row.get("won", 0.0)) for row in measured_rows])
    paired_gain = np.logaddexp(0.0, ref_logit) - test_y * ref_logit
    paired_gain -= np.logaddexp(0.0, int_logit) - test_y * int_logit
    weights = _view_weights(measured_rows, focal=benchmark != "pooled")
    pair_rows: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    for index, row in enumerate(measured_rows):
        pair_rows[tuple(row.get("pairing_key", ()))].append(index)
    pair_diffs: list[float] = []
    pair_events: list[str] = []
    pair_players1: list[str] = []
    pair_players2: list[str] = []
    for key, indices in sorted(pair_rows.items(), key=lambda item: repr(item[0])):
        local_weights = weights[indices]
        local_total = float(local_weights.sum())
        if local_total <= 0:
            continue
        pair_diffs.append(float(np.average(paired_gain[indices], weights=local_weights)))
        row = measured_rows[indices[0]]
        pair_events.append(str(row.get("date_block_id") or _event_block(context, row)))
        pair_players1.append(str(row.get("player1") or "unknown"))
        pair_players2.append(str(row.get("player2") or "unknown"))
    h1 = decide_h1(
        pair_diffs, pair_events, seed=int(protocol.get("inference", {}).get("seed", 1312)),
        draws=int(protocol.get("inference", {}).get("h1_bootstrap_draws", 2000)),
        min_gain=float(protocol.get("h1", {}).get("minimum_gain_nats_per_pairing", 0.0001)),
    ) if pair_diffs else {"status": "inconclusive", "estimate": None, "interval": None, "cluster_count": 0}
    block_by_view = [str(row.get("date_block_id") or _event_block(context, row)) for row in measured_rows]
    summary_weights = _unit_pair_view_weights(measured_rows, weights)
    balanced = _weighted_h1_summary(
        paired_gain, block_by_view,
        [str(row.get("archetype_id") or "unknown") for row in measured_rows], summary_weights,
    ) if len(paired_gain) else None
    balanced_groups = balanced_h1_summary(
        paired_gain, block_by_view,
        [str(row.get("archetype_id") or "unknown") for row in measured_rows],
        sample_weight=summary_weights,
    ) if len(paired_gain) else None
    player_sensitivity = player_aware_interval(
        pair_diffs, pair_events, pair_players1, pair_players2,
        seed=int(protocol.get("inference", {}).get("seed", 1312)),
        draws=int(protocol.get("inference", {}).get("h1_bootstrap_draws", 2000)),
    ) if pair_diffs and len(set(pair_events)) > 1 else {"status": "inconclusive", "interval": None}
    field = dict(context.get("field", {}) if field_override is None else field_override)
    if field:
        total = float(sum(float(value) for value in field.values()))
        if not np.isfinite(total) or total <= 0 or any(float(value) < 0 for value in field.values()):
            raise ValueError("field override must have finite non-negative weights and positive mass")
        field = {str(key): float(value) / total for key, value in field.items()}
    nuisance_config = {"kind": "nuisance", "group_scope": "pooled", "penalty": 0.0}
    try:
        nuisance_model, _ = _model_for_config(
            train_rows, context, nuisance_config, _max_iterations(protocol, nuisance_config), _remaining(deadline),
            focal=benchmark != "pooled",
        )
    except FitBudgetExceeded:
        return {
            "status": "incomplete",
            "reason": "fit_budget_exceeded",
            "failed_model_role": "nuisance",
            "completed_model_roles": sorted(fitted),
            "model_cache_dir": str(cache_dir),
            "model_cache_hits": cache_usage,
            "test_outcomes_read": True,
        }
    if not nuisance_model.converged:
        return {
            "status": "inconclusive",
            "reason": "frozen_model_refit_did_not_converge",
            "failed_model_role": "nuisance",
            "completed_model_roles": sorted(fitted),
            "converged": False,
            "model_cache_dir": str(cache_dir),
            "model_cache_hits": cache_usage,
            "test_outcomes_read": True,
        }
    nuisance_test_logits = nuisance_model.predict_logit(
        test_X, nuisance=test_nuisance, groups=_groups("pooled", test_groups)
    )
    def score_list(counts: Mapping[str, int], row: Mapping[str, Any]) -> tuple[float, float]:
        list_archetype = str(row.get("archetype_id") or archetype_id)
        model_score = _score_count_mapping(
            counts, int_model, int_schema, context, list_archetype, field,
            str(interaction_config.get("group_scope", "pooled")),
        )
        baseline_score = _score_count_mapping(
            counts, ref_model, ref_schema, context, list_archetype, field,
            str(ref_config.get("group_scope", "pooled")),
        )
        return model_score, baseline_score

    try:
        ranking_data = _rank_lists(
            measured_rows, nuisance_test_logits, weights, score_list, deadline=deadline
        ) if field else {
            key: [] for key in ("model", "baseline", "proxy", "matches", "events", "archetypes")
        }
    except FitBudgetExceeded:
        return {
            "status": "incomplete",
            "reason": "evaluation_budget_exhausted_during_ranking_scores",
            "model_cache_dir": str(cache_dir),
            "model_cache_hits": cache_usage,
            "test_outcomes_read": True,
        }
    ranking = _ranking_gain_interval(
        ranking_data,
        min_matches=int(protocol.get("h1", {}).get("ranking_minimum_decided_results_per_list", 5)),
        seed=int(protocol.get("inference", {}).get("seed", 1312)),
        draws=int(protocol.get("inference", {}).get("h1_bootstrap_draws", 2000)),
        deadline=deadline,
    ) if ranking_data["model"] else {"status": "inconclusive_sparse_support", "estimate": None, "interval": None}
    if h1.get("status") == "accepted" and ranking.get("status") == "accepted":
        h1_combined_status = "accepted"
    elif h1.get("status") == "rejected" or ranking.get("status") == "rejected":
        h1_combined_status = "rejected"
    else:
        h1_combined_status = "inconclusive"
    test_pairing_count = len(pair_rows)
    per_pair_path = output_path.with_name(f"{output_path.stem}-{benchmark}-pair-predictions.json")
    predictions = []
    for index, row in enumerate(measured_rows):
        key = tuple(row.get("pairing_key", ()))
        digest = hashlib.sha256(json.dumps(key, default=_json_default, separators=(",", ":")).encode()).hexdigest()
        predictions.append({
            "pairing_sha256": digest,
            "event_id": str(row.get("event") or "unknown"),
            "date_block_id": str(row.get("date_block_id") or _event_block(context, row)),
            "side": int(row.get("side", 1)),
            "weight": float(weights[index]),
            "won": float(test_y[index]),
            "primary_additive_logit": float(ref_logit[index]),
            "interaction_logit": float(int_logit[index]),
        })
    _atomic_json(per_pair_path, predictions)
    catalogue = [row for row in context.get("catalogue", ()) if archetype_id in row.get("archetype_ids", ())]
    selections: dict[str, Any] = {}
    if field:
        try:
            for role, model, schema, config in (
                ("primary_additive", ref_model, ref_schema, ref_config),
                ("best_interaction", int_model, int_schema, interaction_config),
            ):
                def scoring(counts: Mapping[str, int], _model=model, _schema=schema,
                            _scope=str(config.get("group_scope", "pooled"))) -> float:
                    if deadline is not None and time.monotonic() >= deadline:
                        raise FitBudgetExceeded("evaluation budget exhausted while scoring catalogue")
                    return _score_count_mapping(
                        counts, _model, _schema, context,
                        archetype_id if benchmark != "pooled" else str(context.get("target_archetype_id", archetype_id)),
                        field, _scope,
                    )

                catalogue_result = select_catalogue(catalogue, score=scoring)
                vote = _vote_list(context, archetype_id)
                vote_result = (
                    {"status": "selected", "counts": vote, "score": scoring(vote), "scope": "consensus_vote"}
                    if vote else {"status": "inconclusive", "counts": None, "score": None, "scope": "consensus_vote"}
                )
                selections[role] = {
                    "exact_catalogue": catalogue_result,
                    "vote": vote_result,
                    "field": field,
                    "field_source": "override" if field_override is not None else "training_context",
                    "skill_gap": 0.0,
                }
        except FitBudgetExceeded:
            return {
                "status": "incomplete",
                "reason": "evaluation_budget_exhausted_during_catalogue_selection",
                "model_cache_dir": str(cache_dir),
                "model_cache_hits": cache_usage,
                "partial_recommendations": selections,
                "test_outcomes_read": True,
            }
    else:
        selections["status"] = "inconclusive"
        selections["reason"] = "training_field_empty"

    # The target and logged actions are real observed-list IDs. The corpus has
    # no cross-fitted mu or logging propensities, so policy_value owns the
    # explicit not-identified result instead of receiving invented arrays.
    action_signatures: dict[tuple[tuple[str, int], ...], int] = {}
    def action_id(counts: Mapping[str, int]) -> int:
        signature = tuple(sorted((str(card), int(copies)) for card, copies in counts.items() if int(copies) > 0))
        if signature not in action_signatures:
            action_signatures[signature] = len(action_signatures)
        return action_signatures[signature]

    logged_actions = np.asarray([action_id(row.get("counts", {})) for row in measured_rows], dtype=int)
    target_counts = (
        selections.get("best_interaction", {}).get("exact_catalogue", {}).get("counts")
        if isinstance(selections.get("best_interaction"), Mapping) else None
    )
    if target_counts:
        target_action = action_id(target_counts)
        target_actions = np.full(len(measured_rows), target_action, dtype=int)
    else:
        target_action = None
        target_actions = logged_actions.copy()
    policy = policy_value(
        test_y,
        logged_actions,
        target_actions,
        None,
        None,
        event_ids=[row.get("date_block_id") or _event_block(context, row) for row in measured_rows],
    )
    policy["target_policy_action_id"] = target_action
    policy["target_policy_exact_list"] = target_counts
    policy["observed_action_count"] = len(action_signatures)
    focal_counts = None
    if benchmark != "pooled":
        focal_entry = selections.get("best_interaction", {}).get("exact_catalogue", {})
        focal_counts = focal_entry.get("counts")
    interval = None
    if focal_counts:
        uq_context = dict(context)
        # Resample exactly the training population used for this benchmark. The
        # focal result must not pull other archetype rows into its refits.
        uq_context["eligible_views"] = train_rows
        uq_context["training_event_ids"] = sorted({str(row.get("event")) for row in train_rows})
        interval = _fixed_interval(
            uq_context, focal_counts, interaction_config, field, archetype_id, int_schema,
            protocol,
            max_seconds=_remaining(deadline),
            checkpoint_path=output_path.with_name(f"{output_path.stem}-{benchmark}-h4-refit-checkpoint.json"),
            benchmark=benchmark,
        )
    substage_statuses = [
        h1.get("status"), ranking.get("status"),
        interval.get("status") if interval is not None else "inconclusive",
    ]
    return {
        "status": (
            "complete"
            if ref_model.converged and int_model.converged and "incomplete" not in substage_statuses
            else "incomplete"
        ),
        "benchmark": benchmark,
        "training_views": len(train_rows),
        "test_views": len(measured_rows),
        "test_pairings": test_pairing_count,
        "test_outcomes_read": True,
        "primary_additive": {"config": dict(ref_config), "test_log_loss": float(ref_test["loss"]), "converged": bool(ref_model.converged)},
        "best_interaction": {
            "config": dict(interaction_config),
            "validation_config": dict(validation_interaction_config),
            "test_log_loss": float(int_test["loss"]),
            "converged": bool(int_model.converged),
        },
        "h1_predictive": h1,
        "h1_combined_status": h1_combined_status,
        "h1_balanced": balanced,
        "h1_balanced_unweighted_diagnostic": balanced_groups,
        "h1_ranking": ranking,
        "player_dependence_sensitivity": player_sensitivity,
        "h4_predictive_coverage": _predictive_coverage(
            interval,
            [_event_block(context, row) for row in measured_rows],
            ref_logit,
            test_y,
        ),
        "h3_policy_value": {**policy, "selection_scores_are_not_policy_value": True},
        "recommendations": selections,
        "h4_fixed_list_interval": interval,
        "model_cache_dir": str(cache_dir),
        "model_cache_hits": cache_usage,
        "unknown_test_categories": ref_test["diagnostics"]["unknown_categories"],
        "pair_predictions_file": str(per_pair_path),
    }


def evaluate_experiment(
    context: dict[str, Any],
    test_views: Sequence[Mapping[str, Any]],
    selection: dict[str, Any],
    protocol: dict[str, Any],
    *,
    archetype_id: str,
    field_override: Mapping[str, float] | None,
    output_path: str | Path,
    max_seconds: int | None,
) -> dict[str, Any]:
    """Refit frozen validation choices and evaluate test outcomes once."""
    target = Path(output_path)
    maximum = None if max_seconds is None else float(max_seconds)
    started = time.monotonic()
    if selection.get("status") != "complete":
        return {"status": "inconclusive", "reason": "validation_selection_is_not_complete", "test_outcomes_read": False}
    if not test_views:
        return {"status": "inconclusive", "reason": "test_split_has_no_eligible_pairing_views", "test_outcomes_read": False}
    results = {}
    benchmark_names = {"pooled": None, archetype_id: None}
    for benchmark in _benchmark_order(benchmark_names, archetype_id, protocol):
        results[benchmark] = _evaluate_benchmark(
            context, test_views, selection, protocol, benchmark,
            archetype_id=archetype_id, field_override=field_override,
            max_seconds=None if maximum is None else max(0.001, maximum - (time.monotonic() - started)),
            output_path=target,
        )
    statuses = [result.get("status") for result in results.values()]
    accounted_statuses = {"complete", "inconclusive", "not_identified"}
    if not statuses or any(value not in accounted_statuses for value in statuses):
        status = "incomplete"
    elif all(value == "complete" for value in statuses):
        status = "complete"
    else:
        status = "inconclusive"
    output = {
        "status": status,
        "benchmarks": results,
        "test_outcomes_read": True,
        "test_use": "one locked evaluation under the saved validation choices",
        "policy_value_identification": "not_identified_without exact-action propensities and cross-fitted nuisance predictions",
    }
    _atomic_json(target.with_name("evaluation-experiment-result.json"), output)
    return output
