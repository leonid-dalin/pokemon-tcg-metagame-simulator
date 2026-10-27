"""Resumable outcome injection on the observed pairing design."""

from __future__ import annotations

import json
import hashlib
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse
from scipy.special import expit

from .estimators import FitBudgetExceeded, fit_model
from .power import interaction_information


def _object_vector(values: list[Any]) -> np.ndarray:
    result = np.empty(len(values), dtype=object)
    result[:] = values
    return result


def _normalise_pair_weights(pairing_ids: np.ndarray, weights: np.ndarray) -> np.ndarray:
    keys = [tuple(item) if isinstance(item, (list, np.ndarray)) else item for item in pairing_ids.tolist()]
    totals: dict[Any, float] = defaultdict(float)
    for key, weight in zip(keys, weights):
        totals[key] += float(weight)
    return np.asarray([float(weight) / totals[key] if totals[key] > 0 else 0.0 for key, weight in zip(keys, weights)])


def _pair_arrays(context: dict[str, Any]) -> dict[str, Any]:
    """Build aligned view arrays from explicit arrays or a corpus context."""
    if all(key in context for key in ("X", "nuisance", "groups", "pairing_ids", "event_ids", "player_ids")):
        arrays = {
            key: (
                sparse.csr_matrix(context[key], dtype=float)
                if key in {"X", "nuisance"} and sparse.issparse(context[key])
                else np.asarray(context[key])
            )
            for key in ("X", "nuisance", "groups", "pairing_ids", "event_ids", "player_ids")
        }
        n_rows = arrays["X"].shape[0] if sparse.issparse(arrays["X"]) else len(arrays["X"])
        arrays["sample_weight"] = np.asarray(context.get("sample_weight", np.ones(n_rows)), dtype=float)
        arrays["side"] = np.asarray(context.get("side", np.ones(n_rows)), dtype=float)
        arrays["base_y"] = np.asarray(context.get("y", np.zeros(n_rows)), dtype=float)
        arrays["pair_base"] = np.asarray(context.get("pair_base", np.zeros(n_rows)), dtype=float)
        arrays["opponent_ids"] = np.asarray(context.get("opponent_ids", np.full(n_rows, "unknown")), dtype=object)
        arrays["block_ids"] = np.asarray(context.get("block_ids", context.get("event_ids")), dtype=object)
        arrays["list_ids"] = np.asarray(context.get("list_ids", [("synthetic-row", i) for i in range(n_rows)]), dtype=object)
        if arrays["list_ids"].ndim > 1:
            arrays["list_ids"] = _object_vector([tuple(row) for row in arrays["list_ids"].tolist()])
        arrays["synthetic_design"] = bool(context.get("synthetic_design", True))
        arrays["feature_pair"] = context.get("feature_pair")
        arrays["feature_names"] = list(context.get("feature_names", [f"feature-{i}" for i in range(arrays["X"].shape[1])]))
        raw_sides = set(arrays["side"].tolist())
        if raw_sides <= {1, 2, "1", "2"}:
            arrays["side"] = np.asarray([-1.0 if side in (2, "2") else 1.0 for side in arrays["side"]])
        arrays["sample_weight"] = _normalise_pair_weights(arrays["pairing_ids"], arrays["sample_weight"])
        return arrays

    views = context.get("eligible_views", context.get("pairing_views"))
    slots = context.get("slot_names")
    pairings = context.get("eligible_pairings")
    if views is None or slots is None or pairings is None:
        raise ValueError("context must contain aligned arrays or eligible_views, slot_names and eligible_pairings")
    slot_index = {tuple(slot): i for i, slot in enumerate(slots)}
    row_indices, column_indices = [], []
    for i, view in enumerate(views):
        for card, count in view["counts"].items():
            for copy_number in range(1, int(count) + 1):
                column = slot_index.get((card, copy_number))
                if column is not None:
                    row_indices.append(i)
                    column_indices.append(column)
    X = sparse.csr_matrix(
        (np.ones(len(row_indices)), (row_indices, column_indices)),
        shape=(len(views), len(slot_index)),
    )
    slot_by_index = sorted(slot_index, key=slot_index.get)
    pair_by_key = {tuple(row.get("pairing_key", row.get("key", ()))): row for row in pairings}
    nuisance_columns = [np.ones(len(views), dtype=float)]
    category_fields = ("opponent_archetype_id", "game", "format", "delivery_mode", "match_mode")
    for field in category_fields:
        values = [str(view.get(field) or "unknown") for view in views]
        levels = sorted(set(values))
        for level in levels[1:]:
            nuisance_columns.append(np.asarray([value == level for value in values], dtype=float))
    nuisance = np.column_stack(nuisance_columns)
    skill_gap = np.asarray([
        float(context.get("strength", {}).get((view.get("event"), view.get("player")), 0.0))
        - float(context.get("strength", {}).get((view.get("event"), view.get("opponent")), 0.0))
        for view in views
    ])
    nuisance = np.column_stack([nuisance[:, 0], skill_gap, nuisance[:, 1:]])
    base_y, side, pair_base = [], [], []
    for i, view in enumerate(views):
        key = tuple(view["pairing_key"])
        pairing = pair_by_key.get(key, {})
        p1, p2 = view.get("player1", pairing.get("player1")), view.get("player2", pairing.get("player2"))
        raw_side = view.get("side", -1 if p1 is not None and view.get("player") == p2 else 1)
        side.append(-1.0 if raw_side in (2, -1, "2", "-1") else 1.0)
        base_y.append(float(view.get("won", 0.0)))
        pair_base.append(0.0)
    def _block(view: dict[str, Any]) -> str:
        event_id = str(view.get("event", "unknown"))
        configured = context.get("event_block_ids", {}).get(event_id)
        date = view.get("date")
        if configured is not None:
            return str(configured)
        return f"utc-date:{str(date)[:10]}" if date else event_id

    pairing_ids = np.empty(len(views), dtype=object)
    pairing_ids[:] = [tuple(view["pairing_key"]) for view in views]
    sample_weight = _normalise_pair_weights(
        pairing_ids, np.asarray([float(view.get("weight", 0.5)) for view in views])
    )
    return {
        "X": X,
        "nuisance": nuisance,
        "groups": np.asarray([view.get("archetype_id", "unknown") for view in views], dtype=object),
        "pairing_ids": pairing_ids,
        "event_ids": np.asarray([view.get("event", "unknown") for view in views], dtype=object),
        "player_ids": np.asarray([view.get("player", "unknown") for view in views], dtype=object),
        "opponent_ids": np.asarray([view.get("opponent", "unknown") for view in views], dtype=object),
        "block_ids": np.asarray([_block(view) for view in views], dtype=object),
        "list_ids": _object_vector([(str(view.get("event", "unknown")), str(view.get("player", "unknown"))) for view in views]),
        "synthetic_design": False,
        "sample_weight": sample_weight,
        "side": np.asarray(side),
        "base_y": np.asarray(base_y),
        "pair_base": np.asarray(pair_base),
        "pairings": pair_by_key,
        "feature_pair": context.get("feature_pair"),
        "feature_names": [slot[0] for slot in slot_by_index],
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple):
        return list(value)
    return str(value)


def _save_checkpoint(path: Path, rows: list[dict[str, Any]], next_index: int, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps({"rows": rows, "next_index": next_index, "metadata": metadata}, default=_json_default, indent=2), encoding="utf-8")
    tmp.replace(path)


def _random_effects(arrays: dict[str, Any], rng: np.random.Generator) -> np.ndarray:
    events = arrays["event_ids"].tolist()
    players = arrays["player_ids"].tolist()
    opponents = arrays["opponent_ids"].tolist()
    event_effects = {event: float(rng.normal(0, 0.35)) for event in sorted(set(events), key=repr)}
    player_effects = {player: float(rng.normal(0, 0.25)) for player in sorted(set(players) | set(opponents), key=repr)}
    return np.asarray([
        side * event_effects[event] + player_effects[player] - player_effects[opponent]
        for event, player, opponent, side in zip(events, players, opponents, arrays["side"])
    ])


def _injected_effect(
    X: np.ndarray, effect: float, mode: str, feature_pair: tuple[int, int] | None = None
) -> np.ndarray:
    n_rows = X.shape[0] if sparse.issparse(X) else len(X)
    if effect == 0:
        return np.zeros(n_rows)
    if X.shape[1] < 2:
        return np.zeros(n_rows)
    if mode == "package":
        if feature_pair is None:
            return np.zeros(n_rows)
        left, right = feature_pair
        if sparse.issparse(X):
            return float(effect) * np.asarray(X[:, left].multiply(X[:, right]).toarray()).reshape(-1)
        return float(effect) * X[:, left] * X[:, right]
    if mode == "low_rank":
        # Two pooled latent dimensions induce a sparse, deterministic low-rank
        # interaction surface over the observed copy-slot design.
        index = np.arange(X.shape[1], dtype=float)
        factors = np.column_stack([np.sin(index + 0.7), np.cos(0.5 * index + 0.2)]) / np.sqrt(30.0)
        latent = X @ factors
        squared = X.multiply(X) if sparse.issparse(X) else X * X
        return float(effect) * 0.5 * np.sum(latent * latent - squared @ (factors * factors), axis=1)
    raise ValueError("mode must be 'package' or 'low_rank'")


def _interaction_contrast(arrays: dict[str, Any], effect: float, mode: str) -> np.ndarray:
    utility = _injected_effect(arrays["X"], effect, mode, arrays.get("feature_pair"))
    contrast = utility.copy()
    pair_ids = [tuple(item) if isinstance(item, (list, np.ndarray)) else item for item in arrays["pairing_ids"].tolist()]
    rows_by_pair: dict[Any, list[int]] = defaultdict(list)
    for row, key in enumerate(pair_ids):
        rows_by_pair[key].append(row)
    for rows in rows_by_pair.values():
        if len(rows) == 2:
            first, second = rows
            contrast[first] = utility[first] - utility[second]
            contrast[second] = -contrast[first]
    return contrast


def _screen_columns(
    X, train_mask: np.ndarray, list_ids: np.ndarray, *, minimum_lists: int
) -> np.ndarray:
    """Screen copy slots by distinct training event/player lists."""
    train_rows = np.flatnonzero(train_mask)
    list_index: dict[Any, int] = {}
    exposures: list[set[int]] = []
    for row_index in train_rows:
        list_id = list_ids[row_index]
        key = tuple(list_id) if isinstance(list_id, (list, np.ndarray)) else list_id
        if key not in list_index:
            list_index[key] = len(exposures)
            exposures.append(set())
        target = exposures[list_index[key]]
        if sparse.issparse(X):
            target.update(X.getrow(row_index).indices.tolist())
        else:
            target.update(np.flatnonzero(X[row_index]).tolist())
    counts = np.zeros(X.shape[1], dtype=int)
    for columns in exposures:
        if columns:
            counts[list(columns)] += 1
    total_lists = len(exposures)
    return np.flatnonzero((counts >= minimum_lists) & (total_lists - counts >= minimum_lists))


def _select_supported_package(
    X,
    nuisance,
    weights: np.ndarray,
    event_ids: np.ndarray,
    feature_names: list[str],
    *,
    max_candidates: int = 128,
) -> tuple[tuple[int, int] | None, dict[str, Any]]:
    """Choose a distinct-card pair by training-only residual information."""
    X = sparse.csr_matrix(X) if sparse.issparse(X) else np.asarray(X, dtype=float)
    z = sparse.csr_matrix(nuisance, dtype=float)
    weighted_main = sparse.csr_matrix(X)
    design = sparse.hstack([z, weighted_main], format="csr")
    precision = 0.25 * np.asarray(weights, dtype=float)
    gram = (design.T @ design.multiply(precision[:, None])).toarray()
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    largest = max(0.0, float(eigenvalues[-1])) if eigenvalues.size else 0.0
    tolerance = largest * max(design.shape) * np.finfo(float).eps
    keep = eigenvalues > tolerance
    rank = int(np.sum(keep))
    inverse = (eigenvectors[:, keep] / eigenvalues[keep]) @ eigenvectors[:, keep].T if rank else np.zeros_like(gram)
    if sparse.issparse(X):
        cooccur = (X.T @ X).tocoo()
        candidates = sorted(
            ((int(i), int(j), float(value)) for i, j, value in zip(cooccur.row, cooccur.col, cooccur.data)
             if i < j and value > 0 and feature_names[i] != feature_names[j]),
            key=lambda item: (-item[2], item[0], item[1]),
        )[:max_candidates]
    else:
        cooccur = X.T @ X
        left, right = np.triu_indices(X.shape[1], k=1)
        candidates = sorted(
            ((int(i), int(j), float(cooccur[i, j])) for i, j in zip(left, right)
             if cooccur[i, j] > 0 and feature_names[i] != feature_names[j]),
            key=lambda item: (-item[2], item[0], item[1]),
        )[:max_candidates]
    best_pair, best_information = None, -1.0
    best_exposure = 0
    best_exposure_rows = np.empty(0, dtype=int)
    for left, right, _ in candidates:
        exposure = X[:, left].multiply(X[:, right]) if sparse.issparse(X) else X[:, left] * X[:, right]
        if sparse.issparse(exposure):
            exposure = exposure.tocsr()
            exposure_rows = exposure.nonzero()[0]
            exposure_count = int(len(exposure_rows))
            cross = (design.T @ exposure.multiply(precision[:, None])).toarray().reshape(-1)
            variance = float(np.asarray(exposure.power(2).T @ precision).reshape(-1)[0])
        else:
            exposure_count = int(np.count_nonzero(exposure))
            exposure_rows = np.flatnonzero(exposure)
            cross = np.asarray(design.T @ (precision * exposure)).reshape(-1)
            variance = float(np.dot(precision, exposure * exposure))
        information = max(0.0, variance - float(cross @ inverse @ cross))
        if information > max(1.0, variance) * 1e-10 and information > best_information:
            best_pair, best_information, best_exposure = (left, right), information, exposure_count
            best_exposure_rows = exposure_rows
    if best_pair is None:
        return None, {"status": "not_identified", "information": 0.0, "design_rank": rank,
                      "candidate_count": len(candidates), "aliases": "all candidate pairs aliased or unsupported"}
    selected_exposure = (
        X[:, best_pair[0]].multiply(X[:, best_pair[1]])
        if sparse.issparse(X) else X[:, best_pair[0]] * X[:, best_pair[1]]
    )
    information = interaction_information(
        design, selected_exposure, probabilities=np.full(X.shape[0], 0.5),
        sample_weight=weights, cluster_ids=event_ids,
    )
    return best_pair, {
        "status": information["status"],
        "feature_names": [feature_names[best_pair[0]], feature_names[best_pair[1]]],
        "feature_indices": list(best_pair),
        "information": information["information"],
        "design_rank": information["rank"],
        "candidate_count": len(candidates),
        "exposure_count": information["exposure_count"],
        "cluster_count": information["cluster_count"],
        "mde80_local_approximation": information["mde80"],
    }


def _pair_outcomes(arrays: dict[str, Any], logits: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    pair_ids = [tuple(item) if isinstance(item, (list, np.ndarray)) else item for item in arrays["pairing_ids"].tolist()]
    side = arrays["side"]
    result = np.zeros(len(logits), dtype=float)
    rows_by_pair: dict[Any, list[int]] = defaultdict(list)
    for row, key in enumerate(pair_ids):
        rows_by_pair[key].append(row)
    for rows_list in rows_by_pair.values():
        rows = np.asarray(rows_list, dtype=int)
        focal = int(rows[np.argmax(side[rows])])
        draw = float(rng.binomial(1, expit(logits[focal])))
        for row in rows:
            result[row] = draw if side[row] == side[focal] else 1.0 - draw
    return result


def _split_ids(
    pair_ids: np.ndarray, event_ids: np.ndarray, block_ids: np.ndarray, rng: np.random.Generator,
    fractions: tuple[float, float, float] = (0.6, 0.2, 0.2),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    unique = list(dict.fromkeys(tuple(item) if isinstance(item, (list, np.ndarray)) else item for item in pair_ids.tolist()))
    if len(unique) < 10:
        raise ValueError("at least ten underlying pairings are required for train/selection/measurement splits")
    pair_event: dict[Any, Any] = {}
    pair_block: dict[Any, Any] = {}
    for pair, event, block in zip(pair_ids.tolist(), event_ids.tolist(), block_ids.tolist()):
        key = tuple(pair) if isinstance(pair, (list, np.ndarray)) else pair
        pair_event.setdefault(key, event)
        pair_block.setdefault(key, block)
    real_blocks = all(str(block).startswith(("utc-date:", "event-block:")) for block in pair_block.values())
    if real_blocks:
        blocks = sorted(set(pair_block.values()), key=lambda item: str(item))
        if len(blocks) >= 5:
            n_train = max(1, int(fractions[0] * len(blocks)))
            n_select = max(1, int(fractions[1] * len(blocks)))
            train_blocks = set(blocks[:n_train])
            select_blocks = set(blocks[n_train : n_train + n_select])
            train_ids = [pair for pair in unique if pair_block[pair] in train_blocks]
            select_ids = [pair for pair in unique if pair_block[pair] in select_blocks]
            measure_ids = [pair for pair in unique if pair_block[pair] not in train_blocks | select_blocks]
            return _object_vector(train_ids), _object_vector(select_ids), _object_vector(measure_ids), "event_block" if any(str(b).startswith("event-block:") for b in blocks) else "utc_date"
    event_levels = sorted(set(pair_event.values()), key=repr)
    if len(event_levels) >= 5:
        shuffled_events = [event_levels[i] for i in rng.permutation(len(event_levels))]
        n_train = max(1, int(fractions[0] * len(event_levels)))
        n_select = max(1, int(fractions[1] * len(event_levels)))
        train_events = set(shuffled_events[:n_train])
        select_events = set(shuffled_events[n_train : n_train + n_select])
        train_ids = [pair for pair in unique if pair_event[pair] in train_events]
        select_ids = [pair for pair in unique if pair_event[pair] in select_events]
        measure_ids = [pair for pair in unique if pair_event[pair] not in train_events | select_events]
        return _object_vector(train_ids), _object_vector(select_ids), _object_vector(measure_ids), "event_random"
    shuffled = np.empty(len(unique), dtype=object)
    shuffled[:] = unique
    shuffled = shuffled[rng.permutation(len(unique))]
    n_train = max(1, int(fractions[0] * len(unique)))
    n_select = max(1, int(fractions[1] * len(unique)))
    return shuffled[:n_train], shuffled[n_train : n_train + n_select], shuffled[n_train + n_select :], "pairing_fallback"


def _log_loss(model, arrays: dict[str, Any], mask: np.ndarray, y: np.ndarray) -> float:
    logits = model.predict_logit(arrays["X"][mask], nuisance=arrays["nuisance"][mask], groups=arrays["groups"][mask])
    loss = np.logaddexp(0.0, logits) - y[mask] * logits
    weights = arrays["sample_weight"][mask]
    return float(np.average(loss, weights=weights))


def _loss_vector(model, arrays: dict[str, Any], mask: np.ndarray, y: np.ndarray) -> np.ndarray:
    logits = model.predict_logit(arrays["X"][mask], nuisance=arrays["nuisance"][mask], groups=arrays["groups"][mask])
    return np.logaddexp(0.0, logits) - y[mask] * logits


def _block_gain(differences: np.ndarray, blocks: np.ndarray, weights: np.ndarray, seed: int) -> dict[str, Any]:
    values = np.asarray(differences, dtype=float)
    labels = np.asarray(blocks).reshape(-1)
    sample_weights = np.asarray(weights, dtype=float)
    levels = sorted(set(labels.tolist()), key=repr)
    block_means = []
    block_weights = []
    for level in levels:
        selected = labels == level
        total = float(sample_weights[selected].sum())
        if total > 0:
            block_means.append(float(np.average(values[selected], weights=sample_weights[selected])))
            block_weights.append(total)
    if not block_means:
        return {"estimate": None, "interval95": None, "status": "inconclusive", "block_count": 0}
    means = np.asarray(block_means)
    mass = np.asarray(block_weights)
    estimate = float(np.average(means, weights=mass))
    if len(means) < 20:
        return {"estimate": estimate, "interval95": None, "status": "inconclusive", "block_count": len(means)}
    rng = np.random.default_rng(seed)
    draws = np.empty(500)
    for i in range(len(draws)):
        sampled = rng.integers(0, len(means), len(means))
        draws[i] = np.average(means[sampled], weights=mass[sampled])
    interval = [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]
    return {"estimate": estimate, "interval95": interval, "status": "accepted" if interval[0] > 0.0001 else "not_accepted", "block_count": len(means)}


def run_simulation(
    context: dict[str, Any],
    replicates: int = 500,
    *,
    effect: float = 0.0,
    mode: str = "package",
    base_seed: int = 1312,
    checkpoint: str | Path | None = None,
    max_seconds: float | None = None,
    start_index: int = 0,
    rank_grid: tuple[int, ...] = (2, 4, 8),
    interaction_penalties: tuple[float, ...] = (1e-5, 1e-4, 1e-3),
    seed_grid: tuple[int, ...] = (1312, 1313, 1314),
    max_iter: int = 5000,
    split_fractions: tuple[float, float, float] = (0.6, 0.2, 0.2),
) -> dict[str, Any]:
    """Run or resume an observed-design interaction power simulation.

    Each seed creates one outcome per underlying pairing. Mirrored views reuse
    that draw with opposite labels. Pair IDs are disjoint across training,
    hyperparameter selection and measurement. The JSON checkpoint is atomically
    replaced after every completed replicate.
    """
    if replicates < 0 or start_index < 0 or effect < 0:
        raise ValueError("replicates, start_index and effect must be nonnegative")
    arrays = _pair_arrays(context)
    n_rows = arrays["X"].shape[0] if sparse.issparse(arrays["X"]) else len(arrays["X"])
    if n_rows != len(arrays["pairing_ids"]):
        raise ValueError("context arrays are not aligned")
    signature = hashlib.sha256()
    for key in ("pairing_ids", "event_ids", "player_ids", "opponent_ids", "list_ids", "groups", "block_ids", "nuisance", "sample_weight", "side"):
        value = arrays[key]
        if sparse.issparse(value):
            signature.update(str(value.shape).encode("ascii"))
            signature.update(value.data.tobytes())
            signature.update(value.indices.tobytes())
            signature.update(value.indptr.tobytes())
        else:
            encoded = np.asarray(value)
            signature.update(str(encoded.shape).encode("ascii"))
            if encoded.dtype.kind in {"O", "U", "S"}:
                signature.update(json.dumps(encoded.tolist(), default=_json_default, sort_keys=True).encode("utf-8"))
            else:
                signature.update(encoded.tobytes())
    if sparse.issparse(arrays["X"]):
        signature.update(arrays["X"].data.tobytes())
        signature.update(arrays["X"].indices.tobytes())
        signature.update(arrays["X"].indptr.tobytes())
    else:
        signature.update(str(arrays["X"].shape).encode("ascii"))
        signature.update(np.asarray(arrays["X"]).tobytes())
    signature.update(json.dumps(arrays.get("feature_pair"), default=_json_default).encode("utf-8"))
    signature.update(json.dumps(arrays.get("feature_names"), default=_json_default).encode("utf-8"))
    metadata = {
        "checkpoint_version": 2,
        "context_sha256": signature.hexdigest(),
        "target_replicates": int(replicates),
        "base_seed": int(base_seed),
        "effect": float(effect),
        "mode": mode,
        "rank_grid": list(rank_grid),
        "interaction_penalties": list(interaction_penalties),
        "seed_grid": list(seed_grid),
        "additive_penalties": list(interaction_penalties),
        "max_iter": int(max_iter),
        "screen_min_unique_lists": 2 if arrays["synthetic_design"] else 30,
        "screen_unit": "synthetic_row" if arrays["synthetic_design"] else "event_player_list",
        "h1_min_gain": 0.0001,
        "h1_min_blocks": 20,
        "h1_bootstrap_draws": 500,
        "split_fractions": list(split_fractions),
        "split_event_minimum": 5,
        "minimum_pairings": 10,
        "event_logit_sd": 0.35,
        "player_logit_sd": 0.25,
        "strength_gap_coefficient": 0.35,
        "selection_metric": "weighted_log_loss",
        "fit_failure_policy": "any_nonconverged_grid_fit_marks_replicate_inconclusive",
        "package_pair_selection": "training_only_top128_cooccurrences_residualized_on_nuisance_and_global_main",
        "dgp_version": 2,
    }
    checkpoint_path = Path(checkpoint) if checkpoint is not None else None
    rows: list[dict[str, Any]] = []
    next_index = int(start_index)
    if checkpoint_path is not None and checkpoint_path.exists():
        saved = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if saved.get("metadata") != metadata:
            raise ValueError("checkpoint context or simulation settings do not match this run")
        rows = list(saved.get("rows", []))
        next_index = max(next_index, int(saved.get("next_index", 0)))
    elif checkpoint_path is not None:
        _save_checkpoint(checkpoint_path, rows, next_index, metadata)
    batch_start_index = next_index
    # ``replicates`` is the total target. A time-limited rerun resumes at the
    # saved index and stops at the same target.
    last_index = int(replicates)
    if next_index > last_index:
        raise ValueError("checkpoint already exceeds the requested replicate target")
    started = time.monotonic()

    def fit_with_remaining_budget(**kwargs):
        if max_seconds is not None:
            remaining = max_seconds - (time.monotonic() - started)
            if remaining <= 0:
                raise FitBudgetExceeded("simulation time budget is exhausted")
            kwargs["max_seconds"] = remaining
        return fit_model(**kwargs)

    for rep in range(next_index, last_index):
        if max_seconds is not None and time.monotonic() - started >= max_seconds:
            break
        seed = int(base_seed + rep)
        rng = np.random.default_rng(seed)
        split_rng = np.random.default_rng(seed + 0x5EED)
        train_ids, select_ids, measure_ids, split_unit = _split_ids(
            arrays["pairing_ids"], arrays["event_ids"], arrays["block_ids"], split_rng, fractions=split_fractions
        )
        pair_ids = [tuple(item) if isinstance(item, (list, np.ndarray)) else item for item in arrays["pairing_ids"].tolist()]
        train_set, selection_set, measurement_set = set(train_ids.tolist()), set(select_ids.tolist()), set(measure_ids.tolist())
        train = np.asarray([item in train_set for item in pair_ids])
        selection = np.asarray([item in selection_set for item in pair_ids])
        measurement = np.asarray([item in measurement_set for item in pair_ids])
        screen_minimum = 2 if arrays["synthetic_design"] else 30
        feature_columns = _screen_columns(arrays["X"], train, arrays["list_ids"], minimum_lists=screen_minimum)
        work_arrays = dict(arrays)
        work_arrays["X"] = arrays["X"][:, feature_columns]
        work_arrays["feature_pair"] = None
        work_arrays["feature_names"] = [arrays["feature_names"][column] for column in feature_columns]
        if sparse.issparse(arrays["X"]):
            work_arrays["X"] = work_arrays["X"].tocsr()
        if mode == "package":
            package_pair, package_info = _select_supported_package(
                work_arrays["X"][train],
                work_arrays["nuisance"][train],
                work_arrays["sample_weight"][train],
                work_arrays["event_ids"][train],
                work_arrays["feature_names"],
            )
        else:
            package_pair, package_info = None, {"status": "not_applicable"}
        work_arrays["feature_pair"] = package_pair
        injection_arrays = dict(arrays)
        injection_arrays["feature_pair"] = tuple(feature_columns[index] for index in package_pair) if package_pair else None
        event_player = _random_effects(arrays, rng)
        beta_nuisance = np.zeros(arrays["nuisance"].shape[1])
        if len(beta_nuisance) > 1:
            beta_nuisance[1] = 0.35
        fixed = arrays["nuisance"] @ beta_nuisance
        injected = _interaction_contrast(injection_arrays, effect, mode)
        pair_y = _pair_outcomes(arrays, fixed + event_player + injected, rng)
        additive_candidates = []
        candidates = []
        failures = 0
        failed_fits = []
        budget_exhausted = False
        for penalty in interaction_penalties:
            try:
                baseline = fit_with_remaining_budget(
                    X=work_arrays["X"][train], y=pair_y[train], nuisance=work_arrays["nuisance"][train],
                    groups=work_arrays["groups"][train], kind="additive", penalty=penalty,
                    sample_weight=work_arrays["sample_weight"][train], seed=seed, max_iter=max_iter,
                )
            except FitBudgetExceeded:
                budget_exhausted = True
                break
            failures += int(not baseline.converged)
            if not baseline.converged:
                failed_fits.append({"kind": "additive", "penalty": penalty, "status": baseline.status,
                                    "message": baseline.message, "iterations": baseline.iterations,
                                    "gradient_norm": baseline.gradient_norm, "parameter_count": baseline.parameter_count})
            if baseline.converged:
                additive_candidates.append((_log_loss(baseline, work_arrays, selection, pair_y), penalty, baseline))
        if additive_candidates:
            additive_candidates.sort(key=lambda row: (row[0], row[1]))
            additive_selection_loss, additive_penalty, additive = additive_candidates[0]
        if not budget_exhausted and additive_candidates:
            for rank in rank_grid:
                for penalty in interaction_penalties:
                    for fm_seed in seed_grid:
                        try:
                            model = fit_with_remaining_budget(
                                X=work_arrays["X"][train], y=pair_y[train], nuisance=work_arrays["nuisance"][train],
                                groups=work_arrays["groups"][train], kind="fm", rank=rank, seed=fm_seed,
                                penalty=penalty, main_effect_penalty=additive_penalty,
                                sample_weight=work_arrays["sample_weight"][train], max_iter=max_iter,
                            )
                        except FitBudgetExceeded:
                            budget_exhausted = True
                            break
                        failures += int(not model.converged)
                        if not model.converged:
                            failed_fits.append({"kind": "fm", "rank": rank, "penalty": penalty, "seed": fm_seed,
                                                "status": model.status, "message": model.message,
                                                "iterations": model.iterations, "gradient_norm": model.gradient_norm,
                                                "parameter_count": model.parameter_count})
                        if model.converged:
                            candidates.append((
                                _log_loss(model, work_arrays, selection, pair_y), rank, penalty, fm_seed, model,
                            ))
                if budget_exhausted:
                    break
        if budget_exhausted:
            if checkpoint_path is not None:
                _save_checkpoint(checkpoint_path, rows, rep, metadata)
            return {
                "status": "incomplete",
                "mode": mode,
                "effect": float(effect),
                "requested_replicates": int(replicates),
                "completed_replicates": int(sum(row.get("status") == "complete" for row in rows)),
                "failed_replicates": int(sum(row.get("status") == "fit_failure" for row in rows)),
                "not_identified_replicates": int(sum(row.get("status") == "not_identified" for row in rows)),
                "next_index": rep,
                "seconds_per_completed_replicate": (time.monotonic() - started) / max(1, rep - batch_start_index),
                "rows": rows,
            }
        if failures:
            rows.append({
                "replicate": rep,
                "seed": seed,
                "status": "fit_failure",
                "fit_failures": failures,
                "failed_fit_details": failed_fits,
                "converged_additive_candidates": len(additive_candidates),
                "converged_fm_candidates": len(candidates),
                "package_identification": package_info,
            })
        elif not additive_candidates or not candidates:
            rows.append({"replicate": rep, "seed": seed, "status": "fit_failure", "fit_failures": failures,
                         "failed_fit_details": failed_fits})
        elif effect > 0 and mode == "package" and package_pair is None:
            rows.append({
                "replicate": rep,
                "seed": seed,
                "status": "not_identified",
                "package_identification": package_info,
                "fit_failures": failures,
                "failed_fit_details": failed_fits,
            })
        else:
            candidates.sort(key=lambda row: (row[0], row[1], row[2], row[3]))
            selected_loss, chosen_rank, chosen_penalty, chosen_seed, chosen = candidates[0]
            additive_loss = _log_loss(additive, work_arrays, measurement, pair_y)
            fm_loss = _log_loss(chosen, work_arrays, measurement, pair_y)
            measured_difference = _loss_vector(additive, work_arrays, measurement, pair_y) - _loss_vector(chosen, work_arrays, measurement, pair_y)
            decision = _block_gain(
                measured_difference,
                work_arrays["block_ids"][measurement],
                work_arrays["sample_weight"][measurement],
                seed + 999,
            )
            interaction_estimate = None
            if mode == "package" and work_arrays.get("feature_pair"):
                left, right = work_arrays["feature_pair"]
                interaction_estimate = float(np.dot(chosen.factors[left], chosen.factors[right]))
            rows.append({
                "replicate": rep,
                "seed": seed,
                "status": "complete",
                "estimate": additive_loss - fm_loss,
                "additive_log_loss": additive_loss,
                "fm_log_loss": fm_loss,
                "additive_penalty": additive_penalty,
                "fm_main_effect_penalty": chosen.main_effect_penalty,
                "h1_decision": decision,
                "detected": decision["status"] == "accepted",
                "selection_gain_over_additive": additive_selection_loss - selected_loss,
                "true_interaction": float(effect) if mode == "package" and package_pair is not None else None,
                "estimated_interaction": interaction_estimate,
                "selection_log_loss": selected_loss,
                "selected_rank": chosen_rank,
                "selected_penalty": chosen_penalty,
                "selected_seed": chosen_seed,
                "fit_failures": failures,
                "failed_fit_details": failed_fits,
                "train_pairings": len(train_ids),
                "selection_pairings": len(select_ids),
                "measurement_pairings": len(measure_ids),
                "split_unit": split_unit,
                "train_events": len(set(arrays["event_ids"][train].tolist())),
                "selection_events": len(set(arrays["event_ids"][selection].tolist())),
                "measurement_events": len(set(arrays["event_ids"][measurement].tolist())),
                "screened_features": int(len(feature_columns)),
                "screen_min_unique_lists": screen_minimum,
                "package_identification": package_info,
            })
        next_index = rep + 1
        if checkpoint_path is not None:
            _save_checkpoint(checkpoint_path, rows, next_index, metadata)
    done = [row["estimate"] for row in rows if row.get("status") == "complete"]
    detected = sum(bool(row.get("detected")) for row in rows if row.get("status") == "complete")
    attempted_all = next_index >= last_index
    failed_count = int(sum(row.get("status") == "fit_failure" for row in rows))
    not_identified_count = int(sum(row.get("status") == "not_identified" for row in rows))
    status = (
        "inconclusive" if attempted_all and failed_count
        else "not_identified" if attempted_all and not_identified_count
        else "complete" if attempted_all
        else "incomplete"
    )
    if done:
        mean = float(np.mean(done))
        mcse = float(np.std(done, ddof=1) / np.sqrt(len(done))) if len(done) > 1 else None
        interval = [mean - 1.96 * mcse, mean + 1.96 * mcse] if mcse is not None else None
    else:
        mean, mcse, interval = None, None, None
    if done:
        count = len(done)
        rate = detected / count
        z = 1.959963984540054
        denom = 1.0 + z * z / count
        centre = (rate + z * z / (2 * count)) / denom
        margin = z * np.sqrt(rate * (1 - rate) / count + z * z / (4 * count * count)) / denom
        detection_interval = [float(max(0.0, centre - margin)), float(min(1.0, centre + margin))]
    else:
        rate, detection_interval = None, None
    return {
        "status": status,
        "mode": mode,
        "effect": float(effect),
        "requested_replicates": int(replicates),
        "completed_replicates": int(sum(row.get("status") == "complete" for row in rows)),
        "failed_replicates": failed_count,
        "not_identified_replicates": not_identified_count,
        "next_index": next_index,
        "mean_log_loss_gain": mean,
        "monte_carlo_se": mcse,
        "monte_carlo_interval95": interval,
        "detected_replicates": int(detected),
        "detection_rate": rate,
        "detection_rate_mc_interval95": detection_interval,
        "injection_status": "selected_per_replicate_from_training" if mode == "package" else "pooled_low_rank",
        "seconds_per_completed_replicate": (time.monotonic() - started) / max(1, next_index - batch_start_index),
        "rows": rows,
    }
