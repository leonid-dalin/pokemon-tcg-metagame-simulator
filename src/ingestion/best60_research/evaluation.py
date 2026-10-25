"""Independent predictive and supported-policy evaluation helpers."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def _missing_ids(values: np.ndarray) -> bool:
    for value in values.tolist():
        if value is None or value == "":
            return True
        if isinstance(value, (float, np.floating)) and np.isnan(value):
            return True
    return False


def _cluster_bootstrap_mean(values: np.ndarray, clusters: np.ndarray, *, seed: int, draws: int) -> tuple[float, float]:
    unique = np.unique(clusters)
    rng = np.random.default_rng(seed)
    by_cluster = [values[clusters == cluster] for cluster in unique]
    estimates = np.empty(draws, dtype=float)
    for i in range(draws):
        take = rng.integers(0, len(unique), len(unique))
        sample = np.concatenate([by_cluster[j] for j in take])
        estimates[i] = float(sample.mean())
    return tuple(float(value) for value in np.quantile(estimates, [0.025, 0.975]))


def decide_h1(
    differences: Sequence[float],
    event_ids: Sequence[Any],
    *,
    seed: int = 1312,
    draws: int = 2000,
    min_gain: float = 0.0001,
) -> dict[str, Any]:
    """Assess additive-minus-challenger log loss by resampling whole events."""
    values = np.asarray(differences, dtype=float)
    clusters = np.asarray(event_ids)
    if values.ndim != 1 or clusters.ndim != 1 or values.size != clusters.size or values.size == 0:
        raise ValueError("differences and event_ids must be non-empty vectors of equal length")
    if not np.isfinite(values).all() or draws < 1 or min_gain < 0:
        raise ValueError("differences, draws and min_gain must be valid")
    if _missing_ids(clusters):
        raise ValueError("event_ids must be present for every pairing")
    cluster_count = int(np.unique(clusters).size)
    estimate = float(values.mean())
    result: dict[str, Any] = {
        "estimate": estimate,
        "interval": None,
        "status": "insufficient_clusters",
        "cluster_count": cluster_count,
        "pairing_count": int(values.size),
        "target": "paired_natural_log_loss_advantage_nats_per_pairing",
        "scope": "predictive_h1",
        "minimum_gain": float(min_gain),
        "resampling_unit": "event_cluster",
    }
    if cluster_count < 2:
        return result
    interval = _cluster_bootstrap_mean(values, clusters, seed=seed, draws=draws)
    result["interval"] = interval
    if cluster_count < 20:
        result["status"] = "inconclusive"
        result["warning"] = "fewer_than_20_independent_clusters"
    else:
        if interval[0] > min_gain:
            result["status"] = "accepted"
        elif interval[1] <= min_gain:
            result["status"] = "rejected"
        else:
            result["status"] = "inconclusive"
    return result


def balanced_h1_summary(
    differences: Sequence[float],
    event_ids: Sequence[Any],
    archetype_ids: Sequence[Any],
    *,
    sample_weight: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Summarise view-level gains with equal weight per event and archetype."""
    values = np.asarray(differences, dtype=float)
    events = np.asarray(event_ids)
    archetypes = np.asarray(archetype_ids)
    if values.ndim != 1 or events.shape != values.shape or archetypes.shape != values.shape or values.size == 0:
        raise ValueError("differences, event_ids and archetype_ids must be non-empty matching vectors")
    if not np.isfinite(values).all():
        raise ValueError("differences must be finite")
    weights = np.ones(values.size, dtype=float) if sample_weight is None else np.asarray(sample_weight, dtype=float)
    if weights.shape != values.shape or not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("sample_weight must be finite, non-negative and match the view-level differences")
    weighted_group_means = lambda labels: [
        float(np.average(values[labels == group], weights=weights[labels == group]))
        for group in np.unique(labels)
        if weights[labels == group].sum() > 0
    ]
    event_means = weighted_group_means(events)
    archetype_means = weighted_group_means(archetypes)
    return {
        "pairing_weighted": float(np.average(values, weights=weights)),
        "event_balanced": float(np.mean(event_means)),
        "archetype_balanced": float(np.mean(archetype_means)),
        "event_count": len(event_means),
        "archetype_count": len(archetype_means),
        "view_count": int(values.size),
        "unit": "test_view; weights should sum to one within each underlying pairing",
    }


def player_aware_interval(
    values: Sequence[float],
    event_ids: Sequence[Any],
    player1_ids: Sequence[Any],
    player2_ids: Sequence[Any],
    *,
    date_block_ids: Sequence[Any] | None = None,
    seed: int = 1312,
    draws: int = 2000,
) -> dict[str, Any]:
    """Run a multiway pigeonhole bootstrap over date blocks and both players.

    Pass ``date_block_ids`` to keep events on the same UTC date together. When omitted,
    event IDs are treated as independent resampling blocks for backwards compatibility.
    """
    scores = np.asarray(values, dtype=float)
    events = np.asarray(event_ids)
    first = np.asarray(player1_ids)
    second = np.asarray(player2_ids)
    blocks = events if date_block_ids is None else np.asarray(date_block_ids)
    arrays_match = all(array.shape == scores.shape for array in (events, first, second, blocks))
    if scores.ndim != 1 or not arrays_match or not scores.size:
        raise ValueError("scores, event IDs and both player ID vectors must match")
    if not np.isfinite(scores).all() or draws < 1:
        raise ValueError("scores and draws must be valid")
    if any(_missing_ids(array) for array in (events, blocks, first, second)):
        raise ValueError("event, date-block and player IDs must be present for every row")
    unique_events = np.unique(events)
    unique_blocks = np.unique(blocks)
    unique_players = np.unique(np.concatenate([first, second]))
    if unique_blocks.size < 2 or unique_players.size < 2:
        return {
            "status": "inconclusive",
            "interval": None,
            "cluster_count": int(unique_blocks.size),
            "event_count": int(unique_events.size),
            "date_block_count": int(unique_blocks.size),
            "player_count": int(unique_players.size),
            "resampling_unit": (
                "date_block_and_both_player_ids" if date_block_ids is not None else "event_and_both_player_ids"
            ),
        }
    rng = np.random.default_rng(seed)
    estimates = np.empty(draws, dtype=float)
    for draw in range(draws):
        block_counts = rng.multinomial(len(unique_blocks), np.full(len(unique_blocks), 1 / len(unique_blocks)))
        player_counts = rng.multinomial(len(unique_players), np.full(len(unique_players), 1 / len(unique_players)))
        block_weight = dict(zip(unique_blocks.tolist(), block_counts.tolist()))
        player_weight = dict(zip(unique_players.tolist(), player_counts.tolist()))
        row_weights = np.asarray(
            [
                block_weight[block] * player_weight[p1] * player_weight[p2]
                for block, p1, p2 in zip(blocks, first, second)
            ],
            dtype=float,
        )
        if row_weights.sum() == 0:
            estimates[draw] = float(scores.mean())
        else:
            estimates[draw] = float(np.average(scores, weights=row_weights))
    return {
        "status": "measured",
        "estimate": float(scores.mean()),
        "interval": tuple(float(value) for value in np.quantile(estimates, [0.025, 0.975])),
        "draws": int(draws),
        "cluster_count": int(unique_blocks.size),
        "event_count": int(unique_events.size),
        "date_block_count": int(unique_blocks.size),
        "player_count": int(unique_players.size),
        "resampling_unit": (
            "date_block_and_both_player_ids" if date_block_ids is not None else "event_and_both_player_ids"
        ),
        "scope": "dependence_sensitivity",
        "assumptions": [
            "Date blocks and player identities receive independent multinomial bootstrap weights.",
            "Each pairing row receives the product of its date-block and both player weights.",
            "This sensitivity interval does not correct unmeasured confounding.",
        ],
    }


def ranking_agreement(
    model_scores: Sequence[float],
    residual_proxy: Sequence[float],
    match_counts: Sequence[int],
    event_ids: Sequence[Any],
    archetype_ids: Sequence[Any],
    *,
    min_matches: int = 5,
) -> dict[str, Any]:
    """Compare model and residual-proxy ordering within event/archetype strata."""
    scores = np.asarray(model_scores, dtype=float)
    proxy = np.asarray(residual_proxy, dtype=float)
    matches = np.asarray(match_counts, dtype=int)
    events = np.asarray(event_ids)
    archetypes = np.asarray(archetype_ids)
    if scores.ndim != 1 or any(array.shape != scores.shape for array in (proxy, matches, events, archetypes)):
        raise ValueError("ranking inputs must be matching vectors")
    if not np.isfinite(scores).all() or not np.isfinite(proxy).all() or (matches < 0).any() or min_matches < 1:
        raise ValueError("ranking scores, counts and minimum match count must be valid")
    eligible = matches >= min_matches
    concordant_weight = 0.0
    pair_weight = 0.0
    eligible_pairs = 0
    strata = sorted(set(zip(events[eligible].tolist(), archetypes[eligible].tolist())), key=str)
    for event, archetype in strata:
        indices = np.flatnonzero(eligible & (events == event) & (archetypes == archetype))
        for offset, i in enumerate(indices):
            for j in indices[offset + 1 :]:
                weight = float(min(matches[i], matches[j]))
                delta_model = scores[i] - scores[j]
                delta_truth = proxy[i] - proxy[j]
                agreement = 0.5 if delta_model == 0 or delta_truth == 0 else float(delta_model * delta_truth > 0)
                concordant_weight += weight * agreement
                pair_weight += weight
                eligible_pairs += 1
    return {
        "status": "measured" if eligible_pairs else "inconclusive_sparse_support",
        "weighted_agreement": concordant_weight / pair_weight if pair_weight else None,
        "eligible_pair_count": eligible_pairs,
        "eligible_pair_weight": pair_weight,
        "eligible_list_count": int(eligible.sum()),
        "minimum_matches_per_list": min_matches,
        "stratum": "event_and_archetype",
        "tie_weight": 0.5,
    }


def _policy_arrays(
    y: Sequence[float],
    actions: Sequence[int],
    target: np.ndarray,
    mu: np.ndarray,
    propensity: np.ndarray,
    event_ids: Sequence[Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    reward = np.asarray(y, dtype=float)
    logged_raw = np.asarray(actions)
    if logged_raw.ndim != 1 or logged_raw.dtype.kind not in {"i", "u"}:
        raise ValueError("logged actions must be integer action IDs")
    logged = logged_raw.astype(int, copy=False)
    target_raw = np.asarray(target)
    if target_raw.ndim == 1:
        if target_raw.dtype.kind not in {"i", "u"}:
            raise ValueError("one-dimensional target policies must contain integer action IDs")
        if target_raw.size == 0 or (target_raw < 0).any():
            raise ValueError("target policy action IDs must be non-negative")
        action_count = int(np.asarray(mu).shape[1]) if np.asarray(mu).ndim == 2 else 0
        if action_count == 0:
            raise ValueError("mu must provide the action count for an action-ID target")
        if (target_raw >= action_count).any():
            raise ValueError("target policy action is outside the action array")
        target_p = np.zeros((target_raw.size, action_count), dtype=float)
        target_p[np.arange(target_raw.size), target_raw.astype(int)] = 1.0
    else:
        target_p = np.asarray(target_raw, dtype=float)
    outcome = np.asarray(mu, dtype=float)
    logging_p = np.asarray(propensity, dtype=float)
    clusters = np.asarray(event_ids)
    if reward.ndim != 1 or logged.ndim != 1 or clusters.ndim != 1:
        raise ValueError("y, actions and event_ids must be vectors")
    if target_p.ndim != 2 or outcome.shape != target_p.shape or logging_p.shape != target_p.shape:
        raise ValueError("target, mu and propensity must have the same (rows, actions) shape")
    if reward.size != logged.size or logged.size != target_p.shape[0] or logged.size != clusters.size:
        raise ValueError("policy arrays must have the same row count")
    if _missing_ids(clusters):
        raise ValueError("event_ids must be present for every policy row")
    if logged.size == 0 or not np.isfinite(reward).all() or ((reward < 0) | (reward > 1)).any():
        raise ValueError("y must contain finite binary/probability rewards in [0, 1]")
    if ((logged < 0) | (logged >= target_p.shape[1])).any():
        raise ValueError("logged action is outside the action array")
    for name, array in (("target", target_p), ("mu", outcome), ("propensity", logging_p)):
        if not np.isfinite(array).all():
            raise ValueError(f"{name} values must be finite")
    if (target_p < 0).any() or (logging_p < 0).any() or (outcome < 0).any() or (outcome > 1).any():
        raise ValueError("policy probabilities must be non-negative and mu must lie in [0, 1]")
    if not np.allclose(target_p.sum(axis=1), 1.0, atol=1e-8):
        raise ValueError("each target policy row must sum to one")
    if not np.allclose(logging_p.sum(axis=1), 1.0, atol=1e-8):
        raise ValueError("each logging propensity row must sum to one")
    return reward, logged, target_p, outcome, logging_p, clusters


def _dr_scores(
    reward: np.ndarray,
    logged: np.ndarray,
    target: np.ndarray,
    mu: np.ndarray,
    propensity: np.ndarray,
    clip: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    rows = np.arange(len(reward))
    logged_target = target[rows, logged]
    logged_propensity = propensity[rows, logged]
    ratio = logged_target / logged_propensity
    used_ratio = ratio if clip is None else np.minimum(ratio, clip)
    baseline = np.sum(target * mu, axis=1)
    scores = baseline + used_ratio * (reward - mu[rows, logged])
    return scores, ratio


def policy_value(
    y: Sequence[float],
    actions: Sequence[int],
    target: np.ndarray,
    mu: np.ndarray | None,
    propensity: np.ndarray | None,
    *,
    event_ids: Sequence[Any],
) -> dict[str, Any]:
    """Estimate a contextual target-policy value from cross-fitted nuisance arrays."""
    if mu is None or propensity is None:
        missing = []
        if mu is None:
            missing.append("cross_fitted_outcome_model")
        if propensity is None:
            missing.append("exact_list_logging_propensities")
        return {
            "status": "not_identified",
            "estimate": None,
            "interval": None,
            "clipped_at_20_estimate": None,
            "target": "expected_decided_pairing_win_probability",
            "estimator": "cross_fitted_doubly_robust",
            "reason": "missing_required_policy_nuisance_arrays",
            "missing_inputs": missing,
            "row_count": len(y),
            "action_count": int(np.asarray(target).shape[-1]) if np.asarray(target).ndim > 1 else None,
        }
    reward, logged, target_p, outcome, logging_p, clusters = _policy_arrays(
        y, actions, target, mu, propensity, event_ids
    )
    target_supported = target_p > 0
    unsupported = target_supported & (logging_p < 0.01)
    low_propensity_rows = int(np.any(unsupported, axis=1).sum())
    logged_probability = logging_p[np.arange(len(logged)), logged]
    logged_target = target_p[np.arange(len(logged)), logged]
    zero_or_invalid_overlap = np.any(target_supported & (logging_p <= 0), axis=1)
    ratio = np.divide(logged_target, logged_probability, out=np.zeros_like(logged_target), where=logged_probability > 0)
    ess = float(ratio.sum() ** 2 / np.square(ratio).sum()) if np.square(ratio).sum() else 0.0
    cluster_count = int(np.unique(clusters).size)
    result: dict[str, Any] = {
        "status": "not_identified",
        "estimate": None,
        "interval": None,
        "clipped_at_20_estimate": None,
        "importance_weight_ess": ess,
        "cluster_count": cluster_count,
        "low_target_propensity_rows": low_propensity_rows,
        "target": "expected_decided_pairing_win_probability",
        "estimator": "cross_fitted_doubly_robust",
        "weighting": "unclipped_primary",
    }
    if zero_or_invalid_overlap.any() or low_propensity_rows:
        return result
    if cluster_count < 50 or ess < 200:
        result["reason"] = "insufficient_operational_overlap"
        result["minimum_event_clusters"] = 50
        result["minimum_importance_weight_ess"] = 200
        return result
    scores, _ = _dr_scores(reward, logged, target_p, outcome, logging_p, None)
    clipped_scores, _ = _dr_scores(reward, logged, target_p, outcome, logging_p, 20.0)
    result["estimate"] = float(scores.mean())
    result["clipped_at_20_estimate"] = float(clipped_scores.mean())
    result["interval"] = _cluster_bootstrap_mean(scores, clusters, seed=1312, draws=2000)
    result["status"] = "identified"
    result["clipping_note"] = "clip-at-20 changes the estimand weighting and can introduce bias"
    return result
