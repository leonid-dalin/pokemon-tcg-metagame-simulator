"""Exact observed-list selection for offline Best-60 research."""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Callable

import numpy as np

from src.ingestion.model import ACE_SPEC_CARDS, BASIC_ENERGY_NAMES, validate_recommendation


class ExpectedScoreFailure(Exception):
    """A declared model-scoring failure for one catalogue candidate."""


def field_scores(logits: np.ndarray, weights: Sequence[float]) -> np.ndarray:
    """Average list win probabilities over one normalised opponent field."""
    values = np.asarray(logits, dtype=float)
    field = np.asarray(weights, dtype=float)
    if values.ndim < 1 or values.shape[-1] != field.size:
        raise ValueError("the final logits axis must match the opponent field")
    if field.ndim != 1 or field.size == 0 or not np.isfinite(field).all() or (field < 0).any():
        raise ValueError("field weights must be a non-empty finite non-negative vector")
    total = float(field.sum())
    if total <= 0:
        raise ValueError("field weights must have positive total mass")
    if not np.isfinite(values).all():
        raise ValueError("logits must be finite")
    probabilities = np.empty_like(values)
    positive = values >= 0
    probabilities[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    probabilities[~positive] = exp_values / (1.0 + exp_values)
    return np.tensordot(probabilities, field / total, axes=([-1], [0]))


def _unpack_candidate(candidate: Any) -> tuple[dict[str, int], int, tuple[str, ...], Mapping[str, Any]]:
    """Read a count mapping or a catalogue row with count and provenance fields."""
    metadata: Mapping[str, Any] = candidate if isinstance(candidate, Mapping) else {}
    raw_counts = metadata.get("counts", candidate)
    if not isinstance(raw_counts, Mapping):
        raise ValueError("catalogue candidate must contain a count mapping")
    counts: dict[str, int] = {}
    for raw_name, raw_count in raw_counts.items():
        name = str(raw_name).strip()
        if not name:
            raise ValueError("card names must be non-empty")
        if isinstance(raw_count, bool) or int(raw_count) != raw_count:
            raise ValueError(f"copy count for {name} must be an integer")
        count = int(raw_count)
        if count < 0:
            raise ValueError(f"copy count for {name} cannot be negative")
        if count:
            counts[name] = count
    support = int(metadata.get("support", metadata.get("support_count", metadata.get("list_count", 1))))
    if support < 1:
        raise ValueError("exact catalogue candidates need at least one observed supporting list")
    sources = tuple(sorted(str(item) for item in metadata.get("source_ids", metadata.get("events", ()))))
    return counts, support, sources, metadata


def _candidate_rules(counts: Mapping[str, int], metadata: Mapping[str, Any]) -> dict[str, dict[str, bool]]:
    rules = {}
    for name in counts:
        if name in ACE_SPEC_CARDS:
            rules[name] = {"ace_spec": True}
        elif name in BASIC_ENERGY_NAMES:
            rules[name] = {"basic_energy": True}
        else:
            rules[name] = {}
    declared = metadata.get("card_rules")
    if isinstance(declared, Mapping):
        for name, value in declared.items():
            if name in rules and isinstance(value, Mapping):
                rules[name].update({str(k): bool(v) for k, v in value.items()})
    return rules


def select_catalogue(candidates: Sequence[Any], *, score: Callable[[Mapping[str, int]], float]) -> dict[str, Any]:
    """Return the deterministic score maximum over legal, exact observed lists."""
    rows: dict[tuple[tuple[str, int], ...], dict[str, Any]] = {}
    validation_limitations: set[str] = set()
    invalid = 0
    for candidate in candidates:
        try:
            counts, support, sources, metadata = _unpack_candidate(candidate)
            if not metadata.get("format"):
                validation_limitations.add(
                    "format-specific restrictions cannot be checked because format metadata is missing"
                )
            if not metadata.get("companions"):
                validation_limitations.add(
                    "companion and evolution requirements are checked only when candidate metadata declares them"
                )
            validate_recommendation(
                [{"card": name, "copies": count} for name, count in counts.items()],
                banned_cards=set(metadata.get("banned_cards", ())),
                card_rules=_candidate_rules(counts, metadata),
            )
            # Explicit companion requirements are expressed as {card: {companion: copies_per_copy}}.
            requirements = metadata.get("companions", {})
            if isinstance(requirements, Mapping):
                for card, needs in requirements.items():
                    if counts.get(str(card), 0) and isinstance(needs, Mapping):
                        for companion, per_copy in needs.items():
                            required = max(1, int(per_copy) * counts[str(card)])
                            if counts.get(str(companion), 0) < required:
                                raise ValueError("companion requirement is not met")
            signature = tuple(sorted(counts.items()))
            row = rows.setdefault(
                signature, {"counts": counts, "support": 0, "source_ids": set(), "metadata": metadata}
            )
            row["support"] += support
            row["source_ids"].update(sources)
        except (TypeError, ValueError, OverflowError):
            invalid += 1
    if not rows:
        return {
            "status": "invalid_catalogue" if invalid else "empty_catalogue",
            "counts": None,
            "score": None,
            "candidate_count": 0,
            "input_candidate_count": len(candidates),
            "invalid_candidate_count": invalid,
            "support": 0,
            "source_ids": [],
            "validation_limitations": sorted(validation_limitations),
        }

    scored: list[tuple[float, tuple[tuple[str, int], ...], dict[str, Any]]] = []
    invalid_scores = 0
    score_failure_messages: dict[str, int] = {}
    for signature, row in rows.items():
        try:
            value = float(score(dict(row["counts"])))
        except ExpectedScoreFailure as exc:
            invalid_scores += 1
            message = str(exc) or type(exc).__name__
            score_failure_messages[message] = score_failure_messages.get(message, 0) + 1
            continue
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            invalid_scores += 1
            continue
        scored.append((value, signature, row))
    if not scored:
        return {
            "status": "invalid_scores",
            "counts": None,
            "score": None,
            "candidate_count": len(rows),
            "input_candidate_count": len(candidates),
            "invalid_candidate_count": invalid,
            "invalid_score_count": invalid_scores,
            "score_failure_messages": score_failure_messages,
            "support": 0,
            "source_ids": [],
            "validation_limitations": sorted(validation_limitations),
        }
    # Highest score wins; lexical card/count order resolves exact ties reproducibly.
    best_value = max(item[0] for item in scored)
    tied = sorted((item for item in scored if item[0] == best_value), key=lambda x: x[1])
    value, _, row = tied[0]
    return {
        "status": "selected",
        "counts": dict(row["counts"]),
        "score": value,
        "candidate_count": len(rows),
        "input_candidate_count": len(candidates),
        "invalid_candidate_count": invalid,
        "invalid_score_count": invalid_scores,
        "score_failure_messages": score_failure_messages,
        "support": int(row["support"]),
        "source_ids": sorted(row["source_ids"]),
        "tie_count": len(tied),
        "scope": "exact_observed_catalogue",
        "validation_limitations": sorted(validation_limitations),
    }
