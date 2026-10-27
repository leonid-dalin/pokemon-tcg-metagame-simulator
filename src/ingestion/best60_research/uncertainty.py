"""Fixed-list model-conditional ranges with resumable refits."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .estimators import FitBudgetExceeded


class RefitFailure(Exception):
    """A declared numerical failure while refitting one bootstrap draw."""


def _missing_ids(values: np.ndarray) -> bool:
    for value in values.tolist():
        if value is None or value == "":
            return True
        if isinstance(value, (float, np.floating)) and np.isnan(value):
            return True
    return False


def _block_map(context: Mapping[str, Any], events: Sequence[Any]) -> dict[Any, Any]:
    for key in ("event_date_blocks", "date_block_by_event", "training_event_blocks", "event_block_ids"):
        candidate = context.get(key)
        if isinstance(candidate, Mapping):
            return {event: candidate.get(event, event) for event in events}
    blocks = context.get("date_blocks", context.get("event_blocks"))
    if isinstance(blocks, Sequence) and not isinstance(blocks, (str, bytes)):
        event_to_block: dict[Any, Any] = {}
        for index, block in enumerate(blocks):
            if isinstance(block, Mapping):
                block_id = block.get("id", block.get("date", index))
                for event in block.get("event_ids", block.get("events", ())):
                    event_to_block[event] = block_id
        if event_to_block:
            return {event: event_to_block.get(event, event) for event in events}
    return {event: event for event in events}


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (tuple, set)):
        return list(value)
    return str(value)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False, default=_json_default).encode("utf-8")


def _save_checkpoint(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                         allow_nan=False, default=_json_default) + "\n"
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _checkpoint_metadata(
    context: Mapping[str, Any], counts: Mapping[str, int], *, context_key: str | None,
    events: Sequence[Any], event_to_block: Mapping[Any, Any], draws: int, seed: int,
    width_target: float,
) -> dict[str, Any]:
    provenance = context.get("provenance", {})
    if context_key is None:
        context_key = hashlib.sha256(_canonical({
            "events": list(events),
            "event_blocks": [[str(event), str(event_to_block[event])] for event in events],
            "source_sha256": provenance.get("source_sha256") if isinstance(provenance, Mapping) else None,
            "cutoff": provenance.get("cutoff") if isinstance(provenance, Mapping) else context.get("cutoff"),
            "embargo_days": context.get("embargo_days"),
        })).hexdigest()
    metadata = {
        "schema_version": 1,
        "context_key": str(context_key),
        "fixed_counts": dict(counts),
        "events": list(events),
        "event_blocks": [[str(event), event_to_block[event]] for event in events],
        "draws": int(draws),
        "seed": int(seed),
        "width_target": float(width_target),
    }
    return {"metadata": metadata, "metadata_sha256": hashlib.sha256(_canonical(metadata)).hexdigest()}


def _unit_label(context: Mapping[str, Any]) -> str:
    """The label for the resampling unit the context actually carries."""
    ids = context.get("event_block_ids", {})
    if ids and all(str(value).startswith("event-block:") for value in ids.values()):
        return "atomic_event_block"
    return "atomic_date_block"


def fixed_list_interval(
    context: Mapping[str, Any],
    counts: Mapping[str, int],
    *,
    fit_and_score: Callable[[Mapping[str, Any], Mapping[str, int]], float],
    draws: int = 200,
    seed: int = 1312,
    width_target: float = 0.10,
    checkpoint_path: str | Path | None = None,
    context_key: str | None = None,
    max_seconds: float | None = None,
) -> dict[str, Any]:
    """Resample training date blocks and refit a frozen model for one fixed list.

    When a checkpoint path is supplied, every completed draw is persisted with
    the next draw index and RNG state. A fit budget timeout leaves the current
    draw pending so it can be retried with the same sampled blocks.
    """
    started = time.monotonic()
    if max_seconds is not None and (not np.isfinite(max_seconds) or max_seconds < 0):
        raise ValueError("max_seconds must be finite and non-negative")
    events = tuple(context.get("training_event_ids", ()))
    if draws < 1 or not events or not 0 < width_target <= 1:
        return {
            "status": "incomplete",
            "interval": None,
            "effective_draws": 0,
            "failed_draws": 0,
            "pending_draws": int(max(draws, 0)),
            "next_index": 0,
            "requested_draws": int(draws),
            "scope": "model_conditional",
            "target": "expected_decided_pairing_win_probability",
            "resampling_unit": _unit_label(context),
        }
    event_to_block = _block_map(context, events)
    block_events: dict[Any, list[Any]] = {}
    for event in events:
        block_events.setdefault(event_to_block[event], []).append(event)
    block_ids = tuple(sorted(block_events, key=str))
    if len(block_ids) < 2:
        return {
            "status": "inconclusive",
            "reason": "fewer_than_two_resampling_blocks",
            "interval": None,
            "effective_draws": 0,
            "failed_draws": 0,
            "pending_draws": 0,
            "next_index": 0,
            "requested_draws": int(draws),
            "block_count": len(block_ids),
            "scope": "model_conditional",
            "target": "expected_decided_pairing_win_probability",
            "resampling_unit": _unit_label(context),
        }
    selected_count = dict(counts)
    checkpoint_target = Path(checkpoint_path) if checkpoint_path is not None else None
    metadata = _checkpoint_metadata(
        context, selected_count, context_key=context_key, events=events,
        event_to_block=event_to_block, draws=draws, seed=seed, width_target=width_target,
    )
    rng = np.random.default_rng(seed)
    checkpoint: dict[str, Any] = {
        **metadata,
        "next_index": 0,
        "rng_state": copy.deepcopy(rng.bit_generator.state),
        "scores": [],
        "failures": [],
    }
    if checkpoint_target is not None and checkpoint_target.is_file():
        try:
            saved = json.loads(checkpoint_target.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot read fixed-list interval checkpoint: {exc}") from exc
        if saved.get("metadata") != metadata["metadata"] or saved.get("metadata_sha256") != metadata["metadata_sha256"]:
            raise ValueError("fixed-list interval checkpoint does not match context, counts or settings")
        checkpoint = saved
        next_index = int(checkpoint.get("next_index", -1))
        if next_index < 0 or next_index > draws:
            raise ValueError("fixed-list interval checkpoint has an invalid next draw index")
        completed = list(checkpoint.get("scores", ())) + list(checkpoint.get("failures", ()))
        completed_indices = sorted(int(item["draw_index"]) for item in completed)
        if completed_indices != list(range(next_index)):
            raise ValueError("fixed-list interval checkpoint draw records are inconsistent")
        rng.bit_generator.state = checkpoint["rng_state"]
    else:
        next_index = 0
        if checkpoint_target is not None:
            _save_checkpoint(checkpoint_target, checkpoint)

    values = [float(item["score"]) for item in checkpoint["scores"]]
    failure_messages = list(checkpoint["failures"])
    failure_reasons: dict[str, int] = {}
    for item in failure_messages:
        reason = str(item["reason"])
        failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
    incomplete_reason = None
    for draw_index in range(next_index, draws):
        if max_seconds is not None and time.monotonic() - started >= max_seconds:
            incomplete_reason = "stage time budget exhausted before the next draw"
            if checkpoint_target is not None:
                _save_checkpoint(checkpoint_target, checkpoint)
            break
        state_before_draw = copy.deepcopy(rng.bit_generator.state)
        draw_blocks = rng.choice(len(block_ids), size=len(block_ids), replace=True)
        sampled_blocks = [block_ids[int(index)] for index in draw_blocks]
        sampled_events = [event for block in sampled_blocks for event in block_events[block]]
        sample_context = dict(context)
        sample_context["resample_event_ids"] = tuple(sampled_events)
        sample_context["resample_block_ids"] = tuple(sampled_blocks)
        try:
            result = fit_and_score(sample_context, dict(selected_count))
            if isinstance(result, Mapping):
                result = result.get("score")
            score = float(result)
        except FitBudgetExceeded as exc:
            rng.bit_generator.state = state_before_draw
            incomplete_reason = str(exc) or "fit budget exhausted during the current draw"
            if checkpoint_target is not None:
                _save_checkpoint(checkpoint_target, checkpoint)
            break
        except RefitFailure as exc:
            reason = str(exc) or type(exc).__name__
            failure_messages.append({"draw_index": draw_index, "reason": reason})
            failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
        else:
            if not np.isfinite(score) or not 0 <= score <= 1:
                reason = "callback returned a non-finite score or a score outside [0, 1]"
                failure_messages.append({"draw_index": draw_index, "reason": reason})
                failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
            else:
                values.append(score)
                checkpoint["scores"].append({"draw_index": draw_index, "score": score})
        if failure_messages and failure_messages[-1].get("draw_index") == draw_index:
            checkpoint["failures"] = list(failure_messages)
        checkpoint["next_index"] = draw_index + 1
        checkpoint["rng_state"] = copy.deepcopy(rng.bit_generator.state)
        if checkpoint_target is not None:
            _save_checkpoint(checkpoint_target, checkpoint)

    next_index = int(checkpoint["next_index"])
    failed = len(failure_messages)
    pending = max(0, draws - next_index)
    interval = tuple(float(value) for value in np.quantile(values, [0.025, 0.975])) if values else None
    width = interval[1] - interval[0] if interval else None
    if pending or incomplete_reason:
        status = "incomplete"
    elif failed:
        status = "incomplete"
    elif len(set(values)) < 2 or width == 0:
        status = "inconclusive"
    else:
        status = "complete"
    result = {
        "status": status,
        "estimate": float(np.mean(values)) if values else None,
        "interval": interval,
        "effective_draws": len(values),
        "failed_draws": failed,
        "pending_draws": pending,
        "next_index": next_index,
        "failure_reasons": failure_reasons,
        "requested_draws": int(draws),
        "scope": "model_conditional",
        "target": "expected_decided_pairing_win_probability",
        "resampling_unit": (
            _unit_label(context)
            if any(block != event for event, block in event_to_block.items())
            else "event_cluster"
        ),
        "block_count": len(block_ids),
        "fixed_counts": selected_count,
        "interval_width": width,
        "interval_width_target": float(width_target),
        "width_target_met": bool(width is not None and width <= width_target),
        "assumptions": [
            "The selected list stays fixed across draws.",
            "The fitted estimator and hyperparameters are conditional on the declared selection procedure.",
            "This interval does not establish causal or post-selection coverage.",
            "The field stays fixed unless field estimation is part of the callback target.",
        ],
    }
    if incomplete_reason:
        result["incomplete_reason"] = incomplete_reason
    if checkpoint_target is not None:
        result["checkpoint_path"] = str(checkpoint_target)
        result["checkpoint_key"] = metadata["metadata_sha256"]
    return result
