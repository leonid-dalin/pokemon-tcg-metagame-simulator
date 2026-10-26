"""Chronological partitioning that keeps UTC date blocks intact."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Sequence


def _utc_date(value: Any) -> str | None:
    if value is None or not str(value).strip():
        return None
    try:
        parsed = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).date().isoformat()


def split_events(events: Sequence[dict[str, Any]], *, fractions: tuple[float, float, float] = (0.6, 0.2, 0.2)) -> dict[str, list[str]]:
    """Split events in chronological order, assigning every UTC date to one partition."""
    if len(fractions) != 3 or any(value <= 0 for value in fractions) or abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError("fractions must be three positive values summing to one")
    seen: set[str] = set()
    by_date: dict[str, list[str]] = {}
    for event in events:
        event_id = str(event.get("id") or "")
        block = _utc_date(event.get("date"))
        if not event_id or block is None:
            raise ValueError("every event must have an ID and a valid date")
        if event_id in seen:
            raise ValueError(f"duplicate event ID: {event_id}")
        seen.add(event_id)
        by_date.setdefault(block, []).append(event_id)
    blocks = [(date, sorted(by_date[date])) for date in sorted(by_date)]
    result = {"development": [], "validation": [], "test": []}
    names = tuple(result)
    total = len(events)
    cumulative = 0
    boundaries = (fractions[0], fractions[0] + fractions[1])
    for date, event_ids in blocks:
        midpoint = (cumulative + len(event_ids) / 2) / total if total else 0.0
        section = 0 if midpoint < boundaries[0] else 1 if midpoint < boundaries[1] else 2
        result[names[section]].extend(event_ids)
        cumulative += len(event_ids)
    return result
