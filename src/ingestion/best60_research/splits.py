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


def event_blocks(events: Sequence[dict[str, Any]], players_by_event: Mapping[str, set[str]]) -> list[tuple[str, list[str]]]:
    """Events that start on the same UTC day and share at least one player form one unit.

    Units are ordered by their earliest UTC date, then by their sorted event IDs.
    """
    parent: dict[str, str] = {str(event.get("id") or ""): str(event.get("id") or "") for event in events}

    def find(item: str) -> str:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    by_day: dict[str, list[str]] = {}
    for event in events:
        event_id = str(event.get("id") or "")
        block = _utc_date(event.get("date"))
        if not event_id or block is None:
            raise ValueError("every event must have an ID and a valid date")
        by_day.setdefault(block, []).append(event_id)
    for ids in by_day.values():
        owner: dict[str, str] = {}
        for event_id in ids:
            for player in sorted(players_by_event.get(event_id, set())):
                if player in owner:
                    first, second = find(event_id), find(owner[player])
                    if first != second:
                        parent[second] = first
                else:
                    owner[player] = event_id
    units: dict[str, list[str]] = {}
    for event_id in parent:
        units.setdefault(find(event_id), []).append(event_id)
    return sorted(
        ((_utc_date(next(event["date"] for event in events if str(event.get("id") or "") == root)) or "", sorted(ids))
         for root, ids in units.items()),
        key=lambda row: (row[0], row[1]),
    )


def split_events(
    events: Sequence[dict[str, Any]],
    *,
    fractions: tuple[float, float, float] = (0.6, 0.2, 0.2),
    unit: str = "date",
    players_by_event: Mapping[str, set[str]] | None = None,
) -> dict[str, list[str]]:
    """Split events in chronological order, assigning every unit to one partition.

    `unit` is `date` (one UTC day) or `event_block` (same day and at least one shared player).
    """
    if len(fractions) != 3 or any(value <= 0 for value in fractions) or abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError("fractions must be three positive values summing to one")
    if unit == "date":
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
    elif unit == "event_block":
        blocks = event_blocks(events, players_by_event or {})
    else:
        raise ValueError(f"unknown split unit: {unit}")
    result = {"development": [], "validation": [], "test": []}
    names = tuple(result)
    if unit == "date":
        total = len(events)
        cumulative = 0
        boundaries = (fractions[0], fractions[0] + fractions[1])
        for _, event_ids in blocks:
            midpoint = (cumulative + len(event_ids) / 2) / total if total else 0.0
            section = 0 if midpoint < boundaries[0] else 1 if midpoint < boundaries[1] else 2
            result[names[section]].extend(event_ids)
            cumulative += len(event_ids)
        return result
    total = len(blocks)
    boundaries = (fractions[0] * total, (fractions[0] + fractions[1]) * total)
    for index, (_, event_ids) in enumerate(blocks):
        section = 0 if index < boundaries[0] else 1 if index < boundaries[1] else 2
        result[names[section]].extend(event_ids)
    return result
