from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from .store import LimitlessStore


def _event_date(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _card_names(decklist: Any) -> set[str]:
    if not isinstance(decklist, dict):
        return set()
    return {
        str(card["name"])
        for group in decklist.values()
        if isinstance(group, list)
        for card in group
        if isinstance(card, dict) and card.get("name") is not None and str(card["name"]).strip()
    }


def inclusion_rates(
    store: LimitlessStore,
    archetype: str,
    window_days: int | float | None = None,
) -> dict[str, float]:
    if window_days is None:
        return store.card_inclusion(archetype)

    store.prepare_for_read()
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=window_days)
    event_ids = {
        str(row[0])
        for row in store.iter_events()
        if (event_date := _event_date(row[4])) is not None and cutoff <= event_date <= now
    }

    counts: dict[str, int] = {}
    total_decks = 0
    for event_id in event_ids:
        for decklist in store._decklists(archetype, event_id):
            if not isinstance(decklist, dict):
                continue
            names = _card_names(decklist)
            total_decks += 1
            for name in names:
                counts[name] = counts.get(name, 0) + 1

    if not total_decks:
        return {}
    return {name: count / total_decks for name, count in sorted(counts.items())}
