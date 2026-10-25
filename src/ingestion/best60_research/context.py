"""Training-only features, fields and observed legal-list catalogue."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any

from src.core.config import (
    BDIF_BEST60_MIN_SLOT_LISTS,
    BDIF_BEST60_PREREQUISITE_SHARE,
    BDIF_BEST60_STRENGTH_PRIOR_GAMES,
)
from src.ingestion.best60 import ACE_SPEC_CARDS, BASIC_ENERGY_NAMES

STRENGTH_PRIOR_GAMES = float(BDIF_BEST60_STRENGTH_PRIOR_GAMES)


def _datetime(value: Any) -> datetime | None:
    if value is None or not str(value).strip():
        return None
    try:
        result = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if result.tzinfo is None:
        result = result.replace(tzinfo=timezone.utc)
    return result.astimezone(timezone.utc)


def _utc_midnight(value: datetime) -> datetime:
    return datetime(value.year, value.month, value.day, tzinfo=timezone.utc)


def _legal(counts: dict[str, int]) -> bool:
    if not counts or any(not name or not isinstance(copies, int) or copies <= 0 for name, copies in counts.items()):
        return False
    if sum(counts.values()) != 60:
        return False
    if any(copies > 4 for name, copies in counts.items() if name not in BASIC_ENERGY_NAMES):
        return False
    return sum(counts.get(name, 0) for name in ACE_SPEC_CARDS) <= 1


def _prerequisites(lists: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """Match ``best60.prerequisites`` with sparse pair counts instead of a dense matrix."""
    present_counts: Counter[str] = Counter()
    cooccurrence: Counter[tuple[str, str]] = Counter()
    per_copy: Counter[tuple[str, str]] = Counter()
    for row in lists:
        counts = row["counts"]
        cards = sorted(counts)
        for card in cards:
            present_counts[card] += 1
            for companion in cards:
                if companion == card:
                    continue
                cooccurrence[(card, companion)] += 1
                if counts[companion] >= counts[card]:
                    per_copy[(card, companion)] += 1
    minimum = BDIF_BEST60_MIN_SLOT_LISTS
    share = BDIF_BEST60_PREREQUISITE_SHARE
    ratios: dict[tuple[str, str], int] = {}
    for (card, companion), together in cooccurrence.items():
        card_lists = present_counts[card]
        ratios[(card, companion)] = int(
            card_lists >= minimum and together / card_lists >= share
            and per_copy[(card, companion)] / card_lists >= share
        )
    needs: dict[str, dict[str, int]] = {}
    for (card, companion), together in cooccurrence.items():
        card_lists = present_counts[card]
        if together / card_lists < share:
            continue
        required = ratios.get((card, companion), 0) if ratios.get((card, companion), 0) > ratios.get((companion, card), 0) else 0
        needs.setdefault(card, {})[companion] = required
    return needs


def build_context(corpus: dict[str, Any], *, cutoff: str, embargo_days: int = 7,
                  min_slot_lists: int = 30) -> dict[str, Any]:
    """Build all estimator inputs from events strictly before cutoff minus the embargo."""
    target = _datetime(cutoff)
    if target is None:
        raise ValueError("cutoff must be a valid ISO-8601 timestamp")
    if embargo_days < 0 or min_slot_lists < 1:
        raise ValueError("embargo_days must be non-negative and min_slot_lists must be positive")
    boundary = _utc_midnight(target - timedelta(days=embargo_days))
    event_dates = {str(event["id"]): _datetime(event.get("date")) for event in corpus.get("events", [])}
    missing_dates = sorted(event_id for event_id, date in event_dates.items() if date is None)
    training_event_ids = sorted(event_id for event_id, date in event_dates.items() if date is not None and date < boundary)
    training_set = set(training_event_ids)
    day_by_event = {event_id: date.date().isoformat() for event_id, date in event_dates.items() if date is not None}
    block_ids = {event_id: f"utc-date:{day}" for event_id, day in day_by_event.items()}

    eligible_lists = [row for row in corpus.get("lists", [])
                      if row.get("event") in training_set and _legal(dict(row.get("counts") or {}))]
    support: Counter[tuple[str, int]] = Counter()
    for row in eligible_lists:
        for card, copies in row["counts"].items():
            for copy_number in range(1, copies + 1):
                support[(str(card), copy_number)] += 1
    slot_names = sorted(slot for slot, count in support.items()
                        if count >= min_slot_lists and len(eligible_lists) - count >= min_slot_lists)
    eligible_pairings = [row for row in corpus.get("pairings", [])
                         if row.get("event") in training_set and row.get("exclusion") is None
                         and int(row.get("eligible_view_count") or 0) > 0]
    eligible_keys = {tuple(row["key"]) for row in eligible_pairings}
    eligible_views = [dict(row) for row in corpus.get("pairing_views", [])
                      if row.get("event") in training_set and tuple(row.get("pairing_key", ())) in eligible_keys]
    eligible_records = [dict(row) for row in corpus.get("records", [])
                        if row.get("event") in training_set and event_dates.get(str(row.get("event"))) is not None]

    history: dict[str, list[tuple[datetime, int, int, str]]] = defaultdict(list)
    players_by_event: dict[str, set[str]] = defaultdict(set)
    for row in eligible_records:
        when = event_dates[str(row["event"])]
        wins, losses = int(row.get("wins") or 0), int(row.get("losses") or 0)
        event_id, player = str(row["event"]), str(row["player"])
        history[player].append((when, wins, losses, event_id))
        players_by_event[event_id].add(player)
    strength: dict[tuple[str, str], float] = {}
    for event_id in training_event_ids:
        when = event_dates[event_id]
        for player in players_by_event.get(event_id, ()):
            history_boundary = _utc_midnight(when - timedelta(days=embargo_days))
            earlier = [(wins, losses) for date, wins, losses, other_event in history[player]
                       if date < history_boundary and other_event != event_id]
            total_wins = sum(item[0] for item in earlier)
            total_losses = sum(item[1] for item in earlier)
            strength[(event_id, player)] = math.log((total_wins + STRENGTH_PRIOR_GAMES) /
                                                    (total_losses + STRENGTH_PRIOR_GAMES))
    cutoff_totals: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for row in eligible_records:
        total = cutoff_totals[str(row["player"])]
        total[0] += int(row.get("wins") or 0)
        total[1] += int(row.get("losses") or 0)
    strength_at_cutoff = {player: math.log((wins + STRENGTH_PRIOR_GAMES) /
                                           (losses + STRENGTH_PRIOR_GAMES))
                          for player, (wins, losses) in sorted(cutoff_totals.items())}

    field_counts: Counter[str] = Counter()
    for row in eligible_views:
        field_counts[str(row["opponent_archetype_id"])] += float(row["weight"])
    field_total = sum(field_counts.values())
    field = {key: value / field_total for key, value in sorted(field_counts.items())} if field_total else {}
    catalogue_by_hash: dict[str, dict[str, Any]] = {}
    for row in eligible_lists:
        counts = dict(sorted((str(name), int(count)) for name, count in row["counts"].items()))
        fingerprint = hashlib.sha256(json.dumps(counts, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        entry = catalogue_by_hash.setdefault(fingerprint, {"counts": counts, "list_count": 0,
                                                            "events": set(), "archetype_ids": set()})
        entry["list_count"] += 1
        entry["events"].add(str(row["event"]))
        entry["archetype_ids"].add(str(row["archetype_id"]))
    companion_rules = _prerequisites(eligible_lists)
    catalogue = [{**entry, "events": sorted(entry["events"]),
                  "archetype_ids": sorted(entry["archetype_ids"]),
                  "companions": {card: companion_rules.get(card, {}) for card in entry["counts"]}}
                 for _, entry in sorted(catalogue_by_hash.items())]
    event_block_rows = sorted({(day_by_event[event_id], block_ids[event_id]) for event_id in training_event_ids})
    paired_view_counts = Counter(tuple(row["pairing_key"]) for row in eligible_views)
    coverage = {"training_event_count": len(training_event_ids), "training_list_count": len(eligible_lists),
                "eligible_pairing_count": len(eligible_pairings), "eligible_pairing_view_count": len(eligible_views),
                "unknown_date_event_count": len(missing_dates),
                "events_missing_format": sorted(event_id for event_id in training_event_ids
                                                 if not next((e.get("metadata", {}).get("format")
                                                              for e in corpus.get("events", []) if str(e["id"]) == event_id), None)),
                "pairings_with_one_eligible_view": sum(count == 1 for count in paired_view_counts.values()),
                "pairings_with_two_eligible_views": sum(count == 2 for count in paired_view_counts.values()),
                "pairings_with_other_view_count": sum(count not in (1, 2) for count in paired_view_counts.values()),
                "missing_event_metadata": {
                    "format": sum(not event.get("metadata", {}).get("format") for event in corpus.get("events", [])
                                  if str(event["id"]) in training_set),
                    "game": sum(not event.get("metadata", {}).get("game") for event in corpus.get("events", [])
                                if str(event["id"]) in training_set),
                    "platform": sum(not event.get("metadata", {}).get("details", {}).get("platform")
                                    for event in corpus.get("events", []) if str(event["id"]) in training_set),
                    "isOnline": sum(not isinstance(event.get("metadata", {}).get("details", {}).get("isOnline"), bool)
                                    for event in corpus.get("events", []) if str(event["id"]) in training_set),
                    "phase_mode": sum(not isinstance(event.get("metadata", {}).get("details", {}).get("phases"), list)
                                      for event in corpus.get("events", []) if str(event["id"]) in training_set),
                },
                "events_without_source_metadata": sum(
                    not event.get("metadata", {}).get("details", {}).get("source")
                    for event in corpus.get("events", []) if str(event["id"]) in training_set
                ),
                "unknown_pairing_metadata": {
                    field: sum(row.get(field) in (None, "", "unknown") for row in eligible_views)
                    for field in ("source", "platform", "delivery_mode", "match_mode", "game", "format")
                },
                "catalogue_list_count": len(catalogue)}
    provenance = {"source": corpus.get("audit", {}).get("source_path"),
                  "source_sha256": corpus.get("audit", {}).get("source_sha256"),
                  "snapshot_fingerprint": corpus.get("audit", {}).get("snapshot_fingerprint"),
                  "cutoff": target.isoformat(), "embargo_days": embargo_days,
                  "strict_training_boundary": boundary.isoformat(),
                  "training_event_dates": {event_id: day_by_event[event_id] for event_id in training_event_ids},
                  "event_block_ids": {event_id: block_ids[event_id] for event_id in training_event_ids},
                  "strength_prior_games": STRENGTH_PRIOR_GAMES,
                  "event_start_timestamp_is_result_availability_proxy": True,
                  "source_mode_values": sorted({(str(row.get("source") or "unknown"),
                                                  str(row.get("platform") or "unknown"),
                                                  str(row.get("delivery_mode") or "unknown"),
                                                  str(row.get("match_mode") or "unknown"),
                                                  str(row.get("game") or "unknown"),
                                                  str(row.get("format") or "unknown")) for row in eligible_views}),
                  "missing_source_views": sum(not row.get("source") or row.get("source") == "unknown"
                                              for row in eligible_views),
                  "missing_mode_views": sum(not row.get("delivery_mode") or row.get("delivery_mode") == "unknown"
                                            for row in eligible_views),
                  "missing_match_mode_views": sum(not row.get("match_mode") or row.get("match_mode") == "unknown"
                                                  for row in eligible_views)}
    return {"training_event_ids": training_event_ids, "cutoff": target.isoformat(),
            "embargo_days": embargo_days, "slot_names": slot_names, "field": field,
            "catalogue": catalogue, "strength": strength, "strength_at_cutoff": strength_at_cutoff,
            "strength_key_type": "(event_id, player_id) -> log-odds float; strength_at_cutoff is player_id -> log-odds float",
            "eligible_pairings": eligible_pairings, "eligible_views": eligible_views,
            "eligible_records": eligible_records, "training_lists": eligible_lists,
            "companions": companion_rules,
            "event_block_ids": {event_id: block_ids[event_id] for event_id in training_event_ids},
            "event_blocks": [{"date": date, "id": block_id,
                              "event_ids": sorted(event_id for event_id in training_event_ids
                                                  if block_ids[event_id] == block_id)}
                             for date, block_id in event_block_rows],
            "coverage": coverage, "provenance": provenance}
