import copy
from datetime import datetime, timedelta, timezone

from src.ingestion.best60_research.context import build_context
from src.ingestion.best60_research.context import _prerequisites
from src.ingestion.best60 import ArchetypeList, prerequisites
from src.ingestion.best60_research.splits import split_events


def _list(event, player, counts=None):
    return {"event": event, "player": player, "archetype_id": "a", "archetype": "A",
            "counts": counts or {"Core": 1, "Grass Energy": 59}, "raw": {}}


def _corpus(count=20):
    events, lists, pairings, views, records = [], [], [], [], []
    for index in range(count):
        event = f"e{index}"
        day_offset = {1: 2, 2: 20}.get(index, 10 * index)
        date = (datetime(2026, 9, 1, tzinfo=timezone.utc) + timedelta(days=day_offset)).isoformat()
        events.append({"id": event, "date": date, "metadata": {"format": "standard"}})
        lists += [_list(event, "p"), _list(event, "q")]
        records += [{"event": event, "player": "p", "date": date,
                     "wins": 2 if index == 0 else 1, "losses": 0 if index == 0 else 1},
                    {"event": event, "player": "q", "date": date, "wins": 1, "losses": 1}]
        key = (event, 1, 1, "", "p", "q")
        pairings.append({"key": key, "event": event, "exclusion": None, "eligible_view_count": 2})
        for player, opponent in (("p", "q"), ("q", "p")):
            views.append({"pairing_key": key, "event": event, "player": player, "opponent": opponent,
                          "archetype_id": "a", "opponent_archetype_id": "b", "opponent_archetype": "B",
                          "counts": {"Core": 1, "Grass Energy": 59}, "won": int(player == "p"),
                          "weight": .5, "source": "unknown", "platform": "tournament-platform",
                          "delivery_mode": "online", "match_mode": "BO1", "game": "PTCG",
                          "format": "standard", "date": date})
    return {"events": events, "lists": lists, "pairings": pairings, "pairing_views": views, "records": records,
            "audit": {"source_sha256": "abc", "source_path": "readonly.db"}}


def test_chronology_keeps_utc_date_ties_together():
    events = [{"id": f"e{i}", "date": f"2026-09-{1+i//2:02d}T00:00:00Z"} for i in range(20)]
    parts = split_events(list(reversed(events)), fractions=(0.6, 0.2, 0.2))
    all_ids = parts["development"] + parts["validation"] + parts["test"]
    assert set(parts) == {"development", "validation", "test"}
    assert len(all_ids) == len(set(all_ids)) == 20
    dates = {event["id"]: event["date"] for event in events}
    owner = {event: part for part, ids in parts.items() for event in ids}
    for i in range(0, 20, 2):
        assert owner[f"e{i}"] == owner[f"e{i+1}"]
    assert max(dates[event] for event in parts["development"]) < min(dates[event] for event in parts["validation"])
    assert max(dates[event] for event in parts["validation"]) < min(dates[event] for event in parts["test"])


def test_split_normalizes_timezone_and_rejects_missing_dates():
    parts = split_events([{"id": "a", "date": "2026-09-01T23:30:00-02:00"},
                          {"id": "b", "date": "2026-09-02T00:30:00Z"},
                          {"id": "c", "date": "2026-09-03"}], fractions=(0.34, 0.33, 0.33))
    assert set(parts["development"]) == {"a", "b"}
    try:
        split_events([{"id": "x", "date": None}])
    except ValueError:
        pass
    else:
        raise AssertionError("missing date must fail split construction")


def test_embargo_future_poisoning_and_unknown_slots():
    corpus = _corpus()
    first = build_context(corpus, cutoff="2026-09-09T00:00:00Z", embargo_days=7, min_slot_lists=1)
    poisoned = copy.deepcopy(corpus)
    poisoned["lists"] = [dict(row, counts={"Future Only": 4, "Grass Energy": 56})
                         if row["event"] == "e1" else row for row in poisoned["lists"]]
    poisoned["records"] = [dict(row, wins=999, losses=0) if row["event"] == "e1" else row
                           for row in poisoned["records"]]
    second = build_context(poisoned, cutoff="2026-09-09T00:00:00Z", embargo_days=7, min_slot_lists=1)
    for field in ("training_event_ids", "slot_names", "field", "catalogue", "strength"):
        assert first[field] == second[field]
    assert first["training_event_ids"] == [f"e{i}" for i in range(1)]
    assert all(slot[0] != "Future Only" for slot in first["slot_names"])
    assert first["eligible_pairings"][0]["key"] in {tuple(row["pairing_key"]) for row in first["eligible_views"]}


def test_cutoff_equality_is_embargoed_and_strength_uses_prior_records_only():
    corpus = _corpus(10)
    # The event on the cutoff minus embargo boundary is not a training event.
    context = build_context(corpus, cutoff="2026-09-08T00:00:00Z", embargo_days=7, min_slot_lists=1)
    assert context["training_event_ids"] == []
    context = build_context(corpus, cutoff="2026-10-01T00:00:00Z", embargo_days=7, min_slot_lists=1)
    assert context["strength_key_type"].startswith("(event_id, player_id)")
    assert context["strength"][("e1", "p")] == 0.0
    assert context["strength"][("e2", "p")] > 0.0
    assert context["event_block_ids"]["e1"] == "utc-date:2026-09-03"
    assert context["provenance"]["source_mode_values"] == [
        ("unknown", "tournament-platform", "online", "BO1", "PTCG", "standard")
    ]


def test_sparse_companion_counts_match_legacy_prerequisite_semantics():
    lists = []
    for index in range(40):
        counts = {"A": 2, "Grass Energy": 58} if index < 36 else {"A": 1, "Grass Energy": 59}
        if index < 34:
            counts["B"] = 2
            counts["Grass Energy"] -= 2
        lists.append(_list(f"e{index}", f"p{index}", counts))
    expected = prerequisites([ArchetypeList("e", "p", None, "", 0, None, 0, 0,
                                            row["counts"], {}) for row in lists])
    assert _prerequisites(lists) == expected
