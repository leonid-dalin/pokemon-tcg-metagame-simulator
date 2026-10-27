import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from src.ingestion.best60_research.context import build_context
from src.ingestion.best60_research.experiment import _event_block, _predictive_coverage, prepare_simulation_context
from src.ingestion.best60_research.simulation import _split_ids
from src.ingestion.best60_research.splits import event_blocks, split_events


def _event(event_id, date):
    return {"id": event_id, "date": date}


def test_events_on_the_same_day_sharing_a_player_form_one_unit():
    events = [_event("a", "2026-09-01T00:00:00Z"), _event("b", "2026-09-01T06:00:00Z")]
    players = {"a": {"p1", "p2"}, "b": {"p2", "p3"}}

    blocks = event_blocks(events, players)

    assert [ids for _, ids in blocks] == [["a", "b"]]


def test_events_on_the_same_day_without_a_shared_player_stay_separate():
    events = [_event("a", "2026-09-01T00:00:00Z"), _event("b", "2026-09-01T06:00:00Z")]
    players = {"a": {"p1"}, "b": {"p2"}}

    blocks = event_blocks(events, players)

    assert [ids for _, ids in blocks] == [["a"], ["b"]]


def test_events_on_different_days_never_join_even_with_a_shared_player():
    events = [_event("a", "2026-09-01T23:00:00Z"), _event("b", "2026-09-02T01:00:00Z")]
    players = {"a": {"p1"}, "b": {"p1"}}

    blocks = event_blocks(events, players)

    assert [ids for _, ids in blocks] == [["a"], ["b"]]


def test_event_block_split_takes_sixty_twenty_twenty_by_unit():
    events = [_event(f"e{index}", f"2026-09-{index + 1:02d}T00:00:00Z") for index in range(10)]
    players = {f"e{index}": {f"p{index}"} for index in range(10)}

    splits = split_events(events, unit="event_block", players_by_event=players)

    assert [len(splits[name]) for name in ("development", "validation", "test")] == [6, 2, 2]


def test_simulation_split_reads_the_protocol_fractions():
    pairs = np.array([f"p{index}" for index in range(20)], dtype=object)
    events = np.array([f"e{index}" for index in range(20)], dtype=object)
    blocks = np.array([f"utc-date:2026-09-{index + 1:02d}" for index in range(20)], dtype=object)

    train, selection, measurement, unit = _split_ids(
        pairs, events, blocks, np.random.default_rng(7), fractions=(0.5, 0.25, 0.25),
    )

    assert unit == "utc_date"
    assert (len(train), len(selection), len(measurement)) == (10, 5, 5)


def test_the_v2_protocol_split_is_read_as_a_mapping():
    from src.ingestion.best60_research.runner import _split_fractions

    protocol = json.loads(Path("docs/research/best60-protocol-v2.json").read_text(encoding="utf-8"))

    assert _split_fractions(protocol) == (0.6, 0.2, 0.2)


def test_event_block_ids_flow_into_the_simulation_split():
    views = []
    block_ids = {}
    for day in range(40):
        date = f"2026-01-{day + 1:02d}"
        for slot in range(3):
            event = f"e{day}-{slot}"
            block_ids[event] = f"event-block:{date}:{event}"
            views.append({
                "event": event, "player": f"p{day}-{slot}", "opponent": f"o{day}-{slot}",
                "side": 1, "won": slot % 2, "counts": {"A": 1},
                "archetype_id": "kangaskhan-bouffalant", "pairing_key": (event, slot),
            })
    context = {"eligible_views": views, "event_block_ids": block_ids, "slot_names": [("A", 1)], "strength": {}}

    arrays = prepare_simulation_context(context)
    train, selection, measurement, unit = _split_ids(
        arrays["pairing_ids"], arrays["event_ids"], arrays["block_ids"], np.random.default_rng(7),
    )

    block_of = {tuple(pair): block for pair, block in zip(arrays["pairing_ids"], arrays["block_ids"])}

    assert unit == "event_block"
    assert len({block_of[tuple(pair)] for pair in measurement}) >= 20


def test_build_context_emits_event_block_ids_under_v2():
    events = [{"id": f"e{index}", "date": f"2026-01-{index + 1:02d}T00:00:00Z"}
              for index in range(4)]
    lists = [{"event": event["id"], "player": f"p{event['id']}", "counts": {"A": 4, "B": 56},
              "archetype_id": "kangaskhan-bouffalant"} for event in events]
    corpus = {"events": events, "lists": lists, "pairings": []}
    players = {event["id"]: {f"p{event['id']}"} for event in events}

    context = build_context(corpus, cutoff="2026-02-01T00:00:00Z", embargo_days=0, min_slot_lists=1,
                            unit="event_block", players_by_event=players)

    assert context["event_block_ids"]
    assert all(str(value).startswith("event-block:") for value in context["event_block_ids"].values())


def test_predictive_coverage_is_inconclusive_with_a_measured_row_count():
    blocks = [f"b{index}" for index in range(10)]
    logits = [-9.0] * 10
    outcomes = [0.0] * 10

    result = _predictive_coverage({"interval": [0.85, 0.95]}, blocks, logits, outcomes)

    assert result["status"] == "inconclusive"
    assert result["held_out_rows"] == 10
    assert result["latent_mean_coverage"] is None
    assert result["future_aggregate_coverage"] is None


def test_predictive_coverage_is_inconclusive_without_an_interval():
    result = _predictive_coverage(None, ["b0", "b1"], [0.0, 0.0], [1.0, 0.0])

    assert result["status"] == "inconclusive"
    assert result["reason"] == "no_fixed_list_interval"
    assert result["latent_mean_coverage"] is None
    assert result["future_aggregate_coverage"] is None


def test_the_v2_protocol_split_is_read_as_a_mapping():
    from src.ingestion.best60_research.runner import _split_fractions

    protocol = json.loads(Path("docs/research/best60-protocol-v2.json").read_text(encoding="utf-8"))
    protocol["time"]["split_fractions"] = {"development": 0.5, "validation": 0.25, "test": 0.25}

    assert _split_fractions(protocol) == (0.5, 0.25, 0.25)


def test_stage_context_uses_the_protocol_unit(monkeypatch, tmp_path):
    from src.ingestion.best60_research import runner

    database = tmp_path / "corpus.db"
    import sqlite3
    connection = sqlite3.connect(database)
    connection.execute("CREATE TABLE standings (tournament_id TEXT, player_id TEXT)")
    connection.execute("INSERT INTO standings VALUES ('e1', 'p1')")
    connection.commit()
    connection.close()
    captured = {}
    monkeypatch.setattr(runner, "build_context", lambda *args, **kwargs: captured.update(kwargs) or {})
    protocol = {"time": {"primary_embargo_days": 7, "atomic_unit": "event_block"},
                "features": {"slot_minimum_lists": 1}}

    runner._stage_context(str(database), {"events": []}, protocol, "2026-06-01T00:00:00Z")

    assert captured["unit"] == "event_block"
    assert captured["players_by_event"] == {"e1": {"p1"}}


def test_test_rows_cluster_on_event_blocks_under_v2():
    events = []
    lists = []
    players = {}
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for day in range(40):
        date = (start + timedelta(days=day)).isoformat()
        for slot in range(3):
            event = f"e{day}-{slot}"
            events.append({"id": event, "date": date})
            lists.append({"event": event, "player": f"p{day}-{slot}", "counts": {"A": 4, "B": 56},
                          "archetype_id": "kangaskhan-bouffalant"})
            players[event] = {f"p{day}-{slot}"}
    corpus = {"events": events, "lists": lists, "pairings": []}

    context = build_context(corpus, cutoff="2026-02-01T00:00:00Z", embargo_days=0, min_slot_lists=1,
                            unit="event_block", players_by_event=players)
    clusters = {_event_block(context, {"event": event["id"], "date": event["date"]}) for event in events}
    days = {event["date"][:10] for event in events}

    assert all(cluster.startswith("event-block:") for cluster in clusters)
    assert len(clusters) == 120
    assert len(days) == 40
