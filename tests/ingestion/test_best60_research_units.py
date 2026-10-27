import json
from pathlib import Path

import numpy as np

from src.ingestion.best60_research.context import build_context
from src.ingestion.best60_research.experiment import _predictive_coverage, prepare_simulation_context
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


def test_predictive_coverage_measures_latent_and_future_hits():
    blocks = [f"b{index}" for index in range(10)]
    logits = [0.0] * 10
    outcomes = [1.0, 0.0] * 5

    inside = _predictive_coverage({"interval": [0.4, 0.6]}, blocks, logits, outcomes)
    outside = _predictive_coverage({"interval": [0.7, 0.9]}, blocks, logits, outcomes)

    assert inside["status"] == "complete"
    assert inside["latent_mean_coverage"] == 1.0
    assert inside["future_aggregate_coverage"] == 1.0
    assert outside["latent_mean_coverage"] == 0.0
    assert outside["future_aggregate_coverage"] == 0.0


def test_predictive_coverage_is_inconclusive_without_an_interval():
    result = _predictive_coverage(None, ["b0", "b1"], [0.0, 0.0], [1.0, 0.0])

    assert result["status"] == "inconclusive"
    assert result["latent_mean_coverage"] is None
