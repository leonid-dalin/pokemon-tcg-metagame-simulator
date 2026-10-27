import numpy as np

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
