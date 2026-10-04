import hashlib
import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from src.ingestion.model import PlayerObservation, fit_card_model
from src.ingestion.model_cache import load_or_fit_card_model
from src.ingestion.store import LimitlessStore


def _observations():
    rows = []
    for index in range(120):
        has_tech = index % 2 == 0
        a_cards = frozenset({"Core", "Tech"} if has_tech else {"Core"})
        result = int(index % 10 < (8 if has_tech else 2))
        result = 1 - result if index % 11 == 0 else result
        players = (f"a{index}", f"b{index}")
        rows.append(PlayerObservation("a", "b", a_cards, frozenset({"Core"}), result, players))
        rows.append(PlayerObservation("b", "a", frozenset({"Core"}), a_cards, 1 - result, players[::-1]))
    return rows


@pytest.mark.unit
def test_card_model_uses_configured_iteration_limit(monkeypatch, tmp_path):
    calls = []
    real = LogisticRegression

    def factory(*args, **kwargs):
        calls.append(kwargs["max_iter"])
        return real(*args, **kwargs)

    monkeypatch.setattr("src.ingestion.model.LogisticRegression", factory)

    fit_card_model(_observations(), ["Tech"], max_iter=10_000)

    assert calls == [10_000]


@pytest.mark.unit
def test_fitted_model_cache_is_reused_only_for_matching_database_sha(tmp_path, monkeypatch):
    cache = tmp_path / "limitless_model_fit.json"
    calls = []
    real = fit_card_model

    def counted(observations, cards, **kwargs):
        calls.append(tuple(cards))
        return real(observations, cards, **kwargs)

    monkeypatch.setattr("src.ingestion.model_cache.fit_card_model", counted)
    before = hashlib.sha256(b"snapshot-1").hexdigest()
    after = hashlib.sha256(b"snapshot-2").hexdigest()

    first, loaded_first = load_or_fit_card_model(_observations(), before, cache)
    reloaded, loaded_second = load_or_fit_card_model([], before, cache)
    refitted, loaded_third = load_or_fit_card_model(_observations(), after, cache)

    assert loaded_first is False
    assert loaded_second is True
    assert loaded_third is False
    assert len(calls) == 2
    assert first.probability("a", "b") == pytest.approx(reloaded.probability("a", "b"))
    assert first.probability("a", "b") == pytest.approx(refitted.probability("a", "b"))
    assert json.loads(cache.read_text(encoding="utf-8"))["database_sha256"] == after


@pytest.mark.unit
def test_model_cache_with_another_database_sha_is_not_loaded(tmp_path):
    cache = tmp_path / "limitless_model_fit.json"
    old_sha = hashlib.sha256(b"old").hexdigest()
    new_sha = hashlib.sha256(b"new").hexdigest()

    fitted, reused = load_or_fit_card_model(_observations(), old_sha, cache)

    assert reused is False
    loaded = load_or_fit_card_model(_observations(), new_sha, cache)[0]
    assert loaded.probability("a", "b") == pytest.approx(fitted.probability("a", "b"))


@pytest.mark.unit
def test_event_bundle_rejects_duplicate_pairing_identity_before_writing(tmp_path):
    store = LimitlessStore(tmp_path / "snapshot.db", deck_mapping={})
    event = {"id": "event-1"}
    pairings = [
        {"round": 1, "phase": 1, "match": "A", "player1": "p1", "player2": "p2", "winner": "p1"},
        {"round": 1, "phase": 1, "match": "A", "player1": "p1", "player2": "p2", "winner": "p2"},
    ]

    with pytest.raises(sqlite3.IntegrityError, match="duplicate pairing identity"):
        store.upsert_event_bundle(event, {}, [], pairings)

    with store.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM tournaments").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM pairings").fetchone()[0] == 0
