import argparse
import importlib.util
import json
from pathlib import Path

import pytest

from src.ingestion.model import PlayerObservation
from src.ingestion.store import LimitlessStore

_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "refresh_limitless_snapshot.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("refresh_limitless_snapshot", _SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


class _FakeClient:
    def tournaments(self, **params):
        return []

    def game_decks(self, **kwargs):
        return []


@pytest.mark.unit
def test_refresh_hashes_the_database_once_for_the_manifest_and_model_status(monkeypatch, tmp_path):
    module = _load_module()
    database = tmp_path / "limitless.db"
    database.write_bytes(b"snapshot-before")

    monkeypatch.setattr(module.LimitlessClient, "from_environment", classmethod(lambda cls: _FakeClient()))
    monkeypatch.setattr(module.LimitlessStore, "ensure_schema", lambda self: None)
    monkeypatch.setattr(module.LimitlessStore, "existing_tournament_ids", lambda self: set())
    monkeypatch.setattr(module.LimitlessStore, "aggregate_rows", lambda self: [])
    monkeypatch.setattr(module.LimitlessStore, "card_inclusion", lambda self, archetype, event_id=None: {})
    monkeypatch.setattr(module.LimitlessStore, "player_observations", lambda self: _observations())
    monkeypatch.setattr(module.LimitlessStore, "pairings_with_decklists", lambda self, pattern: [])
    monkeypatch.setattr(module.LimitlessStore, "summary", lambda self: {"decks": 0, "pairings": 0, "decklists": 0})

    real_sha256 = module._sha256
    calls = {"count": 0}

    def changing_sha256(path):
        calls["count"] += 1
        value = real_sha256(path)
        if calls["count"] == 1:
            database.write_bytes(b"snapshot-after")
        return value

    monkeypatch.setattr(module, "_sha256", changing_sha256)

    args = argparse.Namespace(
        database=str(database),
        artifact_dir=str(tmp_path / "artifacts"),
        manifest=str(tmp_path / "manifest.json"),
        start="2026-01-01",
        end="2026-01-02",
        page_size=10,
        event_id=[],
        refresh_event_id=[],
    )

    result = module.refresh(args)

    assert result["database"]["sha256"] == result["artefacts"]["model_status"]["database_sha256"]
    assert calls["count"] == 1
