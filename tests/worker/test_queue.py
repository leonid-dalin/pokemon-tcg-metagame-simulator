import os

import numpy as np

import pytest

from src.worker import queue
from src.core.scraper import normalize_archetype
from src.ingestion.model import CardModelNotIdentifiable, PlayerObservation


@pytest.mark.unit
def test_daily_pipeline_reraises_scraper_failures(monkeypatch):
    def fail_fetch(*args, **kwargs):
        raise RuntimeError("scraper failed")

    monkeypatch.setattr(queue, "fetch_live_matchup_data", fail_fetch)

    with pytest.raises(RuntimeError, match="scraper failed"):
        queue.automated_daily_pipeline.call_local()


@pytest.mark.unit
def test_daily_pipeline_writes_through_same_directory_temp_file_and_replaces_target(
    monkeypatch,
    tmp_path,
):
    target = tmp_path / "ea_input.json"
    target.write_text("old data")
    target.chmod(0o600)
    target_mode = target.stat().st_mode & 0o777
    monkeypatch.setattr(queue, "INPUT_DATA", str(target))
    monkeypatch.setattr(
        queue,
        "fetch_live_matchup_data",
        lambda target_urls, canonical_map: [
            {
                "deck_archetype": "Archetype",
                "opponent_archetype": "Archetype",
                "total_matches": 0,
                "wins": 0,
                "losses": 0,
                "ties": 0,
                "win_rate": 0.5,
            }
        ],
    )
    monkeypatch.setattr(
        queue,
        "build_complete_matchup_matrix",
        lambda matchups: {
            "archetypes": ["Archetype"],
            "matchup_matrix": {
                "Archetype": {
                    "Archetype": {"win_rate": 0.5, "match_count": 0},
                },
            },
        },
    )

    replaced = []
    chmod_calls = []
    real_replace = queue.os.replace
    real_chmod = queue.os.chmod

    def record_replace(source, destination):
        replaced.append((source, destination))
        return real_replace(source, destination)

    def record_chmod(path, mode):
        assert mode == target_mode
        chmod_calls.append((path, mode))
        return real_chmod(path, mode)

    monkeypatch.setattr(queue.os, "replace", record_replace)
    monkeypatch.setattr(queue.os, "chmod", record_chmod)

    queue.automated_daily_pipeline.call_local()

    assert replaced
    source, destination = replaced[0]
    assert source == f"{target}.tmp"
    assert destination == str(target)
    assert target.read_text() != "old data"
    assert target.stat().st_mode & 0o777 == target_mode
    assert not os.path.exists(source)
    assert chmod_calls == [(source, target_mode)]


@pytest.mark.unit
def test_daily_pipeline_removes_temp_file_when_write_fails(monkeypatch, tmp_path):
    target = tmp_path / "ea_input.json"
    target.write_text("old data")
    monkeypatch.setattr(queue, "INPUT_DATA", str(target))
    monkeypatch.setattr(
        queue,
        "fetch_live_matchup_data",
        lambda target_urls, canonical_map: [
            {
                "deck_archetype": "Archetype",
                "opponent_archetype": "Archetype",
                "total_matches": 0,
                "wins": 0,
                "losses": 0,
                "ties": 0,
                "win_rate": 0.5,
            }
        ],
    )
    monkeypatch.setattr(
        queue,
        "build_complete_matchup_matrix",
        lambda matchups: {
            "archetypes": ["Archetype"],
            "matchup_matrix": {
                "Archetype": {
                    "Archetype": {"win_rate": 0.5, "match_count": 0},
                },
            },
        },
    )
    monkeypatch.setattr(
        queue.json,
        "dump",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("write failed")),
    )

    with pytest.raises(OSError, match="write failed"):
        queue.automated_daily_pipeline.call_local()

    assert target.read_text() == "old data"
    assert not os.path.exists(str(tmp_path / "ea_input.json.tmp"))


@pytest.mark.unit
def test_daily_pipeline_removes_temp_file_when_replace_fails(monkeypatch, tmp_path):
    target = tmp_path / "ea_input.json"
    target.write_text("old data")
    monkeypatch.setattr(queue, "INPUT_DATA", str(target))
    monkeypatch.setattr(
        queue,
        "fetch_live_matchup_data",
        lambda target_urls, canonical_map: [
            {
                "deck_archetype": "Archetype",
                "opponent_archetype": "Archetype",
                "total_matches": 0,
                "wins": 0,
                "losses": 0,
                "ties": 0,
                "win_rate": 0.5,
            }
        ],
    )
    monkeypatch.setattr(
        queue,
        "build_complete_matchup_matrix",
        lambda matchups: {
            "archetypes": ["Archetype"],
            "matchup_matrix": {
                "Archetype": {
                    "Archetype": {"win_rate": 0.5, "match_count": 0},
                },
            },
        },
    )
    monkeypatch.setattr(queue.os, "replace", lambda source, destination: (_ for _ in ()).throw(OSError("replace failed")))

    with pytest.raises(OSError, match="replace failed"):
        queue.automated_daily_pipeline.call_local()

    assert target.read_text() == "old data"
    assert not os.path.exists(str(tmp_path / "ea_input.json.tmp"))


@pytest.mark.unit
def test_limitless_ingestion_is_disabled_by_default(monkeypatch):
    from src.bdif.settings import BdifSettings
    monkeypatch.setattr(BdifSettings, "from_environment", classmethod(lambda cls: BdifSettings(False, False, "unused.db", "input.json", "ingest.json", "model.json", 0.01, 10, (), 10)))
    assert queue.ingest_limitless_results.call_local() == {"status": "disabled"}


@pytest.mark.unit
def test_limitless_ingestion_records_failed_events_and_writes_partial_artifact(monkeypatch, tmp_path):
    class Client:
        @classmethod
        def from_environment(cls):
            return cls()

        def iter_tournaments(self, **params):
            return [{"id": "good"}, {"id": "bad"}]

        def game_decks(self):
            return []

        def fetch_event_bundle(self, event_id):
            if event_id == "bad":
                raise RuntimeError("event unavailable")
            return ({}, [], [])

    class Store:
        def __init__(self, path):
            self.path = path

        def ensure_schema(self):
            pass

        def backfill_deck_names(self, deck_names):
            pass

        def existing_tournament_ids(self):
            return set()

        def upsert_tournament(self, event, details):
            pass

        def upsert_standings(self, event_id, standings):
            pass

        def upsert_pairings(self, event_id, pairings):
            pass

        def player_observations(self):
            return []

    monkeypatch.chdir(tmp_path)
    from src.bdif.settings import BdifSettings
    monkeypatch.setattr(BdifSettings, "from_environment", classmethod(lambda cls: BdifSettings(True, True, "data/limitless.db", "input.json", "data/input/limitless_input.json", "data/input/limitless_model_input.json", 0.01, 10, ("fallback",), 10)))
    monkeypatch.setattr("src.ingestion.client.LimitlessClient", Client)
    monkeypatch.setattr("src.ingestion.store.LimitlessStore", Store)
    monkeypatch.setattr("src.ingestion.aggregate.build_artifact", lambda store: {"archetypes": []})

    result = queue.ingest_limitless_results.call_local()

    assert result["failed_events"] == [{"id": "bad", "error": "event unavailable"}]
    assert (tmp_path / "data" / "input" / "limitless_input.json").exists()
    assert result["model_status"] == "insufficient observations"
