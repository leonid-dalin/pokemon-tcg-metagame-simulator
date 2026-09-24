import os

import numpy as np

import pytest

from src.worker import queue
from src.core.scraper import normalize_archetype


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
    monkeypatch.setattr(queue, "LIMITLESS_INGESTION_ENABLED", False)
    assert queue.ingest_limitless_results.call_local() == {"status": "disabled"}


@pytest.mark.unit
def test_panel_decks_map_limitless_ids_to_simulation_names(monkeypatch):
    monkeypatch.setattr(queue, "BDIF_USE_CARD_MODEL", True)
    monkeypatch.setattr(queue.os.path, "exists", lambda path: True)
    monkeypatch.setattr("src.ingestion.model.select_panel_decks", lambda *args, **kwargs: [
        "alakazam-dudunsparce",
        "n-zoroark",
        "unknown-deck",
    ])

    class Store:
        def __init__(self, path):
            pass

        def prepare_for_read(self):
            pass

        def deck_weights(self):
            return {"alakazam-dudunsparce": 0.5}

    monkeypatch.setattr("src.ingestion.store.LimitlessStore", Store)
    assert queue._panel_decks_for_report(["Alakazam Dudunsparce", "N's Zoroark"]) == [
        "Alakazam Dudunsparce",
        "N's Zoroark",
    ]


@pytest.mark.unit
def test_panel_deck_resolution_strips_unique_suffixes_and_preserves_compounds():
    assert normalize_archetype("N's Zoroark") == "n zoroark"
    assert normalize_archetype("Grass") == "grass"
    assert queue._map_panel_decks_to_matrix(
        ["slowking-scr", "n-zoroark", "dragapult-dusknoir"],
        ["Slowking", "N's Zoroark", "Dragapult", "Dragapult Dusknoir"],
    ) == ["Slowking", "N's Zoroark", "Dragapult Dusknoir"]


@pytest.mark.unit
def test_panel_deck_resolution_drops_ambiguous_and_missing_ids(monkeypatch):
    dropped = []

    class Logger:
        def warning(self, event, **kwargs):
            dropped.extend(kwargs["deck_ids"])

    monkeypatch.setattr(queue, "q_logger", Logger())
    assert queue._map_panel_decks_to_matrix(
        ["foo-bar-baz", "missing-deck"],
        ["Foo Bar", "Foo  Bar"],
    ) == []
    assert dropped == ["foo-bar-baz", "missing-deck"]


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

        def observations(self):
            return []

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(queue, "LIMITLESS_INGESTION_ENABLED", True)
    monkeypatch.setattr("src.ingestion.client.LimitlessClient", Client)
    monkeypatch.setattr("src.ingestion.store.LimitlessStore", Store)
    monkeypatch.setattr("src.ingestion.aggregate.build_artifact", lambda store: {"archetypes": []})

    result = queue.ingest_limitless_results.call_local()

    assert result["failed_events"] == [{"id": "bad", "error": "event unavailable"}]
    assert (tmp_path / "data" / "input" / "limitless_input.json").exists()


@pytest.mark.unit
def test_limitless_ingestion_paginates_skips_stored_events_and_preserves_pairings(monkeypatch, tmp_path):
    fetched = []
    stored_ids = {"stored"}
    pairings = []

    class Client:
        @classmethod
        def from_environment(cls):
            return cls()

        def iter_tournaments(self, **params):
            return iter([{"id": "stored"}, {"id": "fresh"}])

        def game_decks(self):
            return []

        def fetch_event_bundle(self, event_id):
            fetched.append(event_id)
            return ({"decklists": False}, [], [{"player1": "p1", "player2": "p2"}])

    class Store:
        def __init__(self, path):
            self.path = path

        def ensure_schema(self):
            pass

        def backfill_deck_names(self, deck_names):
            pass

        def existing_tournament_ids(self):
            return set(stored_ids)

        def upsert_tournament(self, event, details):
            stored_ids.add(event["id"])

        def upsert_standings(self, event_id, standings):
            assert standings == []

        def upsert_pairings(self, event_id, rows):
            pairings.extend(rows)

        def observations(self):
            return []

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(queue, "LIMITLESS_INGESTION_ENABLED", True)
    monkeypatch.setattr(queue, "LIMITLESS_BACKFILL_TOURNAMENTS", 200)
    monkeypatch.setattr("src.ingestion.client.LimitlessClient", Client)
    monkeypatch.setattr("src.ingestion.store.LimitlessStore", Store)
    monkeypatch.setattr("src.ingestion.aggregate.build_artifact", lambda store: {"archetypes": []})

    first = queue.ingest_limitless_results.call_local()
    second = queue.ingest_limitless_results.call_local()

    assert fetched == ["fresh"]
    assert pairings == [{"player1": "p1", "player2": "p2"}]
    assert first["skipped_events"] == 1
    assert second["skipped_events"] == 2


@pytest.mark.unit
def test_bdif_builder_is_not_invoked_when_card_model_is_disabled(monkeypatch):
    monkeypatch.setattr(queue, "BDIF_USE_CARD_MODEL", False)
    monkeypatch.setattr(queue.os.path, "exists", lambda path: (_ for _ in ()).throw(AssertionError(path)))
    assert queue._build_bdif_report_addons() == ({}, {})


@pytest.mark.unit
def test_bdif_builder_prepares_supplied_store_before_reads(monkeypatch, tmp_path):
    calls = []

    class Store:
        def prepare_for_read(self):
            calls.append("prepare")

        def deck_weights(self):
            calls.append("deck_weights")
            return {}

    monkeypatch.setattr(queue, "BDIF_USE_CARD_MODEL", True)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "limitless.db").touch()

    assert queue._build_bdif_report_addons(Store()) == ({}, {})
    assert calls == ["prepare", "deck_weights"]


@pytest.mark.unit
def test_simulation_job_reuses_one_bdif_store(monkeypatch, tmp_path):
    stores = []
    panel_stores = []
    addon_stores = []
    received = []

    class Store:
        def __init__(self, path):
            stores.append(self)

    monkeypatch.setattr(queue, "BDIF_USE_CARD_MODEL", True)
    monkeypatch.setattr(queue.os.path, "exists", lambda path: True)
    monkeypatch.setattr(queue, "LimitlessStore", Store, raising=False)
    monkeypatch.setattr("src.ingestion.store.LimitlessStore", Store)
    monkeypatch.setattr(queue, "_panel_decks_for_report", lambda deck_names, store=None: panel_stores.append(store) or [])
    monkeypatch.setattr(queue, "_build_bdif_report_addons", lambda store=None: addon_stores.append(store) or ({}, {}))
    monkeypatch.setattr(queue, "INPUT_DATA", str(tmp_path / "input.json"))
    monkeypatch.setattr(queue, "load_matchup_data", lambda *args: (["a", "b"], np.array([[0.5, 0.5], [0.5, 0.5]]), {}))
    monkeypatch.setattr(queue, "predict_best_decks", lambda request: {"full_meta": {"a": 0.5, "b": 0.5}})
    monkeypatch.setattr(queue, "swiss_rounds_from_players", lambda players: 1)
    monkeypatch.setattr(queue, "run_monte_carlo_analytics", lambda **kwargs: received.append(kwargs) or {
        "metrics": {},
        "ranked_metrics": {},
        "insufficient_data": [],
        "matchup_panel": {"rows": {}, "unmatched": [], "opponents": []},
    })

    queue.execute_simulation_job.call_local({
        "job_id": "job",
        "deck_names": ["a", "b"],
        "matchup_matrix": [[0.5, 0.5], [0.5, 0.5]],
        "total_players": 4,
    })

    assert len(stores) == 1
    assert panel_stores == [stores[0]]
    assert addon_stores == [stores[0]]


@pytest.mark.unit
def test_simulation_job_continues_when_bdif_report_builder_raises(monkeypatch, tmp_path):
    received = []
    monkeypatch.setattr(queue, "INPUT_DATA", str(tmp_path / "input.json"))
    monkeypatch.setattr(queue, "load_matchup_data", lambda *args: (["a", "b"], np.array([[0.5, 0.5], [0.5, 0.5]]), {}))
    monkeypatch.setattr(queue, "predict_best_decks", lambda request: {"full_meta": {"a": 0.5, "b": 0.5}})
    monkeypatch.setattr(queue, "swiss_rounds_from_players", lambda players: 1)
    monkeypatch.setattr(queue, "_build_bdif_report_addons", lambda: (_ for _ in ()).throw(RuntimeError("addon failed")))
    monkeypatch.setattr(queue, "run_monte_carlo_analytics", lambda **kwargs: received.append(kwargs) or {
        "metrics": {},
        "ranked_metrics": {},
        "insufficient_data": [],
        "matchup_panel": {"rows": {}, "unmatched": [], "opponents": []},
    })

    queue.execute_simulation_job.call_local({
        "job_id": "job",
        "deck_names": ["a", "b"],
        "matchup_matrix": [[0.5, 0.5], [0.5, 0.5]],
        "total_players": 4,
    })

    assert "best60_recommendations" not in received[0]
    assert "h1_report" not in received[0]


@pytest.mark.unit
def test_simulation_job_passes_matchup_details_to_monte_carlo(monkeypatch, tmp_path):
    details = {
        ("a", "a"): {"win_rate": 0.5, "match_count": 10},
        ("a", "b"): {"win_rate": 0.6, "match_count": 10},
        ("b", "a"): {"win_rate": 0.4, "match_count": 10},
        ("b", "b"): {"win_rate": 0.5, "match_count": 10},
    }
    received = []
    monkeypatch.setattr(queue, "INPUT_DATA", str(tmp_path / "input.json"))
    monkeypatch.setattr(
        queue,
        "load_matchup_data",
        lambda *args: (["a", "b"], np.array([[0.5, 0.6], [0.4, 0.5]]), details),
    )
    monkeypatch.setattr(queue, "predict_best_decks", lambda request: {"full_meta": {"a": 0.5, "b": 0.5}})
    monkeypatch.setattr(queue, "run_monte_carlo_analytics", lambda **kwargs: received.append(kwargs) or {
        "metrics": {},
        "ranked_metrics": {},
        "insufficient_data": [],
        "matchup_panel": {"rows": {}, "unmatched": [], "opponents": []},
    })
    monkeypatch.setattr(queue, "_build_bdif_report_addons", lambda: ({}, {}))
    monkeypatch.setattr(queue, "swiss_rounds_from_players", lambda players: 1)

    queue.execute_simulation_job.call_local({
        "job_id": "job",
        "deck_names": ["a", "b"],
        "matchup_matrix": [[0.5, 0.6], [0.4, 0.5]],
        "total_players": 4,
    })

    assert received[0]["matchup_details"] == details
    assert received[0]["matchup_details"][("a", "b")]["match_count"] == 10
