import os

import numpy as np

import pytest

from src.worker import queue


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
def test_simulation_job_continues_when_bdif_report_builder_raises(monkeypatch, tmp_path):
    received = []
    monkeypatch.setattr(queue, "INPUT_DATA", str(tmp_path / "input.json"))
    monkeypatch.setattr(queue, "load_matchup_data", lambda *args: (["a", "b"], np.array([[0.5, 0.5], [0.5, 0.5]]), {}))
    monkeypatch.setattr(queue, "predict_best_decks", lambda request: {"full_meta": {"a": 0.5, "b": 0.5}})
    monkeypatch.setattr(queue, "swiss_rounds_from_players", lambda players: 1)
    monkeypatch.setattr(queue, "_build_bdif_report_addons", lambda: (_ for _ in ()).throw(RuntimeError("addon failed")))
    monkeypatch.setattr(queue, "run_monte_carlo_analytics", lambda **kwargs: received.append(kwargs) or {})

    queue.execute_simulation_job.call_local({
        "job_id": "job",
        "deck_names": ["a", "b"],
        "matchup_matrix": [[0.5, 0.5], [0.5, 0.5]],
        "total_players": 4,
    })

    assert received[0]["best60_recommendations"] == {}
    assert received[0]["h1_report"] == {}


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
    monkeypatch.setattr(queue, "run_monte_carlo_analytics", lambda **kwargs: received.append(kwargs) or {})
    monkeypatch.setattr(queue, "swiss_rounds_from_players", lambda players: 1)

    queue.execute_simulation_job.call_local({
        "job_id": "job",
        "deck_names": ["a", "b"],
        "matchup_matrix": [[0.5, 0.6], [0.4, 0.5]],
        "total_players": 4,
    })

    assert received[0]["matchup_details"] is details
