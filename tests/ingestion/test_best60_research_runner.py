import json
import sqlite3
import subprocess
import time
from argparse import Namespace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.ingestion.best60_research.runner import (
    RunnerError, _evaluate_locked, _output_is_ignored, _selection_validation_views, _simulate_locked, run_stage,
)
from src.ingestion.best60_research.estimators import FitBudgetExceeded
from src.ingestion.best60_research import __main__ as cli


def _database(path, count=60):
    with sqlite3.connect(path) as con:
        con.executescript("""
            CREATE TABLE tournaments (id TEXT PRIMARY KEY, game TEXT, format TEXT, name TEXT,
                date TEXT, players INTEGER, details_json TEXT);
            CREATE TABLE standings (tournament_id TEXT, player_id TEXT, placing INTEGER,
                wins INTEGER, losses INTEGER, ties INTEGER, deck_id TEXT, deck_name TEXT,
                decklist_json TEXT, dropped_round INTEGER, PRIMARY KEY(tournament_id, player_id));
            CREATE TABLE pairings (tournament_id TEXT, round INTEGER, phase INTEGER, match TEXT,
                player1 TEXT, player2 TEXT, winner TEXT,
                PRIMARY KEY(tournament_id,round,phase,match,player1,player2));
        """)
        for i in range(count):
            event = f"e{i}"
            date = (datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(days=i)).isoformat()
            con.execute("INSERT INTO tournaments VALUES (?,?,?,?,?,?,?)",
                        (event, "PTCG", "STANDARD", event, date, 16, "{}"))
            decklist = json.dumps({"trainer": [{"name": "Core", "count": 1}],
                                   "energy": [{"name": "Grass Energy", "count": 59}]})
            con.executemany("INSERT INTO standings VALUES (?,?,?,?,?,?,?,?,?,?)", [
                (event, "p", 1, 1, 1, 0, "a", "A", decklist, None),
                (event, "q", 2, 1, 1, 0, "b", "B", None, None),
            ])
            winner = "test-outcome-secret" if i >= 48 else "p"
            con.execute("INSERT INTO pairings VALUES (?,?,?,?,?,?,?)", (event, 1, 1, "", "p", "q", winner))
        con.commit()
    return path


def _args(database, output, config, stage, resume=None, field_file=None):
    return Namespace(database=str(database), output=str(output), config=str(config), stage=stage,
                     as_of=None, field_file=field_file, archetype_id=None, resume=resume, max_seconds=30)


def test_audit_does_not_extract_locked_test_outcomes(tmp_path):
    database = _database(tmp_path / "source.db")
    config = tmp_path / "protocol.json"
    config.write_text(json.dumps({"schema_version": 1, "time": {"primary_embargo_days": 7},
                                  "features": {"slot_minimum_lists": 1},
                                  "estimators": {"max_iterations": 5}}), encoding="utf-8")
    output = tmp_path / "runs"
    result = run_stage(_args(database, output, config, "audit"))
    audit = json.loads((output / result["run_id"] / "audit.json").read_text(encoding="utf-8"))
    assert audit["test_outcomes_read"] is False
    assert "test-outcome-secret" not in json.dumps(audit)
    result_ids = json.loads(
        (output / result["run_id"] / "manifest.json").read_text(encoding="utf-8")
    )["included_event_ids"]
    assert audit["source"]["included_event_ids"] == result_ids
    assert set(result_ids).isdisjoint(json.loads(
        (output / result["run_id"] / "splits.json").read_text(encoding="utf-8")
    )["test"]["event_ids"])


def test_resume_requires_run_and_matching_config(tmp_path):
    database = _database(tmp_path / "source.db")
    config = tmp_path / "protocol.json"
    config.write_text(json.dumps({"schema_version": 1}), encoding="utf-8")
    with pytest.raises(RunnerError) as error:
        run_stage(_args(database, tmp_path / "runs", config, "validate", resume="missing"))
    assert error.value.code == 2


def test_cli_exposes_programming_value_errors(monkeypatch):
    monkeypatch.setattr(cli, "run_stage", lambda _args: (_ for _ in ()).throw(ValueError("implementation bug")))
    with pytest.raises(ValueError, match="implementation bug"):
        cli.main(["--database", "source.db", "--output", "runs", "--config", "protocol.json", "--stage", "audit"])


def test_cli_reports_fit_budget_as_resumable_incomplete(monkeypatch, capsys):
    monkeypatch.setattr(cli, "run_stage", lambda _args: (_ for _ in ()).throw(FitBudgetExceeded("fit budget exhausted")))
    code = cli.main(["--database", "source.db", "--output", "runs", "--config", "protocol.json", "--stage", "validate"])
    assert code == 4
    assert "fit budget exhausted" in capsys.readouterr().err


def test_simulation_runs_full_effect_grid_for_pooled_and_focal_views(monkeypatch, tmp_path):
    from src.ingestion.best60_research import experiment, runner, simulation

    monkeypatch.setattr(runner, "_extract", lambda _database, **_kwargs: {"events": []})
    monkeypatch.setattr(runner, "build_context", lambda *_args, **_kwargs: {
        "cutoff": "2026-01-01T00:00:00+00:00", "embargo_days": 7,
        "coverage": {"training_event_count": 3, "eligible_pairing_count": 12},
        "event_blocks": [{"id": "d0"}, {"id": "d1"}], "slot_names": [("A", 1)],
    })
    scopes = []

    def prepare(_context, archetype_id=None):
        scopes.append(archetype_id or "pooled")
        return {"simulation_scope": archetype_id or "pooled"}

    monkeypatch.setattr(experiment, "prepare_simulation_context", prepare)
    calls = []

    def simulate(context, **kwargs):
        calls.append((context["simulation_scope"], kwargs))
        return {"status": "complete", "completed_replicates": kwargs["replicates"]}

    monkeypatch.setattr(simulation, "run_simulation", simulate)
    manifest = {"archetype_id": "kangaskhan-bouffalant", "splits": {
        "development": {"event_ids": ["d"], "dates": {"d": "2026-01-01"}},
        "validation": {"event_ids": ["v"], "dates": {"v": "2026-01-10"}},
        "test": {"event_ids": ["t"], "dates": {"t": "2026-01-20"}},
    }}
    protocol = {"time": {"primary_embargo_days": 7}, "features": {"slot_minimum_lists": 1},
                "estimators": {"fm_ranks": [2], "fm_seeds": [1312],
                               "interaction_penalties": [0.001], "max_iterations": 5},
                "h1": {"minimum_independent_event_clusters": 1},
                "simulation": {"full_replicates_per_effect": 5, "effects_log_odds": [0.0, 0.2],
                               "full_rank_grid": [2], "full_seed_grid": [1312],
                               "full_interaction_penalty_grid": [0.001],
                               "full_wall_clock_seconds": 30}}
    result = _simulate_locked("unused.db", manifest, protocol, tmp_path, time.monotonic(), None)
    assert result["status"] == "complete"
    assert scopes == ["pooled", "kangaskhan-bouffalant"]
    assert len(calls) == 4
    assert {call[1]["replicates"] for call in calls} == {5}
    assert {call[1]["mode"] for call in calls} == {"package"}
    assert {call[1]["effect"] for call in calls} == {0.0, 0.2}


def test_simulation_checkpoint_shortfall_stays_incomplete(monkeypatch, tmp_path):
    from src.ingestion.best60_research import experiment, runner, simulation

    monkeypatch.setattr(runner, "_extract", lambda _database, **_kwargs: {"events": []})
    monkeypatch.setattr(runner, "build_context", lambda *_args, **_kwargs: {
        "cutoff": "2026-01-01T00:00:00+00:00", "embargo_days": 7,
        "coverage": {"training_event_count": 3, "eligible_pairing_count": 12},
        "event_blocks": [], "slot_names": [],
    })
    monkeypatch.setattr(experiment, "prepare_simulation_context", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(simulation, "run_simulation", lambda *_args, **_kwargs: {
        "status": "incomplete", "completed_replicates": 1,
    })
    manifest = {"archetype_id": "kangaskhan-bouffalant", "splits": {
        "development": {"event_ids": ["d"], "dates": {"d": "2026-01-01"}},
        "validation": {"event_ids": ["v"], "dates": {"v": "2026-01-10"}},
        "test": {"event_ids": ["t"], "dates": {"t": "2026-01-20"}},
    }}
    protocol = {"time": {"primary_embargo_days": 7}, "features": {"slot_minimum_lists": 1},
                "estimators": {"fm_ranks": [2], "fm_seeds": [1312],
                               "interaction_penalties": [0.001], "max_iterations": 5},
                "h1": {"minimum_independent_event_clusters": 1},
                "simulation": {"full_replicates_per_effect": 5, "effects_log_odds": [0.0, 0.2],
                               "full_rank_grid": [2], "full_seed_grid": [1312],
                               "full_interaction_penalty_grid": [0.001],
                               "full_wall_clock_seconds": 30}}
    result = _simulate_locked("unused.db", manifest, protocol, tmp_path, time.monotonic(), None)
    assert result["status"] == "incomplete"
    assert result["benchmarks"]["pooled"]["effects"]["0"]["completed_replicates"] == 1
    assert result["benchmarks"]["kangaskhan-bouffalant"]["status"] == "incomplete"


def test_simulation_skips_full_grid_when_test_date_blocks_cannot_identify_h1(monkeypatch, tmp_path):
    from src.ingestion.best60_research import runner, simulation

    monkeypatch.setattr(runner, "_extract", lambda *_args, **_kwargs: pytest.fail("outcomes must not be extracted"))
    monkeypatch.setattr(simulation, "run_simulation", lambda *_args, **_kwargs: pytest.fail("simulation must not run"))
    manifest = {"archetype_id": "kangaskhan-bouffalant", "splits": {
        "development": {"event_ids": ["d"], "dates": {"d": "2026-01-01"}},
        "validation": {"event_ids": ["v"], "dates": {"v": "2026-01-02"}},
        "test": {"event_ids": ["t1", "t2"], "dates": {"t1": "2026-01-03T01:00:00Z", "t2": "2026-01-03T18:00:00Z"}},
    }}
    protocol = {"time": {"primary_embargo_days": 7},
                "h1": {"minimum_independent_event_clusters": 20}}
    result = _simulate_locked("unused.db", manifest, protocol, tmp_path, time.monotonic(), None)
    assert result["status"] == "inconclusive"
    assert result["simulation_design_status"] == "inconclusive_design"
    assert result["preflight"]["measurement_event_count"] == 2
    assert result["preflight"]["test_split_date_block_count"] == 1
    assert result["preflight"]["pretest_training_date_block_count"] == 0
    assert result["preflight"]["measurement_date_block_upper_bound"] == 0
    assert result["preflight"]["replicates_executed"] == 0
    assert result["preflight"]["test_outcomes_read"] is False


def test_simulation_preflight_uses_training_context_not_final_test_blocks(monkeypatch, tmp_path):
    from src.ingestion.best60_research import experiment, runner, simulation

    monkeypatch.setattr(runner, "_extract", lambda _database, **_kwargs: {"events": []})
    monkeypatch.setattr(runner, "build_context", lambda *_args, **_kwargs: {
        "cutoff": "2026-06-01T00:00:00+00:00", "embargo_days": 7,
        "coverage": {"training_event_count": 120, "eligible_pairing_count": 500},
        "event_blocks": [{"id": f"d{i}"} for i in range(120)], "slot_names": [("A", 1)],
    })
    monkeypatch.setattr(experiment, "prepare_simulation_context", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(simulation, "run_simulation", lambda _context, **kwargs: {
        "status": "complete", "completed_replicates": kwargs["replicates"],
    })
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    development_ids = [f"d{i}" for i in range(120)]
    development_dates = {event_id: (start + timedelta(days=i)).isoformat()
                         for i, event_id in enumerate(development_ids)}
    manifest = {"archetype_id": "kangaskhan-bouffalant", "splits": {
        "development": {"event_ids": development_ids, "dates": development_dates},
        "validation": {"event_ids": ["v"], "dates": {"v": "2026-05-02T00:00:00Z"}},
        "test": {"event_ids": ["t1", "t2"], "dates": {
            "t1": "2026-06-01T01:00:00Z", "t2": "2026-06-01T18:00:00Z"}},
    }}
    protocol = {"time": {"primary_embargo_days": 7}, "features": {"slot_minimum_lists": 1},
                "estimators": {"fm_ranks": [2], "fm_seeds": [1312],
                               "interaction_penalties": [0.001], "max_iterations": 5},
                "h1": {"minimum_independent_event_clusters": 20},
                "simulation": {"full_replicates_per_effect": 1, "effects_log_odds": [0.0],
                               "full_rank_grid": [2], "full_seed_grid": [1312],
                               "full_interaction_penalty_grid": [0.001],
                               "full_wall_clock_seconds": 30}}
    result = _simulate_locked("unused.db", manifest, protocol, tmp_path, time.monotonic(), None)
    assert result["status"] == "complete"
    assert result["preflight"]["test_split_date_block_count"] == 1
    assert result["preflight"]["pretest_training_date_block_count"] == 121
    assert result["preflight"]["measurement_date_block_upper_bound"] == 25


def test_simulation_preflight_gates_each_benchmark_separately(monkeypatch, tmp_path):
    import sqlite3

    import numpy as np

    from src.ingestion.best60_research import experiment, runner, simulation

    database = tmp_path / "corpus.db"
    connection = sqlite3.connect(database)
    connection.execute("CREATE TABLE standings (tournament_id TEXT, player_id TEXT)")
    connection.executemany("INSERT INTO standings VALUES (?, ?)", [(f"d{i}", f"p{i}") for i in range(120)])
    connection.commit()
    connection.close()
    monkeypatch.setattr(runner, "_extract", lambda _database, **_kwargs: {"events": [], "lists": [], "pairings": []})
    monkeypatch.setattr(runner, "build_context", lambda *_args, **_kwargs: {
        "cutoff": "2026-06-01T00:00:00+00:00", "embargo_days": 7,
        "coverage": {"training_event_count": 120, "eligible_pairing_count": 500},
        "event_blocks": [{"id": f"d{i}"} for i in range(120)], "slot_names": [("A", 1)],
    })

    def prepare(_context, archetype_id=None):
        count = 120 if archetype_id is None else 60
        return {
            "pairing_ids": np.array([f"p{i}" for i in range(count)], dtype=object),
            "event_ids": np.array([f"e{i}" for i in range(count)], dtype=object),
            "block_ids": np.array([f"event-block:2026-01-{i:03d}:e{i}" for i in range(count)], dtype=object),
        }

    monkeypatch.setattr(experiment, "prepare_simulation_context", prepare)
    calls = []
    monkeypatch.setattr(simulation, "run_simulation", lambda _context, **kwargs: (
        calls.append(kwargs) or {"status": "complete", "completed_replicates": kwargs["replicates"]}
    ))
    development_ids = [f"d{i}" for i in range(120)]
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    development_dates = {event_id: (start + timedelta(days=index)).isoformat()
                         for index, event_id in enumerate(development_ids)}
    manifest = {"archetype_id": "kangaskhan-bouffalant", "splits": {
        "development": {"event_ids": development_ids, "dates": development_dates},
        "validation": {"event_ids": [], "dates": {}},
        "test": {"event_ids": ["t"], "dates": {"t": "2026-06-01T00:00:00Z"}},
    }}
    protocol = {"time": {"primary_embargo_days": 7, "atomic_unit": "event_block",
                         "split_fractions": {"development": 0.6, "validation": 0.2, "test": 0.2}},
                "features": {"slot_minimum_lists": 1},
                "estimators": {"max_iterations": 5},
                "h1": {"minimum_independent_event_clusters": 20},
                "simulation": {"full_replicates_per_effect": 1, "effects_log_odds": [0.0],
                               "full_rank_grid": [2], "full_seed_grid": [1312],
                               "full_interaction_penalty_grid": [0.001], "full_wall_clock_seconds": 30}}

    result = _simulate_locked(str(database), manifest, protocol, tmp_path, time.monotonic(), None)

    assert result["benchmarks"]["pooled"]["status"] == "complete"
    assert result["benchmarks"]["pooled"]["measurement_units"] == 24
    assert result["benchmarks"]["kangaskhan-bouffalant"]["status"] == "inconclusive_design"
    assert result["benchmarks"]["kangaskhan-bouffalant"]["measurement_units"] == 12
    assert all(call["replicates"] == 1 for call in calls)
    assert len(calls) == 1


def test_simulation_preflight_counts_event_blocks_under_protocol_v2(monkeypatch, tmp_path):
    import sqlite3

    from src.ingestion.best60_research import experiment, runner, simulation

    database = tmp_path / "corpus.db"
    connection = sqlite3.connect(database)
    connection.execute("CREATE TABLE standings (tournament_id TEXT, player_id TEXT)")
    connection.executemany("INSERT INTO standings VALUES (?, ?)", [("d0", "p1"), ("d1", "p2"), ("d2", "p3")])
    connection.commit()
    connection.close()
    monkeypatch.setattr(runner, "_extract", lambda _database, **_kwargs: {"events": []})
    monkeypatch.setattr(runner, "build_context", lambda *_args, **_kwargs: {
        "cutoff": "2026-06-01T00:00:00+00:00", "embargo_days": 7,
        "coverage": {"training_event_count": 10, "eligible_pairing_count": 500},
        "event_blocks": [{"id": f"d{i}"} for i in range(10)], "slot_names": [("A", 1)],
    })
    monkeypatch.setattr(experiment, "prepare_simulation_context", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(simulation, "run_simulation", lambda _context, **kwargs: {
        "status": "complete", "completed_replicates": kwargs["replicates"],
    })
    development_ids = [f"d{i}" for i in range(10)]
    development_dates = {"d0": "2026-01-01T00:00:00Z", "d1": "2026-01-01T12:00:00Z"}
    for index in range(2, 10):
        development_dates[f"d{index}"] = f"2026-01-{index + 1:02d}T00:00:00Z"
    manifest = {"archetype_id": "kangaskhan-bouffalant", "splits": {
        "development": {"event_ids": development_ids, "dates": development_dates},
        "validation": {"event_ids": [], "dates": {}},
        "test": {"event_ids": ["t"], "dates": {"t": "2026-06-01T00:00:00Z"}},
    }}
    protocol = {"time": {"primary_embargo_days": 7, "atomic_unit": "event_block"},
                "h1": {"minimum_independent_event_clusters": 20}}
    result = _simulate_locked(str(database), manifest, protocol, tmp_path, time.monotonic(), None)

    assert result["preflight"]["measurement_split_unit"] == "atomic_event_block"
    assert result["preflight"]["pretest_training_date_block_count"] == 10


def test_evaluation_passes_remaining_stage_budget_after_context_build(monkeypatch, tmp_path):
    from src.ingestion.best60_research import experiment, runner

    monkeypatch.setattr(runner, "_extract", lambda *_args, **_kwargs: {
        "pairing_views": [{"event": "t", "won": 1}],
    })
    monkeypatch.setattr(runner, "build_context", lambda *_args, **_kwargs: {"training_event_ids": ["d"]})
    validation_file = tmp_path / "validation.json"
    validation_file.write_text("{}", encoding="utf-8")
    calls = {}

    def evaluate(_context, views, _selection, _protocol, **kwargs):
        calls["views"] = views
        calls.update(kwargs)
        return {"status": "incomplete", "benchmarks": {}}

    monkeypatch.setattr(experiment, "evaluate_experiment", evaluate)
    monkeypatch.setattr(runner.time, "monotonic", lambda: 103.0)
    manifest = {"archetype_id": "kangaskhan-bouffalant", "splits": {
        "development": {"event_ids": ["d"], "dates": {"d": "2026-01-01"}},
        "validation": {"event_ids": ["v"], "dates": {"v": "2026-01-02"}},
        "test": {"event_ids": ["t"], "dates": {"t": "2026-02-01"}},
    }}
    result = _evaluate_locked(
        "unused.db", manifest,
        {"time": {"primary_embargo_days": 7}, "features": {"slot_minimum_lists": 1}},
        None, tmp_path / "evaluation-progress.json", 10, stage_start=100.0,
    )
    assert result["status"] == "incomplete"
    assert calls["max_seconds"] == pytest.approx(7.0)
    assert [row["event"] for row in calls["views"]] == ["t"]


def test_selection_validation_rows_exclude_pretest_embargo_poison_and_exact_utc_boundary():
    nominal_ids = ["nominal-first", "active", "poison-at-boundary", "poison-offset", "test-day"]
    validation_dates = {
        "nominal-first": "2026-09-15T12:00:00+03:00",
        "active": "2026-09-24T23:59:59Z",
        "poison-at-boundary": "2026-09-25T00:00:00Z",
        # Local date is Sep 25, but this is exactly the UTC boundary.
        "poison-offset": "2026-09-25T03:00:00+03:00",
        "test-day": "2026-10-02T00:00:00Z",
    }
    corpus = {"pairing_views": [{"event": event_id, "won": int(event_id.startswith("poison"))}
                                for event_id in nominal_ids]}

    rows, window = _selection_validation_views(
        corpus,
        nominal_ids,
        validation_dates,
        {"t1": "2026-10-02T01:00:00Z", "t2": "2026-10-02T18:00:00Z"},
        7,
    )

    # Only the eligible rows reach validate_experiment's tuning and selection path.
    assert [row["event"] for row in rows] == ["nominal-first", "active"]
    assert window["nominal_event_ids"] == sorted(nominal_ids)
    assert window["active_event_ids"] == ["active", "nominal-first"]
    assert window["embargoed_event_ids"] == ["poison-at-boundary", "poison-offset", "test-day"]
    assert window["boundary_utc_exclusive"] == "2026-09-25T00:00:00+00:00"
    assert window["first_test_date_utc"] == "2026-10-02"


def test_audit_validate_lock_binds_recomputed_selection_embargo(tmp_path, monkeypatch):
    from src.ingestion.best60_research import experiment, runner

    database = _database(tmp_path / "source.db")
    config = tmp_path / "protocol.json"
    config.write_text(json.dumps({
        "schema_version": 1,
        "time": {"primary_embargo_days": 7},
        "features": {"slot_minimum_lists": 1},
        "estimators": {"max_iterations": 5},
    }), encoding="utf-8")
    output = tmp_path / "runs"
    audited = run_stage(_args(database, output, config, "audit"))
    manifest = json.loads((output / audited["run_id"] / "manifest.json").read_text(encoding="utf-8"))
    test_ids = set(manifest["splits"]["test"]["event_ids"])
    received = {}

    monkeypatch.setattr(runner, "build_context", lambda *_args, **_kwargs: {"cutoff": "unused"})

    def validate(_context, rows, _protocol, **_kwargs):
        received["rows"] = rows
        return {"status": "complete", "benchmarks": {}}

    monkeypatch.setattr(experiment, "validate_experiment", validate)
    validated = run_stage(_args(database, output, config, "validate", audited["run_id"]))
    assert validated["status"] == "complete"
    assert {row["event"] for row in received["rows"]}.isdisjoint(test_ids)
    validation_path = output / audited["run_id"] / "validation.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    window = validation["selection_window"]
    assert set(window["active_event_ids"]).isdisjoint(window["embargoed_event_ids"])
    assert set(window["active_event_ids"]) | set(window["embargoed_event_ids"]) == set(window["nominal_event_ids"])
    assert window["training_cutoff_utc"] == min(manifest["splits"]["validation"]["dates"].values())

    locked = run_stage(_args(database, output, config, "lock", audited["run_id"]))
    assert locked["status"] == "complete"

    # The lock stage rejects a stale or altered selection window even if the cache key is retained.
    locked_validation = json.loads(validation_path.read_text(encoding="utf-8"))
    locked_validation["selection_window"]["active_event_ids"] = []
    validation_path.write_text(json.dumps(locked_validation), encoding="utf-8")
    state_path = output / audited["run_id"] / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    import hashlib
    state["validation_sha256"] = hashlib.sha256(validation_path.read_bytes()).hexdigest()
    state_path.write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(RunnerError, match="selection window"):
        run_stage(_args(database, output, config, "lock", audited["run_id"]))


def _protocol(tmp_path):
    config = tmp_path / "protocol.json"
    config.write_text(json.dumps({"schema_version": 1, "time": {"primary_embargo_days": 7},
                                  "features": {"slot_minimum_lists": 1},
                                  "estimators": {"max_iterations": 5}}), encoding="utf-8")
    return config


def test_audit_records_provenance_relative_to_the_working_directory(tmp_path):
    database = _database(tmp_path / "source.db")
    config = _protocol(tmp_path)
    output = tmp_path / "runs"

    result = run_stage(_args(database, output, config, "audit"))

    run = output / result["run_id"]
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    audit = json.loads((run / "audit.json").read_text(encoding="utf-8"))
    for recorded in (manifest["database"], manifest["config"], audit["source"]["source_path"]):
        assert not Path(recorded).is_absolute()
    assert not Path(result["run_directory"]).is_absolute()
    assert Path(result["run_directory"]).name == result["run_id"]


def test_output_directory_is_ignored_before_it_is_created(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".gitignore").write_text("/output/card-value-research/\n", encoding="utf-8")
    subprocess.run(["git", "init", "--quiet", str(repo)], check=True)

    output = repo / "output" / "card-value-research"
    assert not output.exists()
    assert _output_is_ignored(repo, output) is True
    assert _output_is_ignored(repo, repo / "src") is False


def _validate_ready(monkeypatch):
    from src.ingestion.best60_research import experiment, runner

    monkeypatch.setattr(runner, "build_context", lambda *_args, **_kwargs: {"cutoff": "unused"})
    monkeypatch.setattr(experiment, "validate_experiment",
                        lambda *_args, **_kwargs: {"status": "complete", "benchmarks": {}})


def _audit_with_field(tmp_path):
    field = tmp_path / "field-tree" / "field.json"
    field.parent.mkdir()
    field.write_text(json.dumps({"kangaskhan-bouffalant": 1.0}), encoding="utf-8")
    database = _database(tmp_path / "source.db")
    config = _protocol(tmp_path)
    output = tmp_path / "runs"
    audited = run_stage(_args(database, output, config, "audit", field_file=str(field)))
    return field, database, config, output, audited


def test_audit_records_the_field_file_by_name_only(tmp_path):
    field, _database_path, _config_path, output, audited = _audit_with_field(tmp_path)

    manifest = json.loads((output / audited["run_id"] / "manifest.json").read_text(encoding="utf-8"))
    recorded = manifest["field_file"]

    assert recorded == "field.json"
    assert ".." not in recorded
    assert ":" not in recorded
    assert Path(recorded).name == recorded
    assert manifest["field_values"] == {"kangaskhan-bouffalant": 1.0}


def test_resume_finds_the_saved_field_file_by_its_recorded_name(tmp_path, monkeypatch):
    field, database, config, output, audited = _audit_with_field(tmp_path)
    _validate_ready(monkeypatch)
    monkeypatch.chdir(field.parent)

    validated = run_stage(_args(database, output, config, "validate", audited["run_id"]))

    assert validated["status"] == "complete"


def test_resume_without_field_file_asks_for_it_when_the_name_does_not_resolve(tmp_path, monkeypatch):
    _field, database, config, output, audited = _audit_with_field(tmp_path)
    _validate_ready(monkeypatch)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    with pytest.raises(RunnerError) as error:
        run_stage(_args(database, output, config, "validate", audited["run_id"]))

    assert error.value.code == 2
    assert "--field-file" in str(error.value)


def test_resume_checks_a_repassed_field_file_against_the_recorded_hash(tmp_path, monkeypatch):
    field, database, config, output, audited = _audit_with_field(tmp_path)
    _validate_ready(monkeypatch)
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps({"kangaskhan-bouffalant": 0.5}), encoding="utf-8")

    with pytest.raises(RunnerError, match="field file differs from the audited run"):
        run_stage(_args(database, output, config, "validate", audited["run_id"], field_file=str(changed)))

    validated = run_stage(_args(database, output, config, "validate", audited["run_id"], field_file=str(field)))
    assert validated["status"] == "complete"
