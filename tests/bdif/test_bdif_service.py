import ast
import hashlib
import os
from pathlib import Path

import numpy as np
import pytest

from src.api.models import PredictionRequest
from src.bdif import service
from src.bdif.settings import BdifSettings
from src.core import config
from src.ingestion.model import CardModelNotIdentifiable, PlayerObservation


def settings(**overrides):
    values = dict(use_card_model=True, ingestion_enabled=False, db_path="data/limitless.db", baseline_input_path="input.json", ingestion_input_path="data/input/limitless_input.json", model_input_path="data/input/limitless_model_input.json", panel_share_threshold=0.01, panel_max_decks=10, fallback_panel_decks=("fallback",), backfill_limit=10)
    values.update(overrides)
    return BdifSettings(**values)


@pytest.mark.unit
def test_panel_decks_map_limitless_ids_to_simulation_names(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "limitless.db").touch()
    monkeypatch.setattr("src.ingestion.model.select_panel_decks", lambda *args, **kwargs: ["alakazam-dudunsparce", "n-zoroark", "unknown-deck"])
    class Store:
        def deck_weights(self):
            return {"a": 1}
    assert service.panel_decks_for_report(["Alakazam Dudunsparce", "N's Zoroark"], Store(), settings()) == ["Alakazam Dudunsparce", "N's Zoroark"]


@pytest.mark.unit
def test_panel_decks_for_report_passes_configured_maximum(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "limitless.db").touch()
    calls = []

    def select_panel_decks(*args, **kwargs):
        calls.append(kwargs)
        return ["a"]

    monkeypatch.setattr("src.ingestion.model.select_panel_decks", select_panel_decks)

    class Store:
        def deck_weights(self):
            return {"a": 0.5, "b": 0.3, "c": 0.2}

    service.panel_decks_for_report(["a"], Store(), settings(panel_max_decks=1))

    assert calls == [{"threshold": 0.01, "max_decks": 1}]


@pytest.mark.unit
def test_panel_deck_resolution_strips_unique_suffixes_and_preserves_compounds():
    assert service.map_panel_decks_to_matrix(["slowking-scr", "n-zoroark", "dragapult-dusknoir"], ["Slowking", "N's Zoroark", "Dragapult", "Dragapult Dusknoir"]) == ["Slowking", "N's Zoroark", "Dragapult Dusknoir"]


@pytest.mark.unit
def test_panel_deck_resolution_drops_ambiguous_and_missing_ids(monkeypatch):
    dropped = []
    class Logger:
        def warning(self, event, **kwargs):
            dropped.extend(kwargs["deck_ids"])
    monkeypatch.setattr(service, "logger", Logger())
    assert service.map_panel_decks_to_matrix(["foo-bar-baz", "missing-deck"], ["Foo Bar", "Foo  Bar"]) == []
    assert dropped == ["foo-bar-baz", "missing-deck"]


@pytest.mark.unit
def test_addons_report_non_identifiable_for_each_deck(monkeypatch, tmp_path):
    class Store:
        def prepare_for_read(self): pass
        def deck_weights(self): return {"a": 0.5, "b": 0.5}
        def player_observations(self):
            return [PlayerObservation("a", "b", frozenset({"Tech"}), frozenset(), 1)] * 2 + [PlayerObservation("a", "b", frozenset(), frozenset({"Tech"}), 0)] * 2
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "limitless.db").touch()
    monkeypatch.setattr("src.ingestion.model.fit_card_model", lambda *args: (_ for _ in ()).throw(CardModelNotIdentifiable("rank")))
    assert service.build_report_addons(Store(), settings()) == ({"a": {"status": "not identifiable", "reason": "rank"}, "b": {"status": "not identifiable", "reason": "rank"}}, {})


@pytest.mark.unit
def test_addons_skip_store_when_feature_disabled(monkeypatch):
    monkeypatch.setattr(os.path, "exists", lambda path: (_ for _ in ()).throw(AssertionError(path)))
    assert service.build_report_addons(settings=settings(use_card_model=False)) == ({}, {})


@pytest.mark.unit
def test_addons_prepare_supplied_store_before_reading(monkeypatch, tmp_path):
    calls = []
    class Store:
        def prepare_for_read(self): calls.append("prepare")
        def deck_weights(self): calls.append("deck_weights"); return {}
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "limitless.db").touch()
    assert service.build_report_addons(Store(), settings()) == ({}, {})
    assert calls == ["prepare", "deck_weights"]


@pytest.mark.unit
def test_prediction_reuses_one_store_and_isolates_addon_failure(monkeypatch):
    created, panels, addons, passed = [], [], [], []
    class Store:
        def __init__(self, path): created.append(self)
    monkeypatch.setattr(service, "open_store", lambda cfg: Store(cfg.db_path))
    monkeypatch.setattr(service, "load_matchup_data", lambda *args: (["a", "b"], np.array([[.5,.5],[.5,.5]]), {}))
    monkeypatch.setattr(service, "predict_best_decks", lambda request: {"full_meta": {"a": .5, "b": .5}})
    monkeypatch.setattr(service, "swiss_rounds_from_players", lambda players: 1)
    monkeypatch.setattr(service, "panel_decks_for_report", lambda names, store, cfg: panels.append(store) or [])
    monkeypatch.setattr(service, "build_report_addons", lambda store, cfg: addons.append(store) or (_ for _ in ()).throw(RuntimeError("addon failed")))
    monkeypatch.setattr(service, "run_monte_carlo_analytics", lambda **kwargs: passed.append(kwargs) or {"metrics": {}, "ranked_metrics": {}, "insufficient_data": [], "posterior": {"draws": 0, "interval_status": "posterior disabled"}, "matchup_panel": {"rows": {}, "unmatched": [], "opponents": []}})
    monkeypatch.setattr(service, "build_bdif_report", lambda result, *args: result)
    request = PredictionRequest(job_id="job", deck_names=["a", "b"], matchup_matrix=[[.5,.5],[.5,.5]], total_players=4)
    service.run_prediction(request, settings=settings())
    assert len(created) == 1
    assert panels == addons == [created[0]]
    assert "best60_recommendations" not in passed[0]


@pytest.mark.unit
def test_prediction_passes_matchup_details_and_report_addons(monkeypatch):
    details = {("a", "b"): {"win_rate": .6, "match_count": 10}}
    seen, reports = [], []
    monkeypatch.setattr(service, "load_matchup_data", lambda *args: (["a", "b"], np.array([[.5,.6],[.4,.5]]), details))
    monkeypatch.setattr(service, "predict_best_decks", lambda request: {"full_meta": {"a": .5, "b": .5}})
    monkeypatch.setattr(service, "swiss_rounds_from_players", lambda players: 1)
    monkeypatch.setattr(service, "open_store", lambda cfg: None)
    monkeypatch.setattr(service, "panel_decks_for_report", lambda *args: [])
    monkeypatch.setattr(service, "build_report_addons", lambda *args: ({"best": 1}, {"h1": 2}))
    monkeypatch.setattr(service, "run_monte_carlo_analytics", lambda **kwargs: seen.append(kwargs) or {})
    monkeypatch.setattr(service, "build_bdif_report", lambda result, *args: reports.append(args[:2]) or result)
    request = PredictionRequest(job_id="job", deck_names=["a", "b"], matchup_matrix=[[.5,.6],[.4,.5]], total_players=4)
    service.run_prediction(request, settings=settings())
    assert seen[0]["matchup_details"] == details
    assert reports == [({"best": 1}, {"h1": 2})]


@pytest.mark.unit
def test_prediction_without_a_seed_uses_the_configured_seed(monkeypatch):
    seen = []
    monkeypatch.setattr(service, "load_matchup_data", lambda *args: (["a", "b"], np.array([[.5, .6], [.4, .5]]), {}))
    monkeypatch.setattr(service, "predict_best_decks", lambda request: {"full_meta": {"a": .5, "b": .5}})
    monkeypatch.setattr(service, "swiss_rounds_from_players", lambda players: 1)
    monkeypatch.setattr(service, "open_store", lambda cfg: None)
    monkeypatch.setattr(service, "panel_decks_for_report", lambda *args: [])
    monkeypatch.setattr(service, "build_report_addons", lambda *args: ({}, {}))
    monkeypatch.setattr(service, "run_monte_carlo_analytics", lambda **kwargs: seen.append(kwargs["seed"]) or {})
    monkeypatch.setattr(service, "build_bdif_report", lambda result, *args: result)
    monkeypatch.setattr(config, "RNG_SEED", 4242)
    request = PredictionRequest(deck_names=["a", "b"], matchup_matrix=[[.5, .6], [.4, .5]], total_players=4)

    service.run_prediction(request, settings=settings())

    assert seen == [4242]


@pytest.mark.unit
def test_prediction_reports_its_provenance(monkeypatch, tmp_path):
    source = tmp_path / "input.json"
    source.write_bytes(b"matrix bytes")
    monkeypatch.setattr(service, "load_matchup_data", lambda *args: (["a", "b"], np.array([[.5, .6], [.4, .5]]), {}))
    monkeypatch.setattr(service, "predict_best_decks", lambda request: {"full_meta": {"a": .5, "b": .5}})
    monkeypatch.setattr(service, "swiss_rounds_from_players", lambda players: 1)
    monkeypatch.setattr(service, "open_store", lambda cfg: None)
    monkeypatch.setattr(service, "panel_decks_for_report", lambda *args: ["a"])
    monkeypatch.setattr(service, "build_report_addons", lambda *args: ({}, {}))
    monkeypatch.setattr(service, "run_monte_carlo_analytics", lambda **kwargs: {"posterior": {"draws": 40}})
    monkeypatch.setattr(service, "build_bdif_report", lambda result, best, h1, provenance: provenance)
    request = PredictionRequest(deck_names=["a", "b"], matchup_matrix=[[.5, .6], [.4, .5]], total_players=4, precision_tier="1 - BULLET")

    provenance = service.run_prediction(request, settings=settings(use_card_model=False), seed=77, input_path=str(source))["mc_results"]

    assert provenance == {
        "input_path": str(source),
        "input_sha256": hashlib.sha256(b"matrix bytes").hexdigest(),
        "seed": 77,
        "iterations": 1_000,
        "posterior_draws": 40,
        "use_card_model": False,
        "panel_decks": ["a"],
    }


@pytest.mark.unit
def test_prediction_uses_explicit_seed(monkeypatch):
    seen = []
    monkeypatch.setattr(service, "load_matchup_data", lambda *args: (["a", "b"], np.array([[.5,.6],[.4,.5]]), {}))
    monkeypatch.setattr(service, "predict_best_decks", lambda request: {"full_meta": {"a": .5, "b": .5}})
    monkeypatch.setattr(service, "swiss_rounds_from_players", lambda players: 1)
    monkeypatch.setattr(service, "open_store", lambda cfg: None)
    monkeypatch.setattr(service, "panel_decks_for_report", lambda *args: [])
    monkeypatch.setattr(service, "build_report_addons", lambda *args: ({}, {}))
    monkeypatch.setattr(service, "run_monte_carlo_analytics", lambda **kwargs: seen.append(kwargs) or {})
    monkeypatch.setattr(service, "build_bdif_report", lambda result, *args: result)

    request = PredictionRequest(job_id="job", deck_names=["a", "b"], matchup_matrix=[[.5,.6],[.4,.5]], total_players=4)
    service.run_prediction(request, settings=settings(), seed=9876)

    assert seen[0]["seed"] == 9876


@pytest.mark.unit
def test_bdif_status_reads_legacy_store_without_writing(monkeypatch, tmp_path):
    db = tmp_path / "legacy.db"
    db.touch()
    calls = []
    class Store:
        def __init__(self, path): calls.append("open")
        def prepare_for_read(self): calls.append("prepare")
        def deck_weights(self): calls.append("weights"); return {"a": 1}
        def player_observations(self): calls.append("observations"); return []
    monkeypatch.setattr("src.ingestion.store.LimitlessStore", Store)
    assert service.bdif_status(settings(db_path=str(db))) == {"status": "available", "db_path": str(db), "decks": 1, "observations": 0}
    assert calls == ["open", "prepare", "weights", "observations"]


@pytest.mark.unit
def test_service_has_no_worker_or_ui_imports():
    tree = ast.parse(Path("src/bdif/service.py").read_text())
    imported = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    imported += [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names]
    assert not any(name and (name.startswith("src.worker") or name.startswith("src.ui")) for name in imported)


@pytest.mark.unit
def test_requested_panel_overrides_automatic_selection(monkeypatch):
    received = []
    monkeypatch.setattr(service, "open_store", lambda settings: None)
    monkeypatch.setattr(service, "panel_decks_for_report", lambda *args: pytest.fail("automatic selection ran"))
    monkeypatch.setattr(service, "load_matchup_data", lambda *args: (["a", "b"], np.array([[.5, .6], [.4, .5]]), {}))
    monkeypatch.setattr(service, "predict_best_decks", lambda request: {"full_meta": {"a": .5, "b": .5}})
    monkeypatch.setattr(service, "swiss_rounds_from_players", lambda players: 1)
    monkeypatch.setattr(service, "build_report_addons", lambda *args: ({}, {}))
    monkeypatch.setattr(service, "run_monte_carlo_analytics", lambda **kwargs: received.append(kwargs) or {})
    monkeypatch.setattr(service, "build_bdif_report", lambda result, *args: result)

    request = PredictionRequest(
        deck_names=["a", "b"],
        matchup_matrix=[[.5, .6], [.4, .5]],
        bdif_panel_decks=["b"],
        total_players=4,
    )
    service.run_prediction(request, settings=settings())



@pytest.mark.unit
def test_ingestion_paginates_and_skips_stored_events(monkeypatch, tmp_path):
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

        def backfill_deck_names(self, names):
            pass

        def existing_tournament_ids(self):
            return set(stored_ids)

        def upsert_tournament(self, event, details):
            stored_ids.add(event["id"])

        def upsert_standings(self, event_id, standings):
            assert standings == []

        def upsert_pairings(self, event_id, rows):
            pairings.extend(rows)

        def player_observations(self):
            return []

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("src.ingestion.client.LimitlessClient", Client)
    monkeypatch.setattr("src.ingestion.store.LimitlessStore", Store)
    monkeypatch.setattr("src.ingestion.aggregate.build_artifact", lambda store: {"archetypes": []})

    first = service.run_ingestion(settings(ingestion_enabled=True, backfill_limit=200))
    second = service.run_ingestion(settings(ingestion_enabled=True, backfill_limit=200))

    assert fetched == ["fresh"]
    assert pairings == [{"player1": "p1", "player2": "p2"}]
    assert first["skipped_events"] == 1
    assert second["skipped_events"] == 2
