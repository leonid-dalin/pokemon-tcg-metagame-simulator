import json
import logging
import sys
from types import SimpleNamespace

import pytest

from src.bdif import cli


def _run(monkeypatch, capsys, argv, module):
    monkeypatch.setattr(sys, "argv", ["python -m src.bdif", *argv])
    from src.core import logger as logger_module
    monkeypatch.setattr(logger_module, "setup_structured_logging", lambda: None)
    monkeypatch.setattr(logger_module.logger, "error", lambda *args, **kwargs: print('{"event": "bdif_command_failed"}', file=sys.stderr))
    monkeypatch.setattr(cli, "setup_structured_logging", lambda: None)
    monkeypatch.setattr(cli, "setup_telemetry", lambda name: None)
    monkeypatch.setitem(sys.modules, "src.bdif.service", module)
    return_code = cli.main()
    return return_code, capsys.readouterr()


@pytest.mark.unit
@pytest.mark.parametrize(("command", "method", "result"), [
    ("status", "bdif_status", {"status": "missing"}),
    ("ingest", "run_ingestion", {"status": "disabled"}),
    ("refit", "refit_card_model", {"status": "complete"}),
])
def test_module_commands_write_json_only_to_stdout(monkeypatch, capsys, command, method, result):
    calls = []
    module = SimpleNamespace(**{method: lambda: calls.append(method) or result})
    code, captured = _run(monkeypatch, capsys, [command], module)
    assert code == 0
    assert calls == [method]
    assert json.loads(captured.out) == result
    assert captured.err == ""


@pytest.mark.unit
@pytest.mark.parametrize(("command", "method"), [
    ("status", "bdif_status"),
    ("ingest", "run_ingestion"),
    ("refit", "refit_card_model"),
])
def test_module_commands_use_current_service_module(monkeypatch, capsys, command, method):
    import src.bdif as package

    stale = SimpleNamespace(**{method: lambda: pytest.fail("stale service module used")})
    current_calls = []
    current = SimpleNamespace(**{method: lambda: current_calls.append(method) or {"status": "current"}})
    monkeypatch.setattr(package, "service", stale, raising=False)
    monkeypatch.setitem(sys.modules, "src.bdif.service", current)

    code, captured = _run(monkeypatch, capsys, [command], current)

    assert code == 0
    assert current_calls == [method]
    assert json.loads(captured.out) == {"status": "current"}
    assert captured.err == ""


@pytest.mark.unit
def test_report_uses_local_input_and_bounded_request(monkeypatch, tmp_path, capsys):
    source = tmp_path / "matrix.json"
    source.write_text("{}", encoding="utf-8")
    output = tmp_path / "new-output"
    requests = []
    class Request:
        def __init__(self, **kwargs):
            requests.append(kwargs)
            self.__dict__.update(kwargs)
    monkeypatch.setattr(cli, "PredictionRequest", Request)
    monkeypatch.setattr("src.core.data.load_matchup_data", lambda path: (["A", "B"], __import__("numpy").array([[0.5, 0.6], [0.4, 0.5]]), {}))
    module = SimpleNamespace(run_prediction=lambda req, **kwargs: {"report": req.total_players})
    code, captured = _run(monkeypatch, capsys, ["report", "--input", str(source), "--output", str(output), "-P", "64", "--seed", "8", "--meta", "Pikachu:0.2"], module)
    assert code == 0
    assert output.is_dir()
    assert requests[0]["total_players"] == 64
    assert requests[0]["user_meta_spec"] == {"Pikachu": 0.2}
    assert json.loads(captured.out) == {"report": 64}
    assert captured.err == ""


@pytest.mark.unit
def test_report_passes_seed_to_service(monkeypatch, tmp_path, capsys):
    source = tmp_path / "matrix.json"
    source.write_text("{}", encoding="utf-8")
    seeds = []
    monkeypatch.setattr("src.core.data.load_matchup_data", lambda path: (["A", "B"], __import__("numpy").array([[0.5, 0.6], [0.4, 0.5]]), {}))
    module = SimpleNamespace(run_prediction=lambda request, **kwargs: seeds.append(kwargs["seed"]) or {"report": request.total_players})

    code, captured = _run(monkeypatch, capsys, ["report", "--input", str(source), "--seed", "9876"], module)

    assert code == 0
    assert seeds == [9876]
    assert json.loads(captured.out) == {"report": 256}


@pytest.mark.unit
def test_healthy_report_with_empty_insufficient_data_exits_zero(monkeypatch, tmp_path, capsys):
    source = tmp_path / "matrix.json"
    source.write_text("{}", encoding="utf-8")
    monkeypatch.setattr("src.core.data.load_matchup_data", lambda path: (["A", "B"], __import__("numpy").array([[0.5, 0.6], [0.4, 0.5]]), {}))
    module = SimpleNamespace(run_prediction=lambda req, **kwargs: {
        "solver_results": {"full_meta": {"A": 0.5, "B": 0.5}},
        "mc_results": {
            "metrics": {"A": {"win_rate": 0.5}},
            "ranked_metrics": {"A": {"win_rate": 0.5}},
            "insufficient_data": [],
            "posterior": {"draws": 10, "interval_status": "ok"},
            "field_posterior": {},
        },
    })
    code, captured = _run(monkeypatch, capsys, ["report", "--input", str(source)], module)
    assert code == 0
    assert json.loads(captured.out)["mc_results"]["insufficient_data"] == []


@pytest.mark.unit
@pytest.mark.parametrize("argv", [
    ["report", "--input", "absent.json"],
    ["report", "-P", "3"],
    ["report", "--meta", "bad"],
    ["report", "--meta", "Deck:nope"],
    ["report", "--tournament-style", "other"],
])
def test_invalid_report_arguments_exit_two(monkeypatch, capsys, argv):
    with pytest.raises(SystemExit) as error:
        _run(monkeypatch, capsys, argv, SimpleNamespace())
    assert error.value.code == 2


@pytest.mark.unit
def test_service_failure_exits_one_with_structured_stderr(monkeypatch, capsys):
    code, captured = _run(monkeypatch, capsys, ["status"], SimpleNamespace(bdif_status=lambda: (_ for _ in ()).throw(RuntimeError("failed"))))
    assert code == 1
    assert captured.out == ""
    assert '"event": "bdif_command_failed"' in captured.err


@pytest.mark.unit
def test_unavailable_evidence_exits_three(monkeypatch, capsys, tmp_path):
    source = tmp_path / "matrix.json"
    source.write_text("{}", encoding="utf-8")
    monkeypatch.setattr("src.core.data.load_matchup_data", lambda path: (["A", "B"], __import__("numpy").array([[0.5, 0.6], [0.4, 0.5]]), {}))
    module = SimpleNamespace(run_prediction=lambda req, **kwargs: {
        "solver_results": {},
        "mc_results": {
            "metrics": {},
            "ranked_metrics": {},
            "insufficient_data": ["A"],
            "posterior": {"draws": 0, "interval_status": "ok"},
            "field_posterior": {},
            "matchup_panel": {},
            "best60_recommendations": {},
            "h1_report": {},
        },
    })
    code, captured = _run(monkeypatch, capsys, ["report", "--input", str(source)], module)
    assert code == 3
    output = json.loads(captured.out)
    assert output["mc_results"]["insufficient_data"] == ["A"]


@pytest.mark.unit
def test_report_panel_argument_sets_requested_decks(monkeypatch, tmp_path, capsys):
    source = tmp_path / "matrix.json"
    source.write_text("{}", encoding="utf-8")
    requests = []

    class Request:
        def __init__(self, **kwargs):
            requests.append(kwargs)
            self.__dict__.update(kwargs)

    monkeypatch.setattr(cli, "PredictionRequest", Request)
    monkeypatch.setattr("src.core.data.load_matchup_data", lambda path: (["A", "B"], __import__("numpy").array([[0.5, 0.6], [0.4, 0.5]]), {}))
    module = SimpleNamespace(run_prediction=lambda req, **kwargs: {"report": req.bdif_panel_decks})

    code, captured = _run(monkeypatch, capsys, ["report", "--input", str(source), "--panel", "A,B"], module)

    assert code == 0
    assert requests[0]["bdif_panel_decks"] == ["A", "B"]
    assert json.loads(captured.out) == {"report": ["A", "B"]}
