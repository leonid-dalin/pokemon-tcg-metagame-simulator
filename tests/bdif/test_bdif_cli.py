import json
import sys
from types import SimpleNamespace

import pytest

from src.ui import cli


def _prepare_cli(monkeypatch, argv, module):
    monkeypatch.setattr("sys.argv", ["tcg-cli", *argv])
    monkeypatch.setattr(cli, "setup_structured_logging", lambda: None)
    monkeypatch.setattr(cli, "setup_telemetry", lambda name: None)
    monkeypatch.setitem(sys.modules, "src.bdif.service", module)


@pytest.mark.unit
def test_bdif_status_command_prints_service_result(monkeypatch, capsys):
    _prepare_cli(monkeypatch, ["--bdif-status"], SimpleNamespace(bdif_status=lambda: {"status": "missing", "db_path": "local.db"}))

    cli.main()

    assert json.loads(capsys.readouterr().out) == {"status": "missing", "db_path": "local.db"}


@pytest.mark.unit
@pytest.mark.parametrize(("option", "method", "result"), [
    ("--bdif-ingest", "run_ingestion", {"status": "disabled"}),
    ("--bdif-refit", "refit_card_model", {"status": "complete", "path": "model.json"}),
])
def test_bdif_mutating_commands_call_service_without_network(monkeypatch, capsys, option, method, result):
    calls = []
    module = SimpleNamespace(**{method: lambda: calls.append(method) or result})
    _prepare_cli(monkeypatch, [option], module)

    cli.main()

    assert calls == [method]
    assert json.loads(capsys.readouterr().out) == result


@pytest.mark.unit
def test_bdif_commands_reject_multiple_operations(monkeypatch):
    _prepare_cli(monkeypatch, ["--bdif-status", "--bdif-refit"], SimpleNamespace())

    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
