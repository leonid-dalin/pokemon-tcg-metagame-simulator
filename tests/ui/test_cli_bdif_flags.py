import sys
from types import SimpleNamespace

import pytest

from src.ui import cli


@pytest.mark.unit
@pytest.mark.parametrize("option", ["--bdif-status", "--bdif-ingest", "--bdif-refit"])
def test_legacy_cli_rejects_bdif_commands(monkeypatch, option):
    calls = []
    module = SimpleNamespace(
        bdif_status=lambda: calls.append("status"),
        run_ingestion=lambda: calls.append("ingest"),
        refit_card_model=lambda: calls.append("refit"),
    )
    monkeypatch.setattr("sys.argv", ["tcg-cli", option])
    monkeypatch.setattr(cli, "setup_structured_logging", lambda: None)
    monkeypatch.setattr(cli, "setup_telemetry", lambda name: None)
    monkeypatch.setitem(sys.modules, "src.bdif.service", module)

    with pytest.raises(SystemExit) as raised:
        cli.main()

    assert raised.value.code == 2
    assert calls == []
