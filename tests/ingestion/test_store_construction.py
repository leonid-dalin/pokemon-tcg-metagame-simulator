import os
import subprocess
import sys
import warnings
from pathlib import Path

import pytest

from src.ingestion.store import LimitlessStore


@pytest.mark.unit
def test_store_construction_does_not_read_the_simulation_input(monkeypatch, tmp_path):
    (tmp_path / "data" / "input").mkdir(parents=True)
    (tmp_path / "data" / "input" / "ea_input.json").write_text("not json", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    with pytest.warns(UserWarning, match="canonical_names is ignored"):
        store = LimitlessStore(tmp_path / "limitless.db", canonical_names=["Known"], deck_mapping={})

    assert store.path == str(tmp_path / "limitless.db")
    assert not (tmp_path / "limitless.db").exists()


@pytest.mark.unit
def test_store_warns_that_canonical_names_is_ignored(tmp_path):
    with pytest.warns(UserWarning, match="canonical_names is ignored"):
        LimitlessStore(tmp_path / "limitless.db", canonical_names=["Known"], deck_mapping={})


@pytest.mark.unit
def test_store_without_canonical_names_does_not_warn(tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        store = LimitlessStore(tmp_path / "limitless.db", deck_mapping={})

    assert store.deck_mapping == {}


@pytest.mark.unit
def test_store_warning_is_visible_to_an_ordinary_python_caller(tmp_path):
    root = Path(__file__).resolve().parents[2]
    script = "from src.ingestion.store import LimitlessStore; import sys; LimitlessStore(sys.argv[1], canonical_names=['Known'], deck_mapping={})"
    environment = dict(os.environ, PYTHONPATH=str(root))
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path / "limitless.db")],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "UserWarning: canonical_names is ignored" in result.stderr
