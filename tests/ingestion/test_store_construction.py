import warnings

import pytest

from src.ingestion.store import LimitlessStore


@pytest.mark.unit
def test_store_construction_does_not_read_the_simulation_input(monkeypatch, tmp_path):
    (tmp_path / "data" / "input").mkdir(parents=True)
    (tmp_path / "data" / "input" / "ea_input.json").write_text("not json", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    store = LimitlessStore(tmp_path / "limitless.db", canonical_names=["Known"], deck_mapping={})

    assert store.path == str(tmp_path / "limitless.db")
    assert not (tmp_path / "limitless.db").exists()


@pytest.mark.unit
def test_store_warns_that_canonical_names_is_ignored(tmp_path):
    with pytest.warns(DeprecationWarning, match="canonical_names is ignored"):
        LimitlessStore(tmp_path / "limitless.db", canonical_names=["Known"], deck_mapping={})


@pytest.mark.unit
def test_store_without_canonical_names_does_not_warn(tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        store = LimitlessStore(tmp_path / "limitless.db", deck_mapping={})

    assert store.deck_mapping == {}
