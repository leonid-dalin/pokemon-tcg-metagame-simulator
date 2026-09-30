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
