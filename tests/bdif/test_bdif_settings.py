import pytest

from src.bdif.settings import BdifSettings


def test_from_environment_uses_live_config_defaults(monkeypatch):
    from src.core import config

    monkeypatch.setattr(config, "INPUT_DATA", "custom/input.json")
    monkeypatch.setattr(config, "BDIF_USE_CARD_MODEL", False)
    monkeypatch.setattr(config, "LIMITLESS_INGESTION_ENABLED", False)
    monkeypatch.setattr(config, "BDIF_PANEL_SHARE_THRESHOLD", 0.04)
    monkeypatch.setattr(config, "BDIF_PANEL_MAX_DECKS", 12)
    monkeypatch.setattr(config, "BDIF_PANEL_DECKS", ["Deck A", "Deck B"])
    monkeypatch.setattr(config, "LIMITLESS_BACKFILL_TOURNAMENTS", 250)

    settings = BdifSettings.from_environment()

    assert settings.use_card_model is False
    assert settings.ingestion_enabled is False
    assert settings.baseline_input_path == "custom/input.json"
    assert settings.fallback_panel_decks == ("Deck A", "Deck B")
    assert settings.panel_share_threshold == 0.04
    assert settings.panel_max_decks == 12
    assert settings.backfill_limit == 250
    assert settings.db_path == "data/limitless.db"
    assert settings.ingestion_input_path == "data/input/limitless_input.json"
    assert settings.model_input_path == "data/input/limitless_model_input.json"


def test_from_environment_parses_boolean_aliases(monkeypatch):
    monkeypatch.setenv("BDIF_USE_CARD_MODEL", " YeS ")
    monkeypatch.setenv("LIMITLESS_INGESTION_ENABLED", "OFF")

    settings = BdifSettings.from_environment()

    assert settings.use_card_model is True
    assert settings.ingestion_enabled is False


def test_from_environment_overrides_database_path(monkeypatch):
    monkeypatch.setenv("BDIF_DB_PATH", "custom/store.db")

    assert BdifSettings.from_environment().db_path == "custom/store.db"


@pytest.mark.parametrize("value", ["enabled", "2", "truth"])
def test_from_environment_rejects_invalid_boolean_values(monkeypatch, value):
    monkeypatch.setenv("BDIF_USE_CARD_MODEL", value)

    with pytest.raises(ValueError, match="BDIF_USE_CARD_MODEL"):
        BdifSettings.from_environment()