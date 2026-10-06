import os

from dataclasses import dataclass

from src.core import config
from src.core.logger import logger


def _environment_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default

    normalised = value.strip().lower()
    if normalised in {"1", "true", "yes", "on"}:
        return True
    if normalised in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value, got {value!r}")


@dataclass(frozen=True)
class BdifSettings:
    use_card_model: bool
    ingestion_enabled: bool
    db_path: str
    baseline_input_path: str
    ingestion_input_path: str
    model_input_path: str
    panel_share_threshold: float
    panel_max_decks: int
    fallback_panel_decks: tuple[str, ...]
    backfill_limit: int
    model_cache_path: str = "data/input/limitless_model_fit.json"
    best60_list_mode: str = config.BDIF_BEST60_LIST_MODE

    @classmethod
    def from_environment(cls) -> "BdifSettings":
        artifact_dir = (os.environ.get("BDIF_ARTIFACT_DIR") or "data/input").rstrip("/\\")
        return cls(
            use_card_model=_environment_flag(
                "BDIF_USE_CARD_MODEL", config.BDIF_USE_CARD_MODEL
            ),
            ingestion_enabled=_environment_flag(
                "LIMITLESS_INGESTION_ENABLED", config.LIMITLESS_INGESTION_ENABLED
            ),
            db_path=os.environ.get("BDIF_DB_PATH") or "data/limitless.db",
            baseline_input_path=config.INPUT_DATA,
            ingestion_input_path=f"{artifact_dir}/limitless_input.json",
            model_input_path=f"{artifact_dir}/limitless_model_input.json",
            model_cache_path=os.environ.get("BDIF_MODEL_CACHE_PATH") or f"{artifact_dir}/limitless_model_fit.json",
            panel_share_threshold=config.BDIF_PANEL_SHARE_THRESHOLD,
            panel_max_decks=config.BDIF_PANEL_MAX_DECKS,
            fallback_panel_decks=tuple(config.BDIF_PANEL_DECKS),
            backfill_limit=config.LIMITLESS_BACKFILL_TOURNAMENTS,
            best60_list_mode=os.environ.get("BDIF_BEST60_LIST_MODE") or config.BDIF_BEST60_LIST_MODE,
        )


def simulation_input_path(settings: BdifSettings | None = None) -> str:
    settings = settings or BdifSettings.from_environment()
    if settings.use_card_model and os.path.exists(settings.model_input_path):
        return settings.model_input_path
    if settings.use_card_model:
        logger.warning("card_model_artifact_missing", path=settings.model_input_path)
    return settings.baseline_input_path
