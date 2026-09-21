from __future__ import annotations

from typing import Any


def inclusion_rates(store, archetype: str, window_days: int | None = None) -> dict[str, float]:
    return store.card_inclusion(archetype)


def deck_features(store, archetypes: list[str]) -> dict[str, dict[str, float]]:
    return {archetype: inclusion_rates(store, archetype) for archetype in archetypes}


def observed_skeleton(store, archetype: str) -> list[dict[str, object]]:
    return store.observed_skeleton(archetype)
