from __future__ import annotations

from typing import Any


def inclusion_rates(store, archetype: str) -> dict[str, float]:
    return store.card_inclusion(archetype)


def deck_features(store, archetypes: list[str]) -> dict[str, dict[str, float]]:
    return {archetype: inclusion_rates(store, archetype) for archetype in archetypes}
