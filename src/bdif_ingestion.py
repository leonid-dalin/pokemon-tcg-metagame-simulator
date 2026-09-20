"""Limitless event and decklist ingestion for offline BDIF analysis."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import requests


API_BASE = "https://play.limitlesstcg.com/api"


@dataclass(frozen=True)
class LimitlessClient:
    api_key: str | None = None
    base_url: str = API_BASE
    timeout: float = 20.0

    @classmethod
    def from_environment(cls) -> "LimitlessClient":
        return cls(api_key=os.environ.get("LIMITLESS_API_KEY") or None)

    def _get(self, path: str, **params: Any) -> Any:
        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["X-Access-Key"] = self.api_key
        response = requests.get(
            f"{self.base_url.rstrip('/')}/{path.lstrip('/')}",
            params=params,
            headers=headers,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def tournaments(self, **params: Any) -> Any:
        return self._get("tournaments", **params)

    def event_details(self, event_id: str) -> Any:
        return self._get(f"tournaments/{event_id}/details")

    def standings(self, event_id: str) -> Any:
        return self._get(f"tournaments/{event_id}/standings")


def inclusion_rates(cards: Iterable[str]) -> dict[str, float]:
    cards = list(cards)
    if not cards:
        return {}
    unique_cards = sorted(set(cards))
    return {card: 1.0 / len(unique_cards) for card in unique_cards}


def store_event_snapshot(root: str | Path, event: dict[str, Any]) -> Path:
    """Persist a key-addressable event snapshot without credentials."""
    event_id = str(event["event_id"])
    archetype = str(event["archetype"])
    event_date = str(event["event_date"])
    cards = list(event.get("cards", []))
    record = {
        "event_id": event_id,
        "event_date": event_date,
        "archetype": archetype,
        "cards": cards,
        "inclusion_rates": inclusion_rates(cards),
    }
    destination = Path(root) / event_date / f"{event_id}__{archetype}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2), encoding="utf-8")
    os.replace(temporary, destination)
    return destination
