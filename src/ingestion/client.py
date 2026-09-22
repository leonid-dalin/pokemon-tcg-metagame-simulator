from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

import requests

API_BASE = "https://play.limitlesstcg.com/api"


class LimitlessResponseError(RuntimeError):
    pass


@dataclass
class LimitlessClient:
    api_key: str | None = None
    base_url: str = API_BASE
    timeout: float = 20.0
    min_delay: float = 1.0
    _last_request: float = field(init=False, repr=False, default=0.0)

    @classmethod
    def from_environment(cls) -> "LimitlessClient":
        return cls(api_key=os.environ.get("LIMITLESS_API_KEY") or None)

    def _get(self, path: str, **params: Any) -> Any:
        wait = self.min_delay - (time.monotonic() - self._last_request)
        if wait > 0:
            time.sleep(wait)
        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["X-Access-Key"] = self.api_key
        self._last_request = time.monotonic()
        for attempt in range(2):
            response = requests.get(
                f"{self.base_url.rstrip('/')}/{path.lstrip('/')}",
                params=params,
                headers=headers,
                timeout=self.timeout,
            )
            if response.status_code != 429 or attempt:
                break
            retry_after = float(response.headers.get("Retry-After", "0"))
            if retry_after:
                time.sleep(min(retry_after, 60.0))
            self._last_request = time.monotonic()
        status_code = response.status_code
        if not 200 <= status_code < 300:
            raise LimitlessResponseError(f"Limitless request failed with HTTP {response.status_code}")
        if len(response.content) > 10_000_000:
            raise LimitlessResponseError("Limitless response exceeded 10 MB")
        try:
            return response.json()
        except (TypeError, ValueError) as exc:
            raise LimitlessResponseError("Limitless response was not valid JSON") from exc

    def _get_list(self, path: str, label: str, **params: Any) -> list[dict[str, Any]]:
        result = self._get(path, **params)
        if not isinstance(result, list):
            raise LimitlessResponseError(f"{label} response was not a list")
        return result

    def tournaments(self, **params: Any) -> list[dict[str, Any]]:
        return self._get_list("tournaments", "tournaments", **params)

    def event_details(self, event_id: str) -> dict[str, Any]:
        result = self._get(f"tournaments/{event_id}/details")
        if not isinstance(result, dict):
            raise LimitlessResponseError("details response was not an object")
        return result

    def standings(self, event_id: str) -> list[dict[str, Any]]:
        return self._get_list(f"tournaments/{event_id}/standings", "standings")

    def pairings(self, event_id: str) -> list[dict[str, Any]]:
        return self._get_list(f"tournaments/{event_id}/pairings", "pairings")

    def game_decks(self, game: str = "PTCG") -> list[dict[str, Any]]:
        return self._get_list(f"games/{game}/decks", "deck catalogue")

    def fetch_event_bundle(self, event_id: str) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
        details = self.event_details(event_id)
        standings = self.standings(event_id)
        pairings = self.pairings(event_id)
        return details, standings, pairings
