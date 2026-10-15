from __future__ import annotations

import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import structlog

from src.ingestion.store import LimitlessStore

logger = structlog.get_logger()

API_BASE = "https://mew.limitlesstcg.com/labs/data/tcg/"
SOURCE = "Limitless Labs"
DECK_SIZE = 60
DEFAULT_MIN_DELAY = 1.0
REQUEST_ATTEMPTS = 4
_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")


@dataclass(frozen=True)
class LabsEvent:
    labs_id: str
    public_id: str
    city: str
    date: str

    @property
    def name(self) -> str:
        return f"Regional {self.city}"

    @property
    def source_url(self) -> str:
        return f"https://limitlesstcg.com/tournaments/{self.public_id}"

    @classmethod
    def parse(cls, spec: str) -> "LabsEvent":
        parts = [part.strip() for part in spec.split(":", 3)]
        if len(parts) != 4 or not all(parts):
            raise ValueError(f"event spec must be LABS_ID:PUBLIC_ID:CITY:ISO_DATE, got {spec!r}")
        labs_id, public_id, city, date = parts
        if not _DATE_RE.match(date):
            raise ValueError(f"event date must be ISO-8601, got {date!r}")
        return cls(labs_id=labs_id, public_id=public_id, city=city, date=date)


@dataclass
class _RateLimiter:
    """One shared budget, so concurrent workers cannot exceed the configured request rate."""

    min_delay: float
    _lock: Any = field(init=False, repr=False, default=None)
    _next_allowed: float = field(init=False, repr=False, default=0.0)

    def __post_init__(self) -> None:
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            delay = self._next_allowed - now
            self._next_allowed = max(now, self._next_allowed) + self.min_delay
        if delay > 0:
            time.sleep(delay)


@dataclass
class LabsClient:
    min_delay: float = DEFAULT_MIN_DELAY
    attempts: int = REQUEST_ATTEMPTS
    workers: int = 4
    _limiter: "_RateLimiter" = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._limiter = _RateLimiter(self.min_delay)

    def _get_json(self, url: str) -> Any:
        last: Exception | None = None
        for attempt in range(self.attempts):
            self._limiter.wait()
            try:
                request = Request(url, headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"})
                with urlopen(request, timeout=60) as response:
                    return json.loads(response.read())
            except (URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
                last = exc
                time.sleep(2.0 * (attempt + 1))
        raise RuntimeError(f"Labs request failed after {self.attempts} attempts: {url} ({last})")

    def _api(self, route: str, **params: Any) -> Any:
        payload = self._get_json(API_BASE + route + "?" + urlencode(params))
        if not payload.get("ok"):
            raise RuntimeError(f"Labs {route} failed: {payload.get('message')}")
        return payload.get("message")

    def tournament(self, labs_id: str, division: str = "MA") -> dict[str, Any]:
        return self._api("tournament", id=labs_id, division=division)

    def pairings(self, labs_id: str, round_number: int, division: str = "MA") -> list[dict[str, Any]]:
        return self._api("pairings", tournamentId=labs_id, division=division, round=str(round_number)) or []

    def decklist(self, labs_id: str, tp_id: int | str) -> dict[str, Any] | None:
        """Keyed by the tournament participant id; the stable player id returns null."""

        return self._api("decklist", tournamentId=labs_id, playerId=str(tp_id))

    def standings(self, labs_id: str, division: str = "MA") -> list[dict[str, Any]]:
        return self._api("standings", tournamentId=labs_id, division=division) or []


def _write_json(path: Path, value: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _read_json(path: Path, fallback: Any) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else fallback


def collect_event(client: LabsClient, event: LabsEvent, directory: Path) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=True)
    meta_path = directory / "metadata.json"
    standings_path = directory / "standings.json"
    pairings_path = directory / "pairings.json"
    decklists_path = directory / "decklists.json"

    metadata = _read_json(meta_path, None)
    if metadata is None:
        metadata = client.tournament(event.labs_id)
        _write_json(meta_path, metadata)
    logger.info("labs_metadata_collected", labs_id=event.labs_id, players=metadata.get("players"), rounds=metadata.get("round"))

    standings = _read_json(standings_path, None)
    if standings is None:
        standings = client.standings(event.labs_id)
        _write_json(standings_path, standings)
    logger.info("labs_standings_collected", labs_id=event.labs_id, rows=len(standings))

    pairings = _read_json(pairings_path, [])
    if not pairings or not all("round" in row and "phase" in row for row in pairings):
        pairings = []
        for round_number in range(1, int(metadata["round"]) + 1):
            rows = client.pairings(event.labs_id, round_number)
            pairings.extend({"round": round_number, "phase": 1, **row} for row in rows)
            _write_json(pairings_path, pairings)
            logger.info("labs_pairings_collected", labs_id=event.labs_id, round=round_number, total=len(pairings))

    decklists = _read_json(decklists_path, {})
    pending = [row for row in standings if str(row["player_id"]) not in decklists]
    if pending:
        with ThreadPoolExecutor(max_workers=client.workers) as pool:
            futures = {pool.submit(client.decklist, event.labs_id, row["tp_id"]): row for row in pending}
            for done, future in enumerate(as_completed(futures), 1):
                row = futures[future]
                key = str(row["player_id"])
                try:
                    decklists[key] = future.result()
                except RuntimeError as exc:
                    decklists[key] = None
                    logger.error("labs_decklist_failed", labs_id=event.labs_id, player_id=key, error=str(exc))
                if done % 25 == 0 or done == len(pending):
                    _write_json(decklists_path, decklists)
                    logger.info("labs_decklists_collected", labs_id=event.labs_id, done=len(decklists), total=len(standings))
    _write_json(decklists_path, decklists)

    bundle = {
        "labs_id": event.labs_id,
        "public_id": event.public_id,
        "city": event.city,
        "date": event.date,
        "metadata": metadata,
        "standings": standings,
        "pairings": pairings,
        "decklists": decklists,
    }
    _write_json(directory / "bundle.json", bundle)
    return bundle


def decklist_size(decklist: Any) -> int | None:
    if not isinstance(decklist, dict):
        return None
    total = 0
    for group in decklist.values():
        if not isinstance(group, list):
            return None
        for card in group:
            if not isinstance(card, dict) or not card.get("name") or not isinstance(card.get("count"), int) or card["count"] <= 0:
                return None
            total += card["count"]
    return total


def validate_bundle(bundle: Mapping[str, Any]) -> dict[str, Any]:
    standings = bundle.get("standings") or []
    pairings = bundle.get("pairings") or []
    decklists = bundle.get("decklists") or {}
    player_ids = [str(row.get("player_id")) for row in standings]
    player_id_set = set(player_ids)
    problems: list[str] = []

    if len(player_ids) != len(player_id_set):
        problems.append(f"duplicate player_id rows: {len(player_ids) - len(player_id_set)}")
    missing_lists = 0
    bad_lists: list[str] = []
    for row in standings:
        player_id = str(row.get("player_id"))
        if not row.get("decklist"):
            continue
        size = decklist_size(decklists.get(player_id))
        if size is None:
            missing_lists += 1
            bad_lists.append(player_id)
        elif size != DECK_SIZE:
            missing_lists += 1
            bad_lists.append(f"{player_id}:{size}")
    if bad_lists:
        problems.append(f"lists that are absent or not {DECK_SIZE} cards: {len(bad_lists)} {bad_lists[:5]}")

    bad_participants: list[str] = []
    identities = set()
    duplicates = 0
    for row in pairings:
        for key in ("p1_id", "p2_id"):
            value = row.get(key)
            if value is not None and str(value) not in player_id_set:
                bad_participants.append(f"{key}={value}")
        identity = (row.get("round"), row.get("phase"), row.get("p1_id"), row.get("p2_id"))
        if identity in identities:
            duplicates += 1
        identities.add(identity)
    if bad_participants:
        problems.append(f"pairing participants absent from standings: {len(bad_participants)} {bad_participants[:5]}")
    if duplicates:
        problems.append(f"duplicate pairing identities: {duplicates}")

    return {
        "labs_id": bundle.get("labs_id"),
        "players": len(standings),
        "pairings": len(pairings),
        "decklists": sum(1 for row in standings if row.get("decklist")),
        "lists_valid": sum(1 for row in standings if row.get("decklist")) - missing_lists,
        "problems": problems,
        "ok": not problems,
    }


def _standings_rows(bundle: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in bundle["standings"]:
        stable_id = str(row["player_id"])
        decklist = bundle["decklists"].get(stable_id) if row.get("decklist") else None
        rows.append(
            {
                "player": stable_id,
                "placing": row.get("placement"),
                "record": {"wins": row.get("wins", 0), "losses": row.get("losses", 0), "ties": row.get("ties", 0)},
                "deck": {"id": row.get("deck_id"), "name": row.get("deck_name")},
                "decklist": decklist,
                "drop": row.get("drop_round"),
            }
        )
    return rows


def _pairing_rows(bundle: Mapping[str, Any]) -> list[dict[str, Any]]:
    player_by_tp = {int(row["tp_id"]): str(row["player_id"]) for row in bundle["standings"]}
    player_ids = set(player_by_tp.values())
    rows = []
    for row in bundle["pairings"]:
        player1 = str(row["p1_id"]) if row.get("p1_id") is not None else ""
        player2 = str(row["p2_id"]) if row.get("p2_id") is not None else ""
        if (player1 and player1 not in player_ids) or (player2 and player2 not in player_ids):
            raise RuntimeError(f"pairing participant not in standings: {player1}, {player2}")
        winner_raw = row.get("winner")
        if winner_raw in (None, -1, "-1"):
            winner = "-1"
        elif winner_raw in (0, "0"):
            winner = "0"
        else:
            winner = player_by_tp[int(winner_raw)]
        rows.append({"round": row["round"], "phase": row.get("phase", 1), "match": "", "player1": player1, "player2": player2, "winner": winner})
    return rows


def import_bundles(database: str | Path, bundles: Iterable[Mapping[str, Any]], deck_mapping: Mapping[str, str | None] | None = None) -> dict[str, Any]:
    store = LimitlessStore(database, deck_mapping=deck_mapping)
    store.ensure_schema()
    bundles = list(bundles)

    for bundle in bundles:
        report = validate_bundle(bundle)
        if not report["ok"]:
            raise RuntimeError(f"bundle {bundle.get('labs_id')} failed validation: {report['problems']}")

    overlap = sorted({str(bundle["public_id"]) for bundle in bundles} & store.existing_tournament_ids())
    if overlap:
        raise RuntimeError(f"target event IDs already exist: {overlap}")

    counts = {"tournaments": 0, "standings": 0, "pairings": 0, "decklists": 0}
    for bundle in bundles:
        public_id = str(bundle["public_id"])
        details = {
            "source": SOURCE,
            "source_public_id": public_id,
            "source_url": f"https://limitlesstcg.com/tournaments/{public_id}",
            "labs_id": bundle["labs_id"],
            "labs_metadata": bundle["metadata"],
        }
        store.upsert_tournament(
            {
                "id": public_id,
                "game": "PTCG",
                "format": "STANDARD",
                "name": f"Regional {bundle['city']}",
                "date": bundle["date"],
                "players": bundle["metadata"]["players"],
            },
            details,
        )
        standings_rows = _standings_rows(bundle)
        store.upsert_standings(public_id, standings_rows)
        pairing_rows = _pairing_rows(bundle)
        store.upsert_pairings(public_id, pairing_rows)
        counts["tournaments"] += 1
        counts["standings"] += len(standings_rows)
        counts["decklists"] += sum(1 for row in standings_rows if row["decklist"] is not None)
        counts["pairings"] += len(pairing_rows)
    return counts
