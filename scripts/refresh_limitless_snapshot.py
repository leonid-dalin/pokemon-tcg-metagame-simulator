from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.core.files import write_json_atomic
from src.ingestion.aggregate import build_artifact
from src.ingestion.client import LimitlessClient
from src.ingestion.model import CardModelNotIdentifiable, fit_card_model, model_artifact, select_model_cards
from src.ingestion.model_cache import load_or_fit_card_model
from src.ingestion.store import LimitlessStore


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Refresh an online Limitless snapshot")
    parser.add_argument("--database", required=True, help="SQLite snapshot path")
    parser.add_argument("--artifact-dir", required=True, help="Directory for generated JSON artefacts")
    parser.add_argument("--manifest", required=True, help="Snapshot manifest path")
    parser.add_argument("--start", required=True, help="Inclusive ISO-8601 event start")
    parser.add_argument("--end", required=True, help="Exclusive ISO-8601 event end")
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument("--event-id", action="append", default=[], help="Refresh this event ID without scanning the catalogue")
    parser.add_argument("--refresh-event-id", action="append", default=[], help="Replace this stored event using its current source rows")
    return parser


def _catalogue(client: LimitlessClient, start: str, end: str, page_size: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    page = 1
    while True:
        batch = client.tournaments(game="PTCG", format="STANDARD", limit=page_size, page=page)
        if not batch:
            break
        for row in batch:
            event_id = str(row.get("id"))
            date = str(row.get("date", ""))
            if event_id not in seen and start <= date < end:
                rows.append(row)
                seen.add(event_id)
        oldest = min(str(row.get("date", "")) for row in batch)
        if len(batch) < page_size or oldest < start:
            break
        page += 1
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_model(store: LimitlessStore, path: Path, cache_path: Path, database_sha256: str) -> dict[str, Any]:
    observations = store.player_observations()
    if len(observations) < 4 or len({row.result for row in observations}) != 2:
        return {"status": "insufficient observations"}
    try:
        fitted, reused = load_or_fit_card_model(observations, database_sha256, cache_path)
    except CardModelNotIdentifiable:
        return {"status": "not identifiable"}
    write_json_atomic(model_artifact(fitted), str(path))
    return {"status": "complete", "database_sha256": database_sha256, "model_cache_reused": reused, "model_max_iter": 10_000, "card_packages": sorted(fitted.packages), "not_identified": fitted.not_identified}


def refresh(args: argparse.Namespace) -> dict[str, Any]:
    database = Path(args.database)
    artifact_dir = Path(args.artifact_dir)
    manifest = Path(args.manifest)
    client = LimitlessClient.from_environment()
    client.min_delay = 2.0
    store = LimitlessStore(database)
    store.ensure_schema()
    names = {
        str(row.get("identifier") or row.get("id")): str(row["name"])
        for row in client.game_decks()
        if row.get("name") and (row.get("identifier") or row.get("id"))
    }
    existing = store.existing_tournament_ids()
    if args.event_id:
        source_events = [{"id": event_id} for event_id in args.event_id]
    else:
        source_events = _catalogue(client, args.start, args.end, args.page_size)
    refresh_ids = set(args.refresh_event_id)
    source_events = [row for row in source_events if str(row["id"]) not in existing or str(row["id"]) in refresh_ids]
    added: list[dict[str, Any]] = []
    skipped_non_online = 0
    failed: list[dict[str, str]] = []
    for event in sorted(source_events, key=lambda row: str(row.get("date", ""))):
        event_id = str(event["id"])
        if event_id in existing and event_id not in refresh_ids:
            continue
        try:
            details = client.event_details(event_id)
            if details.get("isOnline") is not True:
                skipped_non_online += 1
                continue
            event = {**event, **details}
            standings = client.standings(event_id) if details.get("decklists", True) else []
            pairings = client.pairings(event_id)
            store.upsert_event_bundle(event, details, standings, pairings, names)
            existing.add(event_id)
            added.append({"id": event_id, "name": event.get("name"), "standings": len(standings), "pairings": len(pairings), "decklists": sum(1 for row in standings if row.get("decklist") is not None)})
        except Exception as exc:
            failed.append({"id": event_id, "error": str(exc)})
    artifact_dir.mkdir(parents=True, exist_ok=True)
    input_path = artifact_dir / "limitless_input.json"
    write_json_atomic(build_artifact(store), str(input_path))
    model_path = artifact_dir / "limitless_model_input.json"
    model_cache_path = artifact_dir / "limitless_model_fit.json"
    model_status = _write_model(store, model_path, model_cache_path, _sha256(database))
    summary = store.summary()
    result = {
        "source": {
            "game": "PTCG",
            "format": "STANDARD",
            "start": args.start,
            "end": args.end,
            "catalogue_events": len(source_events),
        },
        "added_online_events": added,
        "skipped_non_online": skipped_non_online,
        "failed_events": failed,
        "database": {"path": str(database), "sha256": _sha256(database), **summary},
        "artefacts": {"input": str(input_path), "model": str(model_path), "model_cache": str(model_cache_path), "model_status": model_status},
        "refreshed_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json_atomic(result, str(manifest))
    return result


def main() -> int:
    args = _parser().parse_args()
    result = refresh(args)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 1 if result["failed_events"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
