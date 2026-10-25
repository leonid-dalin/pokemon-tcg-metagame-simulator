"""Read-only extraction of event, list, record and pairing data."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import Counter
from contextlib import closing
from pathlib import Path
from typing import Any

from src.ingestion.best60 import parse_list
from src.ingestion.mapping import load_archetype_map

EXTRACTION_SQL = """
SELECT p.tournament_id, p.round, p.phase, p.match, p.player1, p.player2, p.winner,
       t.date, t.game, t.format, t.name, t.players, t.details_json,
       s1.player_id, s1.deck_id, s1.deck_name, s1.decklist_json,
       s1.placing, s1.wins, s1.losses, s1.dropped_round,
       s2.player_id, s2.deck_id, s2.deck_name, s2.decklist_json,
       s2.placing, s2.wins, s2.losses, s2.dropped_round
FROM pairings AS p
LEFT JOIN tournaments AS t ON t.id = p.tournament_id
LEFT JOIN standings AS s1 ON s1.tournament_id = p.tournament_id AND s1.player_id = p.player1
LEFT JOIN standings AS s2 ON s2.tournament_id = p.tournament_id AND s2.player_id = p.player2
ORDER BY p.tournament_id, p.round, p.phase, p.match, p.player1, p.player2
"""


def _connect_readonly(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    connection.execute("PRAGMA query_only = ON")
    return connection


def _safe_json(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return None


def _metadata(value: Any) -> dict[str, Any]:
    decoded = _safe_json(value)
    return decoded if isinstance(decoded, dict) else {}


def _event_mode(details: dict[str, Any]) -> str:
    online = details.get("isOnline")
    return "online" if online is True else "offline" if online is False else "unknown"


def _match_mode(details: dict[str, Any], phase: Any) -> str:
    phases = details.get("phases")
    if not isinstance(phases, list):
        return "unknown"
    for item in phases:
        if not isinstance(item, dict) or str(item.get("phase")) != str(phase):
            continue
        value = item.get("mode")
        return str(value) if value is not None and str(value).strip() else "unknown"
    return "unknown"


def extract_corpus(path: str | Path, *, exclude_after_drop: bool = True,
                   include_event_ids: Any = None) -> dict[str, Any]:
    """Extract the stored pairing corpus without writing to the source database.

    Pairings retain their six-column database identity. Eligible focal views require a legal
    own list and a known opponent archetype; opponent card lists are not required.
    """
    source = Path(path)
    selected_event_ids = None if include_event_ids is None else tuple(sorted({str(item) for item in include_event_ids}))
    with closing(_connect_readonly(source)) as connection:
        connection.execute("BEGIN")
        connection.row_factory = sqlite3.Row
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        expected = {"tournaments", "standings", "pairings"}
        if not expected <= tables:
            raise ValueError(f"database is missing required tables: {sorted(expected - tables)}")
        metadata_rows = connection.execute(
            "SELECT id, date, game, format, name, players, details_json FROM tournaments ORDER BY id"
        ).fetchall()
        if selected_event_ids == ():
            standings_sql = ("SELECT s.tournament_id, s.player_id, t.date, s.deck_id, s.deck_name, s.decklist_json, "
                             "s.placing, s.wins, s.losses, s.ties, s.dropped_round "
                             "FROM standings AS s LEFT JOIN tournaments AS t ON t.id=s.tournament_id WHERE 0 "
                             "ORDER BY s.tournament_id, s.player_id")
            pairing_sql = EXTRACTION_SQL.replace("ORDER BY p.tournament_id", "WHERE 0 ORDER BY p.tournament_id")
            standings_rows, pairing_rows = [], []
        elif selected_event_ids is None:
            standings_sql = ("SELECT s.tournament_id, s.player_id, t.date, s.deck_id, s.deck_name, s.decklist_json, "
                             "s.placing, s.wins, s.losses, s.ties, s.dropped_round "
                             "FROM standings AS s LEFT JOIN tournaments AS t ON t.id=s.tournament_id "
                             "ORDER BY s.tournament_id, s.player_id")
            pairing_sql = EXTRACTION_SQL
            standings_rows = connection.execute(standings_sql).fetchall()
            pairing_rows = connection.execute(pairing_sql).fetchall()
        else:
            placeholders = ",".join("?" for _ in selected_event_ids)
            standings_sql = ("SELECT s.tournament_id, s.player_id, t.date, s.deck_id, s.deck_name, s.decklist_json, "
                             "s.placing, s.wins, s.losses, s.ties, s.dropped_round "
                             "FROM standings AS s LEFT JOIN tournaments AS t ON t.id=s.tournament_id "
                             f"WHERE s.tournament_id IN ({placeholders}) ORDER BY s.tournament_id, s.player_id")
            pairing_sql = EXTRACTION_SQL.replace(
                "ORDER BY p.tournament_id", f"WHERE p.tournament_id IN ({placeholders}) ORDER BY p.tournament_id"
            )
            standings_rows = connection.execute(standings_sql, selected_event_ids).fetchall()
            pairing_rows = connection.execute(pairing_sql, selected_event_ids).fetchall()

    archetype_map = load_archetype_map()
    events = [{
        "id": str(row["id"]), "date": row["date"],
        "metadata": {"game": row["game"], "format": row["format"], "name": row["name"],
                     "players": row["players"], "details": _metadata(row["details_json"])},
    } for row in metadata_rows]
    event_by_id = {event["id"]: event for event in events}

    lists: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    by_event_player: dict[tuple[str, str], dict[str, Any]] = {}
    list_audit: Counter[str] = Counter()
    for row in standings_rows:
        event_id, player_id = str(row["tournament_id"]), str(row["player_id"])
        event = event_by_id.get(event_id)
        date = event["date"] if event else None
        source_deck_id = str(row["deck_id"] or "")
        raw_archetype = str(row["deck_name"] or "")
        resolved_archetype = archetype_map.get(source_deck_id) or (raw_archetype.strip() or None)
        record = {"event": event_id, "player": player_id, "date": date,
                  "wins": row["wins"], "losses": row["losses"], "ties": row["ties"],
                  "deck_id": row["deck_id"],
                  "archetype_id": (source_deck_id or resolved_archetype) if resolved_archetype else None,
                  "archetype": resolved_archetype,
                  "raw_archetype": raw_archetype or None,
                  "placing": row["placing"], "dropped_round": row["dropped_round"]}
        records.append(record)
        by_event_player[(event_id, player_id)] = record
        deck_id = str(row["deck_id"] or "")
        deck_name = str(row["deck_name"] or "")
        if not row["decklist_json"] or str(row["decklist_json"]).strip().lower() == "null":
            list_audit["missing_list"] += 1
            continue
        label = archetype_map.get(deck_id) or (deck_name.strip() if deck_name.strip() else None)
        # The stable source ID wins; a name-only row gets a deterministic raw-name fallback.
        archetype_id = (deck_id or label) if label else None
        if archetype_id is None or label is None:
            list_audit["unknown_archetype"] += 1
            continue
        parsed = parse_list({"event": event_id, "player": player_id, "deck_id": deck_id,
                             "date": date, "placing": row["placing"], "wins": row["wins"],
                             "losses": row["losses"], "players": event["metadata"]["players"] if event else 0,
                             "decklist": row["decklist_json"]})
        if parsed is None:
            list_audit["invalid_list"] += 1
            continue
        item = {"event": event_id, "player": player_id, "archetype_id": archetype_id,
                "archetype": label, "counts": dict(parsed.counts), "raw": dict(record, decklist=row["decklist_json"])}
        lists.append(item)
        by_event_player[(event_id, player_id)]["counts"] = dict(parsed.counts)
        by_event_player[(event_id, player_id)]["list_valid"] = True
        list_audit["valid_lists"] += 1

    pairings: list[dict[str, Any]] = []
    views: list[dict[str, Any]] = []
    exclusions: Counter[str] = Counter()
    for row in pairing_rows:
        event_id = str(row["tournament_id"])
        p1, p2 = str(row["player1"] or ""), str(row["player2"] or "")
        key = (row["tournament_id"], row["round"], row["phase"], row["match"], row["player1"], row["player2"])
        r1, r2 = by_event_player.get((event_id, p1)), by_event_player.get((event_id, p2))
        a1id = (r1 or {}).get("archetype_id")
        a2id = (r2 or {}).get("archetype_id")
        a1 = (r1 or {}).get("archetype")
        a2 = (r2 or {}).get("archetype")
        round_number = int(row["round"] or 0)
        if not p1 or not p2:
            reason = "bye"
        elif r1 is None or r2 is None:
            reason = "missing_join"
        elif str(row["winner"]) == "0":
            reason = "tie"
        elif str(row["winner"]) == "-1":
            reason = "invalid_winner"
        elif str(row["winner"]) not in {p1, p2}:
            reason = "invalid_winner"
        elif not a1id or not a2id:
            reason = "unknown_archetype"
        elif exclude_after_drop and any(
            r.get("dropped_round") is not None and round_number > int(r["dropped_round"])
            for r in (r1, r2)
        ):
            reason = "after_drop"
        else:
            reason = None
        details = _metadata(row["details_json"])
        pairing = {"key": key, "event": event_id, "round": round_number, "phase": key[2],
                   "match": key[3], "player1": p1, "player2": p2, "winner": row["winner"],
                   "date": row["date"], "game": row["game"],
                   "source": details.get("source") or "unknown", "platform": details.get("platform") or "unknown",
                   "is_online": details.get("isOnline") if isinstance(details.get("isOnline"), bool) else None,
                   "delivery_mode": _event_mode(details), "match_mode": _match_mode(details, row["phase"]),
                   "format": row["format"],
                   "archetype1_id": a1id, "archetype2_id": a2id,
                   "archetype1": a1, "archetype2": a2,
                   "raw_archetype1": (r1 or {}).get("raw_archetype"),
                   "raw_archetype2": (r2 or {}).get("raw_archetype"), "exclusion": reason,
                   "eligible_view_count": 0}
        if reason:
            exclusions[reason] += 1
        else:
            orientations = ((p1, p2, r1, r2, 1), (p2, p1, r2, r1, 2))
            selected = []
            for player, opponent, own, opposing, side in orientations:
                if own.get("list_valid") and opposing.get("archetype_id"):
                    selected.append((player, opponent, own, opposing, side))
            count = len(selected)
            pairing["eligible_view_count"] = count
            if count == 0:
                pairing["exclusion"] = "no_legal_focal_list"
                exclusions["no_legal_focal_list"] += 1
            else:
                for player, opponent, own, opposing, side in selected:
                    views.append({"pairing_key": key, "event": event_id, "round": round_number,
                                  "date": row["date"], "source": pairing["source"],
                                  "platform": pairing["platform"], "is_online": pairing["is_online"],
                                  "delivery_mode": pairing["delivery_mode"],
                                  "match_mode": pairing["match_mode"], "game": row["game"],
                                  "format": row["format"],
                                  "player": player, "opponent": opponent, "player1": p1, "player2": p2,
                                  "side": side, "archetype_id": own["archetype_id"],
                                  "archetype": own["archetype"],
                                  "raw_archetype": own.get("raw_archetype"),
                                  "opponent_archetype_id": opposing["archetype_id"],
                                  "opponent_archetype": opposing["archetype"],
                                  "opponent_raw_archetype": opposing.get("raw_archetype"),
                                  "counts": own["counts"], "won": int(str(row["winner"]) == player),
                                  "weight": 1.0 / count})
        pairings.append(pairing)

    fingerprint = _fingerprint(source)
    return {"events": events, "lists": lists, "pairings": pairings, "pairing_views": views,
            "records": records, "audit": {
                "source_path": str(source.resolve()), "source_sha256": fingerprint["sha256"],
                "source_files": fingerprint["files"], "snapshot_fingerprint": fingerprint["snapshot_fingerprint"],
                "queries": {"events": "SELECT id, date, game, format, name, players, details_json FROM tournaments ORDER BY id",
                            "standings": standings_sql, "pairings": pairing_sql.strip()},
                "included_event_ids": list(selected_event_ids) if selected_event_ids is not None else None,
                "pairing_rows": len(pairings),
                "pairing_exclusions": dict(sorted(exclusions.items())),
                "lists": dict(sorted(list_audit.items())),
                "definitions": {"pairing_identity": "(tournament_id, round, phase, match, player1, player2)",
                                "pairing_views": "one per legal focal list against a known opponent archetype; weights sum to one per eligible pairing",
                                "winner": "only stored player IDs are decided outcomes; 0 is a tie and all other sentinels are invalid"}}}


def _fingerprint(path: Path) -> dict[str, Any]:
    files: dict[str, dict[str, Any]] = {}
    for candidate in (path, Path(str(path) + "-wal")):
        if not candidate.is_file():
            continue
        digest = hashlib.sha256()
        with candidate.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        files[candidate.name] = {"bytes": candidate.stat().st_size, "sha256": digest.hexdigest()}
    combined = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"sha256": files[path.name]["sha256"], "files": files, "snapshot_fingerprint": combined}
