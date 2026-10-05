from __future__ import annotations

import json
import re
import sqlite3
import warnings
from dataclasses import dataclass
from contextlib import closing, contextmanager
from pathlib import Path
from typing import Any, Iterable, Mapping

from src.ingestion.mapping import load_archetype_map, resolve_archetype
from src.ingestion.model import PlayerObservation

SCHEMA = """
CREATE TABLE IF NOT EXISTS tournaments (id TEXT PRIMARY KEY, game TEXT, format TEXT, name TEXT, date TEXT, players INTEGER, details_json TEXT);
CREATE TABLE IF NOT EXISTS standings (tournament_id TEXT, player_id TEXT, placing INTEGER, wins INTEGER, losses INTEGER, ties INTEGER, deck_id TEXT, deck_name TEXT, decklist_json TEXT, dropped_round INTEGER, PRIMARY KEY (tournament_id, player_id));
CREATE TABLE IF NOT EXISTS pairings (tournament_id TEXT, round INTEGER, phase INTEGER, match TEXT, player1 TEXT, player2 TEXT, winner TEXT, PRIMARY KEY (tournament_id, round, phase, match, player1, player2));
"""


def _decklist_card_names(raw: str | None) -> frozenset[str]:
    if not raw or raw == "null":
        return frozenset()
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return frozenset()
    if not isinstance(payload, dict):
        return frozenset()
    return frozenset(
        str(card["name"])
        for group in payload.values()
        if isinstance(group, list)
        for card in group
        if isinstance(card, dict) and card.get("name") is not None and str(card["name"]).strip()
    )


class LimitlessStore:
    def __init__(self, path: str | Path, canonical_names: Iterable[str] | None = None, deck_mapping: Mapping[str, str | None] | None = None):
        if canonical_names is not None:
            warnings.warn("canonical_names is ignored; archetypes come from deck_mapping", UserWarning, stacklevel=2)
        self.path = str(path)
        self.deck_mapping = dict(deck_mapping) if deck_mapping is not None else load_archetype_map()
        self._schema_ready = False
        self._read_ready = False

    def ensure_schema(self) -> None:
        if self._schema_ready:
            return
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            conn.executescript(SCHEMA)
            columns = {row[1] for row in conn.execute("PRAGMA table_info(standings)")}
            if "deck_name" not in columns:
                conn.execute("ALTER TABLE standings ADD COLUMN deck_name TEXT")
            pairing_columns = {row[1] for row in conn.execute("PRAGMA table_info(pairings)")}
            if "match" not in pairing_columns:
                conn.execute("ALTER TABLE pairings RENAME TO pairings_legacy")
                conn.execute(
                    "CREATE TABLE pairings (tournament_id TEXT, round INTEGER, phase INTEGER, match TEXT, player1 TEXT, player2 TEXT, winner TEXT, PRIMARY KEY (tournament_id, round, phase, match, player1, player2))"
                )
                legacy_columns = {row[1] for row in conn.execute("PRAGMA table_info(pairings_legacy)")}
                round_expr = "round" if "round" in legacy_columns else "0"
                phase_expr = "phase" if "phase" in legacy_columns else "0"
                conn.execute(
                    f"INSERT INTO pairings (tournament_id, round, phase, match, player1, player2, winner) SELECT tournament_id, {round_expr}, {phase_expr}, '', player1, player2, winner FROM pairings_legacy"
                )
                if conn.execute("SELECT COUNT(*) FROM pairings_legacy").fetchone()[0] != conn.execute("SELECT COUNT(*) FROM pairings").fetchone()[0]:
                    raise sqlite3.IntegrityError("pairing migration changed the row count")
                conn.execute("DROP TABLE pairings_legacy")
        self._schema_ready = True

    def prepare_for_read(self) -> None:
        if self._read_ready:
            return
        self.ensure_schema()
        self.backfill_deck_names()
        self._read_ready = True

    def _resolve_deck_name(self, deck_id: str | None, display_name: str | None = None) -> str | None:
        if not deck_id or deck_id == "other":
            return None
        explicit = resolve_archetype(str(deck_id), self.deck_mapping)
        if str(deck_id) in self.deck_mapping:
            return explicit
        return display_name if display_name and display_name.strip() else None


    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.path)
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def summary(self) -> dict[str, Any]:
        uri = f"{Path(self.path).resolve().as_uri()}?mode=ro"
        with closing(sqlite3.connect(uri, uri=True)) as conn:
            tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if not {"standings", "pairings"} <= tables:
                return {"schema": "incomplete"}
            columns = {row[1] for row in conn.execute("PRAGMA table_info(standings)")}
            if "deck_name" not in columns:
                return {"schema": "legacy"}
            decks = conn.execute("SELECT COUNT(DISTINCT deck_name) FROM standings WHERE deck_name IS NOT NULL").fetchone()[0]
            pairings = conn.execute("SELECT COUNT(*) FROM pairings").fetchone()[0]
            decklists = conn.execute(
                "SELECT COUNT(*) FROM standings WHERE decklist_json IS NOT NULL AND decklist_json != 'null'"
            ).fetchone()[0]
        return {"schema": "current", "decks": decks, "pairings": pairings, "decklists": decklists}

    def _decklists(self, archetype: str, event_id: str | None = None):
        self.prepare_for_read()
        with self.connect() as conn:
            event_filter = " AND tournament_id=?" if event_id is not None else ""
            rows = conn.execute(
                "SELECT decklist_json FROM standings WHERE deck_name=? AND decklist_json IS NOT NULL" + event_filter,
                (archetype,) if event_id is None else (archetype, event_id),
            )
            for (raw,) in rows:
                if raw and raw != "null":
                    yield json.loads(raw)

    def upsert_event_bundle(
        self,
        event: dict[str, Any],
        details: dict[str, Any],
        standings: Iterable[dict[str, Any]],
        pairings: Iterable[dict[str, Any]],
        deck_names: dict[str, str] | None = None,
    ) -> None:
        self.ensure_schema()
        event_id = str(event["id"])
        pairing_rows = list(pairings)
        identities = [
            (
                event_id,
                row.get("round", 0),
                row.get("phase", 0),
                str(row.get("match") or ""),
                str(row.get("player1") or ""),
                str(row.get("player2") or ""),
            )
            for row in pairing_rows
        ]
        if len(identities) != len(set(identities)):
            raise sqlite3.IntegrityError("duplicate pairing identity")
        with self.connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO tournaments (id, game, format, name, date, players, details_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (event_id, event.get("game"), event.get("format"), event.get("name"), event.get("date"), event.get("players"), json.dumps({"source": "Limitless developer API", "catalogue_event": event, "event_details": details})),
            )
            conn.execute("DELETE FROM standings WHERE tournament_id=?", (event_id,))
            conn.execute("DELETE FROM pairings WHERE tournament_id=?", (event_id,))
            for row in standings:
                record = row.get("record") or {}
                deck = row.get("deck") or {}
                deck_id = deck.get("id")
                deck_name = self._resolve_deck_name(deck_id, deck.get("name") or (deck_names or {}).get(deck_id))
                conn.execute(
                    "INSERT OR REPLACE INTO standings (tournament_id, player_id, placing, wins, losses, ties, deck_id, deck_name, decklist_json, dropped_round) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (event_id, str(row.get("player", row.get("name", ""))), row.get("placing"), record.get("wins", 0), record.get("losses", 0), record.get("ties", 0), deck_id, deck_name, json.dumps(row.get("decklist")) if row.get("decklist") is not None else None, row.get("drop")),
                )
            for row in pairing_rows:
                conn.execute(
                    "INSERT OR REPLACE INTO pairings (tournament_id, round, phase, match, player1, player2, winner) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (event_id, row.get("round", 0), row.get("phase", 0), str(row.get("match") or ""), str(row.get("player1") or ""), str(row.get("player2") or ""), str(row.get("winner")) if row.get("winner") is not None else ""),
                )

    def upsert_tournament(self, row: dict[str, Any], details: dict[str, Any]) -> None:
        self.ensure_schema()
        with self.connect() as conn:
            conn.execute("INSERT OR REPLACE INTO tournaments (id, game, format, name, date, players, details_json) VALUES (?, ?, ?, ?, ?, ?, ?)", (
                row["id"], row.get("game"), row.get("format"), row.get("name"), row.get("date"), row.get("players"), json.dumps(details)
            ))

    def existing_tournament_ids(self) -> set[str]:
        self.ensure_schema()
        with self.connect() as conn:
            return {str(row[0]) for row in conn.execute("SELECT id FROM tournaments")}

    def unmapped_deck_ids(self) -> list[str]:
        self.ensure_schema()
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT deck_id FROM standings "
                "WHERE deck_id IS NOT NULL AND deck_name IS NULL "
                "ORDER BY deck_id"
            ).fetchall()
        return [str(row[0]) for row in rows]

    def upsert_standings(
        self,
        tournament_id: str,
        rows: Iterable[dict[str, Any]],
        deck_names: dict[str, str] | None = None,
    ) -> None:
        self.ensure_schema()
        with self.connect() as conn:
            for row in rows:
                record = row.get("record", {})
                deck = row.get("deck") or {}
                deck_id = deck.get("id")
                deck_name = self._resolve_deck_name(
                    deck_id,
                    deck.get("name") or (deck_names or {}).get(deck_id),
                )
                conn.execute("INSERT OR REPLACE INTO standings (tournament_id, player_id, placing, wins, losses, ties, deck_id, deck_name, decklist_json, dropped_round) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (
                    tournament_id, str(row.get("player", row.get("name", ""))), row.get("placing"), record.get("wins", 0),
                    record.get("losses", 0), record.get("ties", 0), deck_id, deck_name, json.dumps(row.get("decklist")) if row.get("decklist") is not None else None, row.get("drop")
                ))

    def backfill_deck_names(self, deck_names: dict[str, str] | None = None) -> int:
        self.ensure_schema()
        if deck_names is None:
            with self.connect() as conn:
                rows = conn.execute(
                    "SELECT tournament_id, player_id, deck_id, deck_name FROM standings WHERE deck_id IS NOT NULL"
                ).fetchall()
            with self.connect() as conn:
                conn.executemany(
                    "UPDATE standings SET deck_name=? WHERE tournament_id=? AND player_id=?",
                    [
                        (
                            self._existing_deck_name(deck_id, deck_name),
                            tournament_id,
                            player_id,
                        )
                        for tournament_id, player_id, deck_id, deck_name in rows
                    ],
                )
            return len(rows)
        updated = 0
        with self.connect() as conn:
            for deck_id in deck_names:
                deck_name = self._resolve_deck_name(deck_id, deck_names[deck_id])
                if deck_name is None:
                    continue
                cursor = conn.execute(
                    "UPDATE standings SET deck_name=? WHERE deck_id=? AND (deck_name IS NULL OR deck_name=deck_id)",
                    (deck_name, deck_id),
                )
                updated += cursor.rowcount
        return updated

    def _existing_deck_name(self, deck_id: str | None, deck_name: str | None) -> str | None:
        if not deck_id or deck_id == "other":
            return None
        if str(deck_id) in self.deck_mapping:
            return resolve_archetype(str(deck_id), self.deck_mapping)
        return deck_name if deck_name and deck_name.strip() else None

    def upsert_pairings(self, tournament_id: str, rows: Iterable[dict[str, Any]]) -> None:
        self.ensure_schema()
        with self.connect() as conn:
            for row in rows:
                conn.execute("INSERT OR REPLACE INTO pairings (tournament_id, round, phase, match, player1, player2, winner) VALUES (?, ?, ?, ?, ?, ?, ?)", (
                    tournament_id, row.get("round", 0), row.get("phase", 0), str(row.get("match") or ""),
                    str(row.get("player1", "")), str(row.get("player2", "")), str(row.get("winner", ""))
                ))

    def iter_events(self):
        with self.connect() as conn:
            yield from conn.execute("SELECT id, game, format, name, date, players, details_json FROM tournaments ORDER BY date DESC")

    def matchup_rows(self):
        self.prepare_for_read()
        query = """
        SELECT s1.deck_name, s2.deck_name, p.winner, p.player1, p.player2
        FROM pairings p JOIN standings s1 ON s1.tournament_id=p.tournament_id AND s1.player_id=p.player1
        JOIN standings s2 ON s2.tournament_id=p.tournament_id AND s2.player_id=p.player2
        WHERE p.player1 != '' AND p.player2 != '' AND p.winner NOT IN ('0', '-1', '')
          AND s1.deck_name IS NOT NULL AND s2.deck_name IS NOT NULL
        """
        with self.connect() as conn:
            yield from conn.execute(query)

    def aggregate_rows(self):
        self.prepare_for_read()
        query = """
        SELECT p.tournament_id, p.round, s1.deck_name, s2.deck_name,
               p.winner, p.player1, p.player2,
               s1.dropped_round, s2.dropped_round
        FROM pairings p
        LEFT JOIN standings s1
          ON s1.tournament_id=p.tournament_id AND s1.player_id=p.player1
        LEFT JOIN standings s2
          ON s2.tournament_id=p.tournament_id AND s2.player_id=p.player2
        ORDER BY p.tournament_id, p.phase, p.round, p.player1, p.player2
        """
        with self.connect() as conn:
            for row in conn.execute(query):
                yield {
                    "event_id": row[0],
                    "round": row[1],
                    "deck1": row[2],
                    "deck2": row[3],
                    "winner": row[4],
                    "player1": row[5],
                    "player2": row[6],
                    "drop1": row[7],
                    "drop2": row[8],
                }

    def card_inclusion(self, archetype: str, event_id: str | None = None) -> dict[str, float]:
        counts: dict[str, int] = {}
        rows = list(self._decklists(archetype, event_id))
        for cards in rows:
            names = {card["name"] for group in (cards or {}).values() if isinstance(group, list) for card in group if isinstance(card, dict) and "name" in card}
            for name in names:
                counts[name] = counts.get(name, 0) + 1
        total = len(rows)
        return {name: count / total for name, count in sorted(counts.items())} if total else {}

    def observations(self) -> list[tuple[str, str, int]]:
        rows = []
        for deck1, deck2, winner, player1, player2 in self.matchup_rows():
            if str(winner) == str(player1):
                rows.append((str(deck1), str(deck2), 1))
            elif str(winner) == str(player2):
                rows.append((str(deck1), str(deck2), 0))
        return rows

    def player_observations(self) -> list[PlayerObservation]:
        self.prepare_for_read()
        query = """
        SELECT s1.deck_name, s2.deck_name, s1.decklist_json, s2.decklist_json,
               p.winner, p.player1, p.player2
        FROM pairings p
        JOIN standings s1 ON s1.tournament_id=p.tournament_id AND s1.player_id=p.player1
        JOIN standings s2 ON s2.tournament_id=p.tournament_id AND s2.player_id=p.player2
        WHERE p.player1 != '' AND p.player2 != '' AND p.winner NOT IN ('0', '-1', '')
          AND s1.deck_name IS NOT NULL AND s2.deck_name IS NOT NULL
          AND s1.decklist_json IS NOT NULL AND s2.decklist_json IS NOT NULL
        """
        with self.connect() as conn:
            rows = conn.execute(query).fetchall()
        observations = []
        for deck, opponent, decklist, opponent_decklist, winner, player1, player2 in rows:
            deck_cards = _decklist_card_names(decklist)
            opponent_cards = _decklist_card_names(opponent_decklist)
            if not deck_cards or not opponent_cards:
                continue
            if str(winner) == str(player1):
                result = 1
            elif str(winner) == str(player2):
                result = 0
            else:
                continue
            observations.append(PlayerObservation(
                str(deck), str(opponent), deck_cards, opponent_cards, result, (str(player1), str(player2)),
            ))
        return observations

    def player_observations(self) -> list[PlayerObservation]:
        self.prepare_for_read()
        query = """
        SELECT s1.deck_name, s2.deck_name, s1.decklist_json, s2.decklist_json,
               p.winner, p.player1, p.player2
        FROM pairings p
        JOIN standings s1 ON s1.tournament_id=p.tournament_id AND s1.player_id=p.player1
        JOIN standings s2 ON s2.tournament_id=p.tournament_id AND s2.player_id=p.player2
        WHERE p.player1 != '' AND p.player2 != '' AND p.winner NOT IN ('0', '-1', '')
          AND s1.deck_name IS NOT NULL AND s2.deck_name IS NOT NULL
          AND s1.decklist_json IS NOT NULL AND s2.decklist_json IS NOT NULL
        """
        with self.connect() as conn:
            rows = conn.execute(query).fetchall()
        observations = []
        for deck, opponent, decklist, opponent_decklist, winner, player1, player2 in rows:
            deck_cards = _decklist_card_names(decklist)
            opponent_cards = _decklist_card_names(opponent_decklist)
            if not deck_cards or not opponent_cards:
                continue
            if str(winner) == str(player1):
                result = 1
            elif str(winner) == str(player2):
                result = 0
            else:
                continue
            observations.append(PlayerObservation(
                str(deck), str(opponent), deck_cards, opponent_cards, result, (str(player1), str(player2)),
            ))
        return observations

    def tournament_date_range(self) -> tuple[str | None, str | None]:
        with self.connect() as conn:
            return conn.execute("SELECT MIN(date), MAX(date) FROM tournaments").fetchone()

    def matchup_games_by_archetype(self) -> dict[str, int]:
        totals: dict[str, int] = {}
        for row in self.aggregate_rows():
            if not row.get("deck1") or not row.get("deck2") or row["deck1"] == row["deck2"]:
                continue
            for deck in (row["deck1"], row["deck2"]):
                totals[str(deck)] = totals.get(str(deck), 0) + 1
        return totals

    def archetype_lists(self, archetype: str) -> list[dict[str, Any]]:
        """Every stored list of one archetype with its event result, for the Best-60 builder."""
        self.prepare_for_read()
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT s.tournament_id, s.player_id, s.deck_id, s.placing, s.wins, s.losses, s.decklist_json, t.players, t.date "
                "FROM standings s LEFT JOIN tournaments t ON t.id=s.tournament_id "
                "WHERE s.deck_name=? AND s.decklist_json IS NOT NULL AND s.decklist_json != 'null' "
                "ORDER BY s.tournament_id, s.player_id",
                (archetype,),
            ).fetchall()
        keys = ("event", "player", "deck_id", "placing", "wins", "losses", "decklist", "players", "date")
        return [dict(zip(keys, row)) for row in rows]

    def player_records(self) -> dict[tuple[str, str], tuple[int, int]]:
        """Wins and losses of every player at every event, for player strength."""
        self.prepare_for_read()
        with self.connect() as conn:
            rows = conn.execute("SELECT tournament_id, player_id, wins, losses FROM standings").fetchall()
        return {(str(event), str(player)): (int(wins or 0), int(losses or 0)) for event, player, wins, losses in rows}

    def deck_weights(self, archetypes: Iterable[str] | None = None) -> dict[str, float]:
        self.prepare_for_read()
        names = list(archetypes or [])
        with self.connect() as conn:
            if names:
                placeholders = ",".join("?" for _ in names)
                rows = conn.execute(f"SELECT deck_name, COUNT(*) FROM standings WHERE deck_name IN ({placeholders}) GROUP BY deck_name", names).fetchall()
            else:
                rows = conn.execute("SELECT deck_name, COUNT(*) FROM standings WHERE deck_name IS NOT NULL GROUP BY deck_name").fetchall()
        total = sum(count for _, count in rows)
        return {str(deck): count / total for deck, count in rows} if total else {}

    def pairings_with_decklists(self, archetype_pattern: str) -> list[tuple[Any, ...]]:
        self.prepare_for_read()
        query = """
        SELECT s1.deck_name, s2.deck_name, s1.decklist_json, s2.decklist_json, p.winner, p.player1, p.player2
        FROM pairings p
        JOIN standings s1 ON s1.tournament_id=p.tournament_id AND s1.player_id=p.player1
        JOIN standings s2 ON s2.tournament_id=p.tournament_id AND s2.player_id=p.player2
        WHERE (lower(s1.deck_name) LIKE lower(?) OR lower(s2.deck_name) LIKE lower(?))
          AND p.player1 != '' AND p.player2 != '' AND p.winner NOT IN ('0', '-1', '')
        """
        with self.connect() as conn:
            return conn.execute(query, (archetype_pattern, archetype_pattern)).fetchall()
