from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from contextlib import closing
from pathlib import Path
from typing import Any, Iterable, Mapping

from src.ingestion.mapping import resolve_archetype
from src.ingestion.model import ACE_SPEC_CARDS, PlayerObservation, _card_limit

SCHEMA = """
CREATE TABLE IF NOT EXISTS tournaments (id TEXT PRIMARY KEY, game TEXT, format TEXT, name TEXT, date TEXT, players INTEGER, details_json TEXT);
CREATE TABLE IF NOT EXISTS standings (tournament_id TEXT, player_id TEXT, placing INTEGER, wins INTEGER, losses INTEGER, ties INTEGER, deck_id TEXT, deck_name TEXT, decklist_json TEXT, dropped_round INTEGER, PRIMARY KEY (tournament_id, player_id));
CREATE TABLE IF NOT EXISTS pairings (tournament_id TEXT, round INTEGER, phase INTEGER, player1 TEXT, player2 TEXT, winner TEXT, PRIMARY KEY (tournament_id, round, phase, player1, player2));
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
        self.path = str(path)
        self.canonical_names = list(canonical_names) if canonical_names is not None else self._load_canonical_names()
        self.deck_mapping = deck_mapping
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
        self._schema_ready = True

    def prepare_for_read(self) -> None:
        if self._read_ready:
            return
        self.ensure_schema()
        self.backfill_deck_names()
        self._read_ready = True

    @staticmethod
    def _load_canonical_names() -> list[str]:
        path = Path("data/input/ea_input.json")
        if not path.exists():
            return []
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        return [str(name) for name in payload.get("archetypes", [])]

    def _resolve_deck_name(self, deck_id: str | None, display_name: str | None = None) -> str | None:
        if not deck_id or deck_id == "other":
            return None
        explicit = resolve_archetype(str(deck_id), self.deck_mapping)
        if explicit is not None:
            return explicit
        return None


    def connect(self):
        return sqlite3.connect(self.path)

    def _decklists(self, archetype: str):
        self.prepare_for_read()
        with closing(self.connect()) as conn:
            rows = conn.execute(
                "SELECT decklist_json FROM standings WHERE deck_name=? AND decklist_json IS NOT NULL",
                (archetype,),
            )
            for (raw,) in rows:
                if raw and raw != "null":
                    yield json.loads(raw)

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
            )
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
                rows = conn.execute("SELECT tournament_id, player_id, deck_id FROM standings WHERE deck_id IS NOT NULL").fetchall()
            with self.connect() as conn:
                conn.executemany(
                    "UPDATE standings SET deck_name=? WHERE tournament_id=? AND player_id=?",
                    [(self._resolve_deck_name(deck_id), tournament_id, player_id) for tournament_id, player_id, deck_id in rows],
                )
            return len(rows)
        updated = 0
        with self.connect() as conn:
            for deck_id in deck_names:
                deck_name = self._resolve_deck_name(deck_id)
                if deck_name is None:
                    continue
                cursor = conn.execute(
                    "UPDATE standings SET deck_name=? WHERE deck_id=? AND (deck_name IS NULL OR deck_name=deck_id)",
                    (deck_name, deck_id),
                )
                updated += cursor.rowcount
        return updated

    def upsert_pairings(self, tournament_id: str, rows: Iterable[dict[str, Any]]) -> None:
        with self.connect() as conn:
            for row in rows:
                conn.execute("INSERT OR REPLACE INTO pairings (tournament_id, round, phase, player1, player2, winner) VALUES (?, ?, ?, ?, ?, ?)", (
                    tournament_id, row.get("round", 0), row.get("phase", 0), str(row.get("player1", "")),
                    str(row.get("player2", "")), str(row.get("winner", ""))
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

    def card_inclusion(self, archetype: str) -> dict[str, float]:
        counts: dict[str, int] = {}
        rows = list(self._decklists(archetype))
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
                str(deck), str(opponent), deck_cards, opponent_cards, result,
            ))
        return observations

    def observed_skeleton(self, archetype: str) -> list[dict[str, object]]:
        counts: dict[str, int] = {}
        rows = list(self._decklists(archetype))
        for decklist in rows:
            for group in decklist.values():
                if not isinstance(group, list):
                    continue
                for card in group:
                    if isinstance(card, dict) and card.get("name"):
                        counts[str(card["name"])] = counts.get(str(card["name"]), 0) + int(card.get("count", 0))
        decks = len(rows)
        skeleton = []
        ace_cards = []
        for card, total in sorted(counts.items(), key=lambda item: item[1], reverse=True):
            if total / decks < 0.75:
                continue
            limit = _card_limit(card, {})
            copies = round(total / decks) if limit is None else min(limit, round(total / decks))
            if card in ACE_SPEC_CARDS:
                ace_cards.append({"card": card, "copies": min(copies, 1)})
                continue
            if copies:
                skeleton.append({"card": card, "copies": copies})
        if ace_cards:
            skeleton.append(ace_cards[0])
        return skeleton

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

    def observed_cards(self, archetype: str) -> set[str]:
        return {row["card"] for row in self.observed_skeleton(archetype)}

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
