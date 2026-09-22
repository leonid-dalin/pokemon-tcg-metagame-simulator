from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any, Iterable

from src.ingestion.model import ACE_SPEC_CARDS, _card_limit

SCHEMA = """
CREATE TABLE IF NOT EXISTS tournaments (id TEXT PRIMARY KEY, game TEXT, format TEXT, name TEXT, date TEXT, players INTEGER, details_json TEXT);
CREATE TABLE IF NOT EXISTS standings (tournament_id TEXT, player_id TEXT, placing INTEGER, wins INTEGER, losses INTEGER, ties INTEGER, deck_id TEXT, decklist_json TEXT, dropped_round INTEGER, PRIMARY KEY (tournament_id, player_id));
CREATE TABLE IF NOT EXISTS pairings (tournament_id TEXT, round INTEGER, phase INTEGER, player1 TEXT, player2 TEXT, winner TEXT, PRIMARY KEY (tournament_id, round, phase, player1, player2));
"""


class LimitlessStore:
    def __init__(self, path: str | Path):
        self.path = str(path)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            conn.executescript(SCHEMA)

    def connect(self):
        return sqlite3.connect(self.path)

    def _decklists(self, archetype: str):
        with closing(self.connect()) as conn:
            rows = conn.execute(
                "SELECT decklist_json FROM standings WHERE deck_id=? AND decklist_json IS NOT NULL",
                (archetype,),
            )
            for (raw,) in rows:
                if raw and raw != "null":
                    yield json.loads(raw)

    def upsert_tournament(self, row: dict[str, Any], details: dict[str, Any]) -> None:
        with self.connect() as conn:
            conn.execute("INSERT OR REPLACE INTO tournaments (id, game, format, name, date, players, details_json) VALUES (?, ?, ?, ?, ?, ?, ?)", (
                row["id"], row.get("game"), row.get("format"), row.get("name"), row.get("date"), row.get("players"), json.dumps(details)
            ))

    def upsert_standings(self, tournament_id: str, rows: Iterable[dict[str, Any]]) -> None:
        with self.connect() as conn:
            for row in rows:
                record = row.get("record", {})
                deck = row.get("deck") or {}
                conn.execute("INSERT OR REPLACE INTO standings (tournament_id, player_id, placing, wins, losses, ties, deck_id, decklist_json, dropped_round) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", (
                    tournament_id, str(row.get("player", row.get("name", ""))), row.get("placing"), record.get("wins", 0),
                    record.get("losses", 0), record.get("ties", 0), deck.get("id"), json.dumps(row.get("decklist")) if row.get("decklist") is not None else None, row.get("drop")
                ))

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
        query = """
        SELECT s1.deck_id, s2.deck_id, p.winner, p.player1, p.player2
        FROM pairings p JOIN standings s1 ON s1.tournament_id=p.tournament_id AND s1.player_id=p.player1
        JOIN standings s2 ON s2.tournament_id=p.tournament_id AND s2.player_id=p.player2
        WHERE p.player1 != '' AND p.player2 != '' AND p.winner NOT IN ('0', '-1', '')
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
        names = list(archetypes or [])
        with self.connect() as conn:
            if names:
                placeholders = ",".join("?" for _ in names)
                rows = conn.execute(f"SELECT deck_id, COUNT(*) FROM standings WHERE deck_id IN ({placeholders}) GROUP BY deck_id", names).fetchall()
            else:
                rows = conn.execute("SELECT deck_id, COUNT(*) FROM standings WHERE deck_id IS NOT NULL GROUP BY deck_id").fetchall()
        total = sum(count for _, count in rows)
        return {str(deck): count / total for deck, count in rows} if total else {}

    def observed_cards(self, archetype: str) -> set[str]:
        return {row["card"] for row in self.observed_skeleton(archetype)}

    def pairings_with_decklists(self, archetype_pattern: str) -> list[tuple[Any, ...]]:
        query = """
        SELECT s1.deck_id, s2.deck_id, s1.decklist_json, s2.decklist_json, p.winner, p.player1, p.player2
        FROM pairings p
        JOIN standings s1 ON s1.tournament_id=p.tournament_id AND s1.player_id=p.player1
        JOIN standings s2 ON s2.tournament_id=p.tournament_id AND s2.player_id=p.player2
        WHERE (lower(s1.deck_id) LIKE lower(?) OR lower(s2.deck_id) LIKE lower(?))
          AND p.player1 != '' AND p.player2 != '' AND p.winner NOT IN ('0', '-1', '')
        """
        with self.connect() as conn:
            return conn.execute(query, (archetype_pattern, archetype_pattern)).fetchall()
