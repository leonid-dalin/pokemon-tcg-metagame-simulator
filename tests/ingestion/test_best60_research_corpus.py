import hashlib
import json
import sqlite3
from datetime import datetime, timedelta, timezone

from src.ingestion.best60_research.corpus import extract_corpus
from src.ingestion.model import ACE_SPEC_CARDS


def _database(path, *, invalid=False):
    with sqlite3.connect(path) as con:
        con.executescript("""
            CREATE TABLE tournaments (id TEXT PRIMARY KEY, game TEXT, format TEXT, name TEXT,
                date TEXT, players INTEGER, details_json TEXT);
            CREATE TABLE standings (tournament_id TEXT, player_id TEXT, placing INTEGER,
                wins INTEGER, losses INTEGER, ties INTEGER, deck_id TEXT, deck_name TEXT,
                decklist_json TEXT, dropped_round INTEGER, PRIMARY KEY(tournament_id, player_id));
            CREATE TABLE pairings (tournament_id TEXT, round INTEGER, phase INTEGER, match TEXT,
                player1 TEXT, player2 TEXT, winner TEXT,
                PRIMARY KEY(tournament_id,round,phase,match,player1,player2));
        """)
        con.execute("INSERT INTO tournaments VALUES (?,?,?,?,?,?,?)",
                    ("e0", "PTCG", "standard", "e0", "2026-09-01T00:00:00Z", 16, "{}"))
        legal = json.dumps({"trainer": [{"name": "Core", "count": 1}],
                            "energy": [{"name": "Grass Energy", "count": 59}]})
        bad = "{broken" if invalid else json.dumps({"trainer": [{"name": sorted(ACE_SPEC_CARDS)[0], "count": 2}],
                                                     "energy": [{"name": "Grass Energy", "count": 58}]})
        con.executemany("INSERT INTO standings VALUES (?,?,?,?,?,?,?,?,?,?)", [
            ("e0", "p", 1, 1, 1, 0, "a", "A", legal, 2),
            ("e0", "q", 2, 1, 1, 0, "b", "B", None, None),
            ("e0", "r", 3, 1, 1, 0, "c", "C", bad, None),
            ("e0", "q2", 4, 1, 1, 0, "b", "B", legal, None),
            ("e0", "ghost", 5, 1, 1, 0, "unmapped-id", None, None, None),
        ])
        con.executemany("INSERT INTO pairings VALUES (?,?,?,?,?,?,?)", [
            ("e0", 1, 1, "", "p", "q", "p"),
            ("e0", 2, 1, "", "p", "q", "q"),
            ("e0", 3, 1, "", "p", "q", "0"),
            ("e0", 4, 1, "", "p", "", "p"),
            ("e0", 5, 1, "", "p", "q", "-1"),
            ("e0", 6, 1, "", "p", "q", "p"),
            ("e0", 1, 2, "", "p", "missing", "p"),
            ("e0", 7, 1, "", "p", "r", "p"),
            ("e0", 8, 1, "", "q", "r", "q"),
            ("e0", 9, 1, "", "p", "q2", "p"),
            ("e0", 10, 1, "", "p", "ghost", "p"),
        ])
        con.commit()
    return path


def test_readonly_identity_repeated_rounds_and_both_focal_views(tmp_path):
    path = _database(tmp_path / "corpus.db")
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    corpus = extract_corpus(path, exclude_after_drop=False)
    keys = [tuple(row["key"]) for row in corpus["pairings"]]
    assert ("e0", 1, 1, "", "p", "q") in keys
    assert ("e0", 2, 1, "", "p", "q") in keys
    assert len(keys) == len(set(keys))
    views = [row for row in corpus["pairing_views"] if row["pairing_key"][1] == 1]
    assert {row["player"] for row in views} == {"p"}
    assert sum(row["weight"] for row in views) == 1.0
    assert corpus["pairings"][0]["eligible_view_count"] == 1
    mirrored = [row for row in corpus["pairing_views"] if row["pairing_key"][1] == 9]
    assert {row["player"] for row in mirrored} == {"p", "q2"}
    assert {row["weight"] for row in mirrored} == {0.5}
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    assert not (tmp_path / "corpus.db-journal").exists()


def test_exclusions_distinguish_missing_join_bye_tie_winner_drop_and_list(tmp_path):
    corpus = extract_corpus(_database(tmp_path / "corpus.db"))
    exclusion = {tuple(row["key"]): row["exclusion"] for row in corpus["pairings"]}
    assert exclusion[("e0", 3, 1, "", "p", "q")] == "tie"
    assert exclusion[("e0", 4, 1, "", "p", "")] == "bye"
    assert exclusion[("e0", 5, 1, "", "p", "q")] == "invalid_winner"
    assert exclusion[("e0", 6, 1, "", "p", "q")] == "after_drop"
    assert exclusion[("e0", 1, 1, "", "p", "q")] is None
    assert exclusion[("e0", 2, 1, "", "p", "q")] is None
    assert exclusion[("e0", 8, 1, "", "q", "r")] == "no_legal_focal_list"
    assert exclusion[("e0", 10, 1, "", "p", "ghost")] == "unknown_archetype"
    assert corpus["audit"]["pairing_exclusions"]["missing_join"] == 1
    assert corpus["audit"]["pairing_exclusions"]["no_legal_focal_list"] == 1
    assert corpus["audit"]["lists"]["missing_list"] == 2
    assert corpus["audit"]["lists"]["invalid_list"] == 1


def test_one_underlying_key_retains_repeated_pairings(tmp_path):
    corpus = extract_corpus(_database(tmp_path / "corpus.db"))
    repeated = [tuple(row["key"]) for row in corpus["pairings"] if row["player1"] == "p" and row["player2"] == "q"]
    assert len(repeated) == 5
    assert len(set(repeated)) == len(repeated)


def test_malformed_json_is_counted_as_an_invalid_list(tmp_path):
    corpus = extract_corpus(_database(tmp_path / "corpus.db", invalid=True))
    assert corpus["audit"]["lists"]["invalid_list"] == 1
    assert all(row["player"] != "r" for row in corpus["lists"])


def test_event_filter_keeps_global_dates_but_does_not_read_other_outcomes(tmp_path):
    path = _database(tmp_path / "corpus.db")
    with sqlite3.connect(path) as con:
        con.execute("INSERT INTO tournaments VALUES (?,?,?,?,?,?,?)",
                    ("e-test", "PTCG", "standard", "test", "2026-10-01T00:00:00Z", 16, "{}"))
        legal = json.dumps({"trainer": [{"name": "Core", "count": 1}],
                            "energy": [{"name": "Grass Energy", "count": 59}]})
        con.executemany("INSERT INTO standings VALUES (?,?,?,?,?,?,?,?,?,?)", [
            ("e-test", "p", 1, 99, 0, 0, "a", "A", legal, None),
            ("e-test", "q", 2, 0, 99, 0, "b", "B", None, None),
        ])
        con.execute("INSERT INTO pairings VALUES (?,?,?,?,?,?,?)",
                    ("e-test", 1, 1, "", "p", "q", "p"))
        con.commit()
    corpus = extract_corpus(path, include_event_ids=["e0"])
    assert {event["id"] for event in corpus["events"]} == {"e0", "e-test"}
    assert {row["event"] for row in corpus["pairings"]} == {"e0"}
    assert {row["event"] for row in corpus["records"]} == {"e0"}
    assert corpus["audit"]["included_event_ids"] == ["e0"]
