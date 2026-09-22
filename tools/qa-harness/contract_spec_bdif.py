import json, os, sqlite3, tempfile

from src.core.config import MIN_GAMES
from src.core.data import load_matchup_data
from src.ingestion.aggregate import build_artifact
from src.ingestion.model import fit_model, model_artifact


def _write(artifact):
    path = os.path.join(tempfile.mkdtemp(), "artifact.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(artifact, handle)
    return path


def _produce_card_model():
    observations = [("a", "b", 1)] * 80 + [("b", "a", 0)] * 20
    inclusion = {"a": {"Misty": 1.0}, "b": {"Misty": 0.0}}
    return model_artifact(fit_model(observations, inclusion))


def _produce_ingestion():
    class Store:
        def matchup_rows(self):
            return iter([("a", "b", "p1", "p1", "p2")] * 400 + [("b", "a", "p1", "p2", "p1")] * 400)

        def card_inclusion(self, archetype):
            return {}

    return build_artifact(Store())


def _degenerate(loaded):
    names, matrix, details = loaded
    if not names:
        return f"load_matchup_data returned 0 decks (MIN_GAMES={MIN_GAMES}); every archetype was filtered out"
    if not details:
        return "matchup_details is empty; posterior mode will have no evidence"
    return None


def _produce_legacy_store():
    root = tempfile.mkdtemp()
    path = os.path.join(root, "limitless.db")
    with sqlite3.connect(path) as conn:
        conn.executescript(
            "CREATE TABLE standings (tournament_id TEXT, player_id TEXT, deck_id TEXT, decklist_json TEXT);"
            "CREATE TABLE pairings (tournament_id TEXT, player1 TEXT, player2 TEXT, winner TEXT);"
        )
        conn.execute("INSERT INTO standings VALUES ('event', 'p1', 'a', '{\"pokemon\": []}')")
        conn.commit()
    return path


def _consume_legacy_store(path):
    from src.ingestion.store import LimitlessStore
    store = LimitlessStore(path, canonical_names=["a"])
    list(store.matchup_rows())
    with sqlite3.connect(path) as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(standings)")}
        deck_name = conn.execute("SELECT deck_name FROM standings").fetchone()[0]
    return {"has_deck_name": "deck_name" in columns, "backfilled": deck_name == "a"}


def _legacy_store_degenerate(loaded):
    if not loaded["has_deck_name"]:
        return "read path did not prepare the legacy schema"
    if not loaded["backfilled"]:
        return "read path did not backfill canonical deck names"
    return None


CASES = [
    {
        "name": "ingestion aggregate -> load_matchup_data (limitless_input.json)",
        "produce": _produce_ingestion,
        "consume": lambda a: load_matchup_data(_write(a), MIN_GAMES),
        "degenerate": _degenerate,
    },
    {
        "name": "card model artifact -> load_matchup_data (limitless_model_input.json, BDIF_USE_CARD_MODEL)",
        "produce": _produce_card_model,
        "consume": lambda a: load_matchup_data(_write(a), MIN_GAMES),
        "degenerate": _degenerate,
    },
    {
        "name": "legacy Limitless SQLite store -> reader schema preparation",
        "produce": _produce_legacy_store,
        "consume": _consume_legacy_store,
        "degenerate": _legacy_store_degenerate,
    },
]
