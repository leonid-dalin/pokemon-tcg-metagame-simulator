import json, os, tempfile

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
]
