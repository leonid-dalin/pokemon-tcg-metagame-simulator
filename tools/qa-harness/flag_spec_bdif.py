import json, os, tempfile

import src.core.config as config
import src.worker.queue as queue
from src.core.data import load_matchup_data
from src.ingestion.model import fit_card_model, model_artifact, select_model_cards
from src.ingestion.model import PlayerObservation

FLAGS = ["src.worker.queue.BDIF_USE_CARD_MODEL"]


def _seed_card_model_artifact(root):
    observations = []
    for index in range(120):
        has_tech = index % 2 == 0
        a_cards = frozenset({"Core", "Tech"} if has_tech else {"Core"})
        b_cards = frozenset({"Core"})
        result = int((index % 8) < (6 if has_tech else 3))
        observations.append(PlayerObservation("a", "b", a_cards, b_cards, result))
        observations.append(PlayerObservation("b", "a", b_cards, a_cards, 1 - result))
    artifact = model_artifact(fit_card_model(observations, select_model_cards(observations)))
    path = os.path.join(root, "data", "input")
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "limitless_model_input.json"), "w", encoding="utf-8") as handle:
        json.dump(artifact, handle)
    with open(os.path.join(path, "baseline.json"), "w", encoding="utf-8") as handle:
        json.dump({"archetypes": ["a", "b"], "win_rate_matrix": {
            "a": {"a": {"win_rate": 0.5, "match_count": 0}, "b": {"win_rate": 0.6, "match_count": 500}},
            "b": {"a": {"win_rate": 0.4, "match_count": 500}, "b": {"win_rate": 0.5, "match_count": 0}}}}, handle)
    return os.path.join(path, "baseline.json")


def smoke():
    root = tempfile.mkdtemp()
    baseline = _seed_card_model_artifact(root)
    cwd = os.getcwd()
    os.chdir(root)
    try:
        previous = queue.INPUT_DATA
        queue.INPUT_DATA = baseline
        try:
            chosen = queue._simulation_input_path()
            names, matrix, details = load_matchup_data(chosen, config.MIN_GAMES)
        finally:
            queue.INPUT_DATA = previous
    finally:
        os.chdir(cwd)
    if not names:
        return (f"_simulation_input_path() chose {os.path.basename(chosen)}; "
                f"load_matchup_data returned 0 decks at MIN_GAMES={config.MIN_GAMES}")
    return None
