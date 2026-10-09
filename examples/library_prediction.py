from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api.models import PredictionRequest
from src.core.config import MIN_GAMES
from src.core.data import load_matchup_data
from src.tournament.solver import predict_best_decks


def main() -> None:
    deck_names, matrix, _ = load_matchup_data(str(PROJECT_ROOT / "data" / "input" / "ea_input.json"), MIN_GAMES)
    request = PredictionRequest(
        job_id="library-example",
        deck_names=deck_names,
        matchup_matrix=matrix.tolist(),
        total_players=512,
        match_format="BO3",
    )
    result = predict_best_decks(request)
    for rank, recommendation in enumerate(result["recommendations"][:5], start=1):
        print(
            f"{rank}. {recommendation['deck']}: meta score {recommendation['base_meta_score']:.1f}, "
            f"expected win rate {recommendation['expected_win_rate']:.2%}"
        )


if __name__ == "__main__":
    main()
