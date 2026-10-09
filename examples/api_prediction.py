import json
import os
import time
import urllib.parse
import urllib.request
import uuid

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api.models import PrecisionTier, PredictionRequest
from src.bdif.settings import simulation_input_path
from src.core.config import MIN_GAMES
from src.core.data import load_matchup_data

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000/api/v1").rstrip("/")


def build_request(url: str, payload: dict | None = None) -> urllib.request.Request:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"} if body else {}
    token = os.environ.get("API_TOKEN")
    if token:
        headers["X-API-Token"] = token
    return urllib.request.Request(url, data=body, headers=headers)


def request_json(url: str, payload: dict | None = None) -> dict:
    with urllib.request.urlopen(build_request(url, payload), timeout=30) as response:
        return json.load(response)


def main() -> None:
    deck_names, matrix, _ = load_matchup_data(simulation_input_path(), MIN_GAMES)
    job_id = str(uuid.uuid4())
    payload = PredictionRequest(
        job_id=job_id,
        deck_names=deck_names,
        matchup_matrix=matrix.tolist(),
        tournament_style="pure_swiss",
        match_format="BO3",
        total_players=32,
        precision_tier=PrecisionTier.BULLET,
        bdif_panel_decks=[deck_names[0]],
        bdif_archetypes=[deck_names[0]],
    )
    queued = request_json(f"{API_URL}/predict", payload.model_dump(mode="json"))
    task_id = urllib.parse.quote(queued["task_id"], safe="")
    stream_url = f"{API_URL}/tasks/{task_id}/stream?job_id={urllib.parse.quote(job_id)}"
    deadline = time.monotonic() + 900

    with urllib.request.urlopen(build_request(stream_url), timeout=900) as response:
        while time.monotonic() < deadline:
            line = response.readline()
            if not line:
                break
            if not line.startswith(b"data:"):
                continue
            result = json.loads(line[5:].strip())
            if result.get("status") == "complete":
                solver = result["data"]["solver_results"]
                print(
                    json.dumps(
                        {"task_id": queued["task_id"], "recommendations": solver["recommendations"][:5]}, indent=2
                    )
                )
                return
            if result.get("status") == "failed":
                raise RuntimeError(result.get("error", "Simulation failed"))
            if result.get("status") == "timeout":
                raise TimeoutError("The API task stream expired")

    raise TimeoutError("The API task did not complete within 15 minutes")


if __name__ == "__main__":
    main()
