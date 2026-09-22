# Run the simulator

This tutorial starts with a clone and ends with a local tournament prediction. Use Docker if you want the dashboard, or use the Python CLI for a local batch run

## Before you start

Install Docker Desktop for the Compose path. For the CLI path, use Python 3.12 and a Rust toolchain that Maturin can call

The repository uses `data/input/ea_input.json` as its default matchup input. Run commands from the repository root

## Start the application

```bash
docker compose up -d --build
```

Wait for the services to start, then open `http://localhost:8501`. The UI sends prediction requests to the API, the API places work on Huey through Redis, and the worker returns progress and results through the API

The API exposes OpenAPI documentation at `http://localhost:8000/docs/`

## Run a prediction

In the dashboard, choose the archetypes and field constraints, then submit the event. The API validates the deck list, matchup matrix, tournament style, player count, and precision tier before it queues the job

The default precision tier is `3 - STANDARD`, which maps to 25,000 Monte Carlo iterations. The available tiers are:

- `1 - BULLET`: 1,000 iterations
- `2 - BLITZ`: 10,000 iterations
- `3 - STANDARD`: 25,000 iterations
- `4 - EXHAUSTIVE`: 100,000 iterations
- `5 - MAXIMUM`: 250,000 iterations

## Run the same workflow from the CLI

```bash
python -m src.ui.cli -i data/input/ea_input.json --predict --players 512
```

The predictor loads the matrix, builds a validated `PredictionRequest`, resolves the field distribution, calculates Swiss structure, and logs the ranked recommendations

## Inspect output

A standard simulation creates a timestamped directory below `output/` with files such as:

- `ess_equilibrium.csv`
- `final_tiers.json`
- `simulation_trace.jsonl`
- `metagame_history_full.csv`
- `deck_similarity.json`
- `metagame_evolution.html`
- `matchup_heatmap.html`
- `matchup_network.html`

Plot files are omitted when you pass `--no-plot`

## Stop the application

```bash
docker compose down
```

Add `-v` only when you deliberately want to remove the shared data volume
