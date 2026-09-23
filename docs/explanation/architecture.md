# Architecture

The repository has two execution paths. The local CLI runs the simulation in one process. Docker Compose separates the web request, queue, computation, data initialisation, and telemetry services

## Compose services

```text
Streamlit UI -> FastAPI API -> Redis -> Huey worker
       |             |                    |
       +-------------+--------------------+
                     Jaeger tracing
```

- `ui` renders the Streamlit dashboard and sends requests to `API_URL`
- `api` validates `PredictionRequest`, enqueues Huey jobs, exposes status, and serves SSE progress
- `worker` loads the matchup data and runs the solver and Monte Carlo engine
- `redis` stores Huey jobs and progress state
- `init-data` creates the shared data directories and assigns their ownership
- `jaeger` receives OpenTelemetry spans

The services share `/app/data` through the `shared_data` volume. The worker is limited to two processes and 2 GB of memory in the Compose file

## Request flow

1. The UI creates a validated `PredictionRequest`
2. The API applies token and rate-limit middleware
3. The API queues `execute_simulation_job`
4. The worker loads the matrix and runs `predict_best_decks`
5. The worker selects tournament structure and runs `run_monte_carlo_analytics`
6. The API exposes task status and streams progress through Redis Pub/Sub

The SSE stream has a ten-minute lifetime. It first reads the stored progress value, then listens for Pub/Sub updates, which lets a reconnecting client recover its latest state

## Data flow

The default data path is `data/input/ea_input.json`. The loader returns deck names, a NumPy win-rate matrix, and matchup details. The matrix contract requires values from `0.0` to `1.0`, exact `0.5` mirrors, and paired values that sum to `1.0`

The daily pipeline fetches live Limitless matchup pages, validates the resulting matrix with `ScrapedMatrix`, and writes the JSON input atomically. The separate Limitless ingestion task stores tournament data in SQLite and can produce card-model artefacts for opt-in BDIF reporting

## Engines

The replicator engine updates deck frequencies over generations with matchup payoffs, selection pressure, mutation, and optional noise

The tournament engine selects pure Swiss or Championship Series structure, then calls the Rust `tcg_engine` through Python bindings. Rayon threads are capped by the detected container core limit

The BDIF card model is isolated behind `BDIF_USE_CARD_MODEL`. Its failures are logged and do not abort the core tournament simulation
