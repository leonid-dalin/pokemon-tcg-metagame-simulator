# Code reference

This page lists the current entry points for the simulator. Read it with the source files named here; it is a navigation aid, not a generated API schema

## API and worker

### `src/api/main.py`

- `lifespan`: starts Redis access and triggers one guarded daily pipeline task
- `start_prediction`: validates and queues `PredictionRequest` payloads
- `get_task_status`: returns `processing`, `complete`, or `failed`
- `stream_task_progress`: streams progress and completion events over SSE
- `ApiTokenAuthMiddleware`: checks `X-API-Token` for protected requests when `API_TOKEN` is set

### `src/api/models.py`

- `PredictionRequest`: validates deck names, matchup matrix shape, field constraints, tournament settings, and feature flags
- `PrecisionTier`: maps `BULLET`, `BLITZ`, `STANDARD`, `EXHAUSTIVE`, and `MAXIMUM` to iteration counts
- `DeckRecommendation`: defines static solver recommendation fields
- `ScrapedMatrix`: validates scraped matchup data before it reaches the input file

### `src/worker/queue.py`

- `execute_simulation_job`: runs the solver, tournament structure selection, Monte Carlo engine, and optional BDIF reports
- `automated_daily_pipeline`: fetches and validates the live matchup matrix every two hours at minute 0 (`0 */2 * * *`). Each run makes outbound requests to LimitlessTCG
- `ingest_limitless_results`: stores Limitless tournaments and builds optional model artefacts
- `_build_bdif_report_addons`: builds Best-60 and H1 reports when the card model flag is enabled

## Core modules

### `src/core/config.py`

Holds input and output paths, simulation defaults, tournament structure thresholds, tier cutoffs, and BDIF evidence thresholds

### `src/core/data.py`

Loads matchup JSON, fills missing matchup evidence, clusters matchup profiles, and computes deck dominance

### `src/core/scraper.py`

Discovers live matchup URLs, fetches HTML, normalises archetype labels, parses matchup pages, and assembles the validated matrix

### `src/core/runtime.py`

Reports the core limit available to the current host or container. `MAX_CORES` can override that value

### `src/core/telemetry.py`

Creates the OpenTelemetry provider and sends spans to `OTEL_EXPORTER_OTLP_ENDPOINT`

## Engines

### `src/evolution/engine.py`

Runs replicator dynamics, applies selection and mutation, and writes generation history

### `src/evolution/analysis.py`

Calculates convergence, final tiers, matchup cycles, and deck similarity

### `src/tournament/solver.py`

Resolves field constraints, calculates Swiss rounds, chooses Championship Series structure, and produces static recommendations

### `src/tournament/monte_carlo.py`

Builds posterior matchup matrices, marks insufficient evidence, constructs the matchup panel, and calls the Rust engine

### `src/tournament/tcg_engine/src/lib.rs`

Implements the compiled tournament loops and Swiss pairing logic through PyO3 and Rayon

## Data and card model

### `src/ingestion/store.py`

Stores tournaments, standings, pairings, decklists, canonical deck names, and card evidence in SQLite. Read preparation handles legacy database schemas

### `src/ingestion/client.py`

Calls the Limitless API with `LIMITLESS_API_KEY` in the `X-Access-Key` header

### `src/ingestion/model.py`

Fits card covariates, computes H1 reports, selects empirical panel decks, and builds legality-aware Best-60 recommendations

### `src/ingestion/aggregate.py`

Aggregates stored matchup rows into the JSON artefact consumed by the standard loader

## User interfaces

### `src/ui/cli.py`

Runs replicator, tournament, prediction, and batch workflows and writes local output artefacts

### `src/ui/app.py`

Renders the Streamlit dashboard, builds `PredictionRequest` payloads, sends them to the API, and reads the SSE task stream
