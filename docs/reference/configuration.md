# Configuration reference

The main constants live in `src/core/config.py`. Environment variables are read by the API, UI, worker, telemetry, and Limitless client modules

## Data and simulation constants

| Name | Current value | Meaning |
| --- | --- | --- |
| `INPUT_DATA` | `data/input/ea_input.json` | Default matchup matrix |
| `OUTPUT_DIR` | `output/` | CLI output root |
| `SIMULATION_MODE` | `replicator` | Default CLI engine |
| `RNG_SEED` | `1312` | Default reproducibility seed |
| `MIN_GAMES` | `100` | Loader match threshold |
| `MAX_GENERATIONS` | `100000` | Replicator generation limit |
| `STABILITY_THRESHOLD` | `5e-5` | Replicator stability threshold |
| `EXTINCTION_THRESHOLD` | `1e-6` | Inactive deck threshold |
| `MUTATION_RATE` | `1e-4` | Deck reintroduction floor |
| `NOISE_SCALE` | `0.0` | Payoff noise scale |
| `SELECTION_PRESSURE` | `1` | Population response pressure |

## BDIF constants

| Name | Current value | Meaning |
| --- | --- | --- |
| `BDIF_MIN_MATCHES` | `1000` | Minimum total matchup evidence per deck |
| `BDIF_COVERAGE_RATIO` | `0.6` | Required covered evidence ratio |
| `BDIF_PAIR_MIN_GAMES` | `250` | Pair count used for reliable evidence |
| `BDIF_PRIOR_STRENGTH` | `10.0` | Posterior prior strength |
| `BDIF_POSTERIOR_DRAWS` | `200` | Maximum posterior matrices sampled per report |
| `BDIF_MIN_ITERATIONS_PER_DRAW` | `100` | Minimum tournament iterations per posterior matrix |
| `BDIF_MIN_INTERVAL_DRAWS` | `50` | Minimum posterior matrices for reported intervals |
| `BDIF_FIELD_POSTERIOR_DRAWS` | `2000` | Field-posterior draws |
| `BDIF_PANEL_SHARE_THRESHOLD` | `0.03` | Empirical panel share threshold |
| `BDIF_PANEL_MAX_DECKS` | `10` | Panel deck cap |
| `BDIF_CARD_MIN_PLAYERS` | `50` | Minimum player appearances for card selection |
| `BDIF_CARD_MIN_WITHIN_RATE` | `0.05` | Minimum within-archetype card presence |
| `BDIF_CARD_MAX_WITHIN_RATE` | `0.95` | Maximum within-archetype card presence |
| `BDIF_CARD_MAX_ABS_LOGIT` | `5.0` | Maximum absolute card-model logit |
| `BDIF_CARD_PRIOR_SD` | `1.0` | Prior standard deviation of each card-model logit coefficient |
| `BDIF_BEST60_MIN_MODEL_LISTS` | `100` | Legal lists an archetype needs before Best-60 scores card counts |
| `BDIF_BEST60_MIN_SLOT_LISTS` | `30` | Lists that must play, and lists that must miss, a card-count slot before it is scored |
| `BDIF_BEST60_PRIOR_GRID` | `(0.02, 0.05, 0.1, 0.2, 0.5)` | Prior widths Best-60 compares on held-out events |
| `BDIF_BEST60_STRENGTH_PRIOR_GAMES` | `10` | Wins and losses added to each player's record before measuring strength |
| `BDIF_BEST60_APPLY_PROBABILITY` | `0.7` | Chance of helping a swap needs before Best-60 applies it |
| `BDIF_BEST60_LEAN_PROBABILITY` | `0.5` | Chance of helping above which a swap is listed as leaning |
| `BDIF_BEST60_MAX_SWAPS` | `12` | Most swaps Best-60 applies to one list |
| `BDIF_BEST60_PREREQUISITE_SHARE` | `0.95` | Share of a card's lists a companion must appear in to be required alongside it |
| `BDIF_BEST60_TREND_DAYS` | `21` | Days at each end of the snapshot that trends compare |
| `BDIF_BEST60_TREND_POINTS` | `0.15` | Play-rate change that makes a card rising or falling |
| `LIMITLESS_INGESTION_ENABLED` | `False` | Limitless ingestion switch |
| `LIMITLESS_BACKFILL_TOURNAMENTS` | `200` | Ingestion tournament limit |
| `BDIF_USE_CARD_MODEL` | `False` | Card-model report switch |
| `BDIF_PANEL_DECKS` | `Crustle`, `N's Zoroark` | Fallback panel names |

## Environment variables

| Variable | Used by | Accepted values or default |
| --- | --- | --- |
| `REDIS_URL` | API and worker | Redis URL; API default `redis://localhost:6379/0`, worker default `redis://localhost:6379/?db=0` |
| `API_URL` | Streamlit UI | URL; default `http://localhost:8000/api/v1` |
| `API_TOKEN` | API and Streamlit UI | Any non-empty shared secret; empty disables token checking |
| `LIMITLESS_API_KEY` | Limitless client | Secret key; empty creates an unauthenticated client |
| `BDIF_USE_CARD_MODEL` | BDIF service | Boolean: `1`, `true`, `yes`, `on`, `0`, `false`, `no`, or `off`; default `False` |
| `LIMITLESS_INGESTION_ENABLED` | BDIF service | Same boolean values; default `False` |
| `BDIF_DB_PATH` | BDIF service | Filesystem path; default `data/limitless.db` |
| `BDIF_ARTIFACT_DIR` | BDIF service | Directory for `limitless_input.json` and `limitless_model_input.json`; default `data/input` |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | Telemetry | Endpoint; default `http://localhost:4317` |
| `MAX_CORES` | Runtime core limit | Explicit override; otherwise cgroup v2 quota, cgroup v1 quota, then half of `os.cpu_count()` with a minimum of 1 |

When `API_TOKEN` is set, the API protects prediction, task-stream, and BDIF status requests with `X-API-Token`. Compose deployments must pass the same value to both the `api` and `ui` services. Do not commit API keys or tokens. Limitless sends `LIMITLESS_API_KEY` in the `X-Access-Key` header