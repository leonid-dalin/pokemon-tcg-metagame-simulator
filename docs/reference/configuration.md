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
| `BDIF_MIN_MATCHES` | `1000` | Minimum total matchup evidence |
| `BDIF_COVERAGE_RATIO` | `0.6` | Required covered evidence ratio |
| `BDIF_PAIR_MIN_GAMES` | `250` | Pair count used for reliable evidence |
| `BDIF_PRIOR_STRENGTH` | `10.0` | Posterior prior strength |
| `BDIF_POSTERIOR_DRAWS` | `200` | Maximum posterior matrices sampled per report |
| `BDIF_MIN_ITERATIONS_PER_DRAW` | `100` | Minimum tournament iterations per posterior matrix |
| `BDIF_MIN_INTERVAL_DRAWS` | `50` | Minimum posterior matrices for reported intervals |
| `BDIF_PANEL_SHARE_THRESHOLD` | `0.03` | Empirical panel share threshold |
| `BDIF_PANEL_MAX_DECKS` | `10` | Panel deck cap |
| `LIMITLESS_INGESTION_ENABLED` | `False` | Limitless ingestion switch |
| `LIMITLESS_BACKFILL_TOURNAMENTS` | `200` | Ingestion tournament limit |
| `BDIF_USE_CARD_MODEL` | `False` | Card-model report switch |
| `BDIF_PANEL_DECKS` | `Crustle`, `N's Zoroark` | Fallback panel names |

## Environment variables

| Variable | Used by | Default |
| --- | --- | --- |
| `REDIS_URL` | API and worker | `redis://localhost:6379/0` in the API, `redis://localhost:6379/?db=0` in the worker |
| `API_URL` | Streamlit UI | `http://localhost:8000/api/v1` |
| `API_TOKEN` | API middleware | Empty, which disables token checking |
| `LIMITLESS_API_KEY` | Limitless client | Empty, which creates an unauthenticated client |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | Telemetry | `http://localhost:4317` |
| `MAX_CORES` | Runtime core limit | Explicit override; otherwise cgroup v2 quota, cgroup v1 quota, then half of `os.cpu_count()` with a minimum of 1 |

Do not commit API keys or tokens. Limitless sends `LIMITLESS_API_KEY` in the `X-Access-Key` header
