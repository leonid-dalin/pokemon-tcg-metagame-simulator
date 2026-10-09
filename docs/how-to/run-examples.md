# Run the Python examples

Run these examples from the repository root after completing the [local development setup](local-development.md). The commands use the checked-in input and snapshot files

## Library prediction

`library_prediction.py` loads `data/input/ea_input.json`, validates a `PredictionRequest`, and calls the static recommendation solver. It prints the top five decks ranked by base meta score, each with its expected win rate

```bash
python examples/library_prediction.py
```

The script loads the full current input, so it may print more log output than the five recommendation rows

## Snapshot summary

`snapshot_summary.py` opens `data/limitless.db` read-only, checks its SHA-256 value against `data/snapshot/manifest.json`, and reports the tournament, standings, pairing, and decklist counts

```bash
python examples/snapshot_summary.py
```

The database is tracked with Git LFS. Run `git lfs pull` before this command when `data/limitless.db` is an LFS pointer

## API prediction

Start the Compose services before running `api_prediction.py`:

```bash
docker compose up -d --build
python examples/api_prediction.py
docker compose down
```

The script submits a 32-player `1 - BULLET` prediction to `POST /api/v1/predict`, follows the task's server-sent event stream, and prints the first five recommendations after the Huey worker finishes

The default API URL is `http://127.0.0.1:8000/api/v1`. To use an API that runs elsewhere, set the `API_URL` environment variable to its base URL before you run the script

If the API has an `API_TOKEN`, set the same value in the environment. The script sends it as `X-API-Token` on both the prediction and the stream requests

The API example requires running API, worker, and Redis services. It is not a replacement for the local library example

## Output

The library example prints one row per deck with its base meta score and expected win rate. The API example prints JSON with the `task_id` and the first five recommendations. The snapshot example prints JSON with `database_sha256`, table counts, and `decklists`
