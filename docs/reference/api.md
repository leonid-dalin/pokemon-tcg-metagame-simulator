# API reference

The FastAPI application is defined in `src/api/main.py`. The base path is `/api/v1`

## Authentication

Set `API_TOKEN` in the API environment to protect prediction and task endpoints. Clients send the value in the `X-API-Token` header. When `API_TOKEN` is empty, the middleware allows these requests without a token

## Endpoints

### `POST /api/v1/predict`

Queues a simulation and returns HTTP `202`:

```json
{"task_id": "huey-task-id", "status": "enqueued"}
```

The request body must contain at least two unique `deck_names` and a square, zero-sum `matchup_matrix`. Matrix diagonals must be exactly `0.5`

Important fields:

| Field | Default | Constraints |
| --- | --- | --- |
| `job_id` | `unknown` | Maximum 64 characters |
| `deck_names` | required | At least 2 names, maximum 64 matrix rows |
| `matchup_matrix` | required | Square, values from `0.0` to `1.0`, mirrored pairs sum to `1.0` |
| `tournament_style` | `pure_swiss` | `pure_swiss` or `championship_series` |
| `match_format` | `BO3` | `BO1` or `BO3` |
| `total_players` | `256` | From 4 to 8192 |
| `precision_tier` | `3 - STANDARD` | Five precision tiers |
| `global_tie_rate` | `0.15` | From `0.0` to `0.5` |
| `min_sample_threshold` | `10` | From 1 to 100 |
| `use_tie_convergence` | `true` | Boolean |
| `use_drop_feature` | `false` | Boolean |

`user_meta_spec` accepts numeric shares or `{ "exact": number }` and `{ "min": number, "max": number }` objects, with values from `0.0` to `1.0`

### `GET /api/v1/tasks/{task_id}`

Returns one of these shapes:

```json
{"task_id": "huey-task-id", "status": "processing"}
```

```json
{"task_id": "huey-task-id", "status": "complete"}
```

A failed task returns `status: "failed"` and an error field. The API hides the failure detail when the request does not satisfy the token policy

### `GET /api/v1/tasks/{task_id}/stream`

Returns an SSE stream. The UI uses it for progress and completion messages. The stream sends `processing`, `complete`, `failed`, or `timeout` status data and closes after ten minutes

## Rate limits

The API applies `10/minute` to prediction requests and `60/minute` to task status and stream requests. Limits use the configured Redis URL

## Operational headers

Responses include `X-Content-Type-Options`, `X-Frame-Options`, `Strict-Transport-Security`, and `Content-Security-Policy`. CORS allows the checked-in Streamlit origin at `http://localhost:8501`
