# Run with Docker Compose

Use this recipe to start the dashboard, API, worker, Redis broker, and Jaeger tracing stack

## Start

```bash
docker compose up -d --build
```

The services use the repository root as their build context. `.dockerignore` excludes Python caches, virtual environments, Rust build output, and other local caches from that context

Check service state:

```bash
docker compose ps
```

Open:

- `http://localhost:8501` for Streamlit
- `http://localhost:8000/docs/` for the FastAPI OpenAPI page
- `http://localhost:16686` for Jaeger

## Configuration

Compose passes `REDIS_URL` to the API and worker. It defaults to `redis://redis:6379/?db=0`. The UI uses `API_URL`, which defaults to `http://api:8000/api/v1` inside the Compose network

Set an alternative Redis URL before starting the stack:

```bash
REDIS_URL=redis://redis.example:6379/0 docker compose up -d --build
```

When the API uses `API_TOKEN`, pass the same secret to both `api` and `ui`. The UI sends it as `X-API-Token` on prediction, task-stream, and BDIF status requests

Add the variable to both service environment blocks in deployment configuration, for example:

```yaml
services:
  api:
    environment:
      - API_TOKEN=${API_TOKEN}
  ui:
    environment:
      - API_TOKEN=${API_TOKEN}
```

Set `API_TOKEN` in the deployment environment before starting the stack. The checked-in Compose file does not set this variable; do not commit a secret

## Check logs

```bash
docker compose logs -f api worker
```

The API starts one scraper task when it obtains the Redis startup lock. Other API processes skip that task while the lock is held

## Stop

```bash
docker compose down
```

Use `docker compose down -v` only when deleting the shared data volume is intended