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

The API reads `API_TOKEN` when you provide one in its environment. Protected requests then need an `X-API-Token` header. The checked-in Compose file does not set this variable, so add it through your deployment configuration rather than committing a secret

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
