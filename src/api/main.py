import asyncio
import json
import os
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from sse_starlette.sse import EventSourceResponse

from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from src.api.models import PredictionRequest
from src.worker.queue import execute_simulation_job, automated_daily_pipeline, huey

# ==========================================
# 0. Useful Func(s)
# ==========================================
def numpy_safe_encoder(obj):
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

# ==========================================
# 1. Lifecycle Management
# ==========================================
@asynccontextmanager
async def lifespan(_: FastAPI):
    """
    Handles startup and shutdown events.
    Triggers the automated scraper immediately on startup to ensure data parity.
    """
    print("Starting automated daily Limitless TCG data scrape...")
    try:
        automated_daily_pipeline()
    except Exception as e:
        print(f"Startup scrape failed: {e}")

    yield
    print("Shutting down Pokémon TCG Simulator API...")

# ==========================================
# 2. Rate Limiter Configuration
# ==========================================
redis_url = "redis://localhost:6379/0"
if "REDIS_URL" in os.environ:
    import os
    redis_url = os.environ["REDIS_URL"]

limiter = Limiter(
    key_func=get_remote_address,
    storage_uri=redis_url,
    strategy="fixed-window"
)
app = FastAPI(
    title="Pokémon TCG Metagame Simulator API",
    description="RESTful gateway for stochastic tournament modeling and metagame evolution",
    lifespan=lifespan,
    version="0.1.0",
    docs_url="/docs/",
    redoc_url=None)
app.state.limiter = limiter

# ==========================================
# 3. Middleware & Exception Handlers
# ==========================================
async def rate_limit_custom_handler(request: Request, exc: Exception) -> Response:
    """
    Handles slowapi rate limits.
    Strictly typed with 'Exception' and 'Response' to satisfy FastAPI's ASGI signatures.
    """
    # Explicitly check for None to satisfy the linter's 'Address | None' warning
    client_ip = "Unknown IP"
    if request.client is not None and hasattr(request.client, 'host'):
        client_ip = request.client.host

    route = request.url.path
    detail_msg = str(exc) if exc else "Rate limit exceeded."

    return JSONResponse(
        status_code=429,
        content={
            "error": "Too Many Requests",
            "message": f"Request limit reached for {client_ip} on {route}. Please slow down.",
            "detail": detail_msg
        },
    )

app.add_exception_handler(RateLimitExceeded, rate_limit_custom_handler)

app.add_middleware(
    CORSMiddleware, # type: ignore
    allow_origins=["http://localhost:8501"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
    allow_credentials=True,
)

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"  # Prevent MIME-type sniffing
        response.headers["X-Frame-Options"] = "DENY"    # Prevent Clickjacking (disallow iframe embedding)
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"   # Enforce HTTPS
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        return response

app.add_middleware(SecurityHeadersMiddleware)

# ==========================================
# 4. API Endpoints
# ==========================================
@app.post("/api/v1/predict")
@limiter.limit("10/minute")
async def start_prediction(request: Request, payload: PredictionRequest):
    """Enqueues a heavy Monte Carlo tournament simulation."""
    _ = request
    payload_dict = payload.model_dump()
    task = execute_simulation_job(payload_dict)
    job_id = payload_dict.get("job_id", "unknown_job")
    huey.storage.put_data(f"link_{task.id}", job_id.encode('utf-8'))
    return JSONResponse(
        status_code=202,
        content={"task_id": task.id, "status": "enqueued"}
    )

@app.get("/api/v1/tasks/{task_id}")
@limiter.limit("60/minute")
async def get_task_status(request: Request, task_id: str):
    """Legacy polling endpoint for task status"""
    _ = request
    result = await asyncio.to_thread(huey.result, task_id, blocking=False)
    if result is None:
        return JSONResponse(
            status_code=200,
            content={"task_id": task_id, "status": "processing"}
        )
    return JSONResponse(
        status_code=200,
        content={"task_id": task_id, "status": "complete"}
    )

@app.get("/api/v1/tasks/{task_id}/stream")
@limiter.limit("60/minute")
async def stream_task_progress(request: Request, task_id: str):
    """
    SSE Endpoint for real-time progress updates.
    Restored to use the Huey storage peek method to correctly read progress.
    """

    async def event_generator():
        try:
            while True:
                # Halt if the Streamlit client drops the connection
                if await request.is_disconnected():
                    break

                # 1. Check if the background math is completely finished
                result = await asyncio.to_thread(huey.result, task_id, blocking=False)

                if isinstance(result, Exception):
                    yield {
                        "event": "message",
                        "data": json.dumps({"status": "failed", "error": str(result)})
                    }
                    break
                elif result is not None:
                    yield {
                        "event": "message",
                        "data": json.dumps({"status": "complete", "data": result}, default=numpy_safe_encoder)
                    }
                    break

                # 2. Fetch the linked job_id to read the correct progress mailbox
                link_bytes = await asyncio.to_thread(huey.storage.peek_data, f"link_{task_id}")
                job_id = link_bytes.decode('utf-8') if isinstance(link_bytes, bytes) else "unknown_job"

                progress_bytes = await asyncio.to_thread(huey.storage.peek_data, f"prog_{job_id}")
                current_pct = int(progress_bytes.decode('utf-8')) if isinstance(progress_bytes, bytes) else 0

                yield {
                    "event": "message",
                    "data": json.dumps({"status": "processing", "progress": current_pct})
                }

                # 3. Yield control back to the event loop
                await asyncio.sleep(0.4)

        except Exception as e:
            print(f"SSE stream error: {e}")
            yield {
                "event": "message",
                "data": json.dumps({"status": "failed", "error": "Stream disconnected internally"})
            }

    return EventSourceResponse(
        event_generator(),
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )