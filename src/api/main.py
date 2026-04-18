import asyncio
import logging
import json
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
from src.api.models import PredictionRequest
from src.worker.queue import execute_simulation_job, huey

@asynccontextmanager
async def lifespan(_: FastAPI):
    logging.info("Booting up: Triggering automated daily pipeline synchronously...")
    try:
        from src.worker.queue import automated_daily_pipeline
        automated_daily_pipeline.call_local()
        logging.info("Successfully hydrated Limitless TCG data matrix.")
    except Exception as e:
        logging.error(f"Failed to execute startup data pipeline: {e}")
    yield
    logging.info("Shutting down Startup API...")

app = FastAPI(
    title="Pokémon TCG Metagame Simulator API",
    description="RESTful gateway for stochastic tournament modeling and metagame evolution",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url=None)

# Allow Streamlit to talk to this API
app.add_middleware(
    CORSMiddleware, # type: ignore
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

def numpy_safe_encoder(obj):
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

@app.post("/api/v1/predict")
async def start_prediction(request: PredictionRequest):
    """Enqueue the simulation and return a Task ID."""
    # Convert Pydantic model to dict and enqueue
    task = execute_simulation_job(request.model_dump())

    return {
        "status": "processing",
        "task_id": task.id
    }

@app.get("/api/v1/tasks/{task_id}/stream")
async def stream_task_status(task_id: str, job_id: str, request: Request):
    """
    Server-Sent Events (SSE) endpoint.
    Streams the status of the Huey task over a single persistent connection.
    """

    async def event_generator():
        while True:
            if await request.is_disconnected():
                break

            # 1. Offload synchronous Redis checks to a threadpool to prevent blocking uvicorn
            result = await asyncio.to_thread(huey.result, task_id, blocking=False)

            if isinstance(result, Exception):
                yield {"event": "message", "data": json.dumps({"status": "failed", "error": str(result)})}
                break
            elif result is not None:
                yield {
                    "event": "message",
                    "data": json.dumps({"status": "complete", "data": result}, default=numpy_safe_encoder)
                }
                break

            # 2. Offload the progress peek
            progress_bytes = await asyncio.to_thread(huey.storage.peek_data, f"prog_{job_id}")

            # Safely check if the data is bytes 
            if isinstance(progress_bytes, bytes):
                current_pct = int(progress_bytes.decode('utf-8'))
            else:
                current_pct = 0

            yield {"event": "message", "data": json.dumps({"status": "processing", "progress": current_pct})}

            # Yield control back to the event loop
            await asyncio.sleep(0.4)

    return EventSourceResponse(
        event_generator(),
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

@app.get("/api/v1/tasks/{task_id}")
async def get_task_status(task_id: str):
    """Poll this endpoint to check if the background math is done."""
    result = await asyncio.to_thread(huey.result, task_id, blocking=False)
    if result is None:
        return {"status": "processing"}
    elif isinstance(result, Exception):
        return {"status": "failed", "error": str(result)}
    return {"status": "complete", "data": result}