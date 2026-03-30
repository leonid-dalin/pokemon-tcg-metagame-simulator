from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.models import PredictionRequest
from src.worker.queue import execute_simulation_job, huey

app = FastAPI(title="TCG Simulator API")

# Allow Streamlit to talk to this API
app.add_middleware(
    CORSMiddleware, # type: ignore
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/v1/predict")
async def start_prediction(request: PredictionRequest):
    """Enqueue the simulation and return a Task ID."""
    # Convert Pydantic model to dict and enqueue
    task = execute_simulation_job(request.model_dump())

    return {
        "status": "processing",
        "task_id": task.id
    }


@app.get("/api/v1/tasks/{task_id}")
async def get_task_status(task_id: str):
    """Poll this endpoint to check if the background math is done."""
    result = huey.result(task_id, blocking=False)

    if result is None:
        return {"status": "processing"}
    elif isinstance(result, Exception):
        return {"status": "failed", "error": str(result)}

    return {"status": "complete", "data": result}