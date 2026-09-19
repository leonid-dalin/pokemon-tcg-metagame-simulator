import json

import pytest

from src.api import main


@pytest.mark.anyio
async def test_task_status_reports_huey_exceptions_as_failed(monkeypatch):
    async def failed_to_thread(*args, **kwargs):
        return RuntimeError("worker failed")

    monkeypatch.setattr(main.asyncio, "to_thread", failed_to_thread)

    request = type("RequestStub", (), {"scope": {"type": "http"}})()
    response = await main.get_task_status.__wrapped__(request, "task-id")

    assert json.loads(response.body) == {
        "task_id": "task-id",
        "status": "failed",
    }