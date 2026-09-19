import asyncio
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
        "error": "worker failed",
    }


@pytest.mark.anyio
async def test_startup_scrape_is_dispatched_without_blocking_lifespan(monkeypatch):
    scrape_called = False
    release_scrape = asyncio.Event()
    dispatched = []

    async def fake_to_thread(func, *args, **kwargs):
        if func.__name__ == "fake_scrape":
            await release_scrape.wait()
        return True

    def fake_scrape():
        nonlocal scrape_called
        scrape_called = True

    real_create_task = main.asyncio.create_task

    def record_create_task(coro):
        task = real_create_task(coro)
        dispatched.append(task)
        return task

    class FakeRedis:
        async def aclose(self):
            return None

    monkeypatch.setattr(main.asyncio, "to_thread", fake_to_thread)
    monkeypatch.setattr(main.asyncio, "create_task", record_create_task)
    monkeypatch.setattr(main.aioredis, "from_url", lambda *args, **kwargs: FakeRedis())
    monkeypatch.setattr(main, "automated_daily_pipeline", fake_scrape)

    async with main.lifespan(main.app):
        assert scrape_called is False
        assert len(dispatched) == 1
        release_scrape.set()
        await asyncio.gather(*dispatched)
