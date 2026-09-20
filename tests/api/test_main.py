import asyncio
import json
from types import SimpleNamespace

import pytest

from src.api import main


PROTECTED_ENDPOINTS = (
    ("POST", "/api/v1/predict"),
    ("GET", "/api/v1/tasks/task-id"),
    ("GET", "/api/v1/tasks/task-id/stream"),
)


def request_for(method, path, token=None):
    headers = [(b"x-api-token", token.encode())] if token is not None else []
    return main.Request(
        {
            "type": "http",
            "method": method,
            "path": path,
            "headers": headers,
        }
    )


def stream_request(redis=None, token=None):
    if redis is None:
        redis = FakeRedis()

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    return main.Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/api/v1/tasks/task-id/stream",
            "headers": [(b"x-api-token", token.encode())] if token is not None else [],
            "query_string": b"",
            "app": SimpleNamespace(state=SimpleNamespace(redis=redis)),
        },
        receive=receive,
    )


class FakeClock:
    def __init__(self, values):
        self.values = iter(values)
        self.last = values[0]

    def __call__(self):
        try:
            self.last = next(self.values)
        except StopIteration:
            pass
        return self.last


class FakePubSub:
    def __init__(self, messages=None):
        self.messages = list(messages or [])
        self.closed = False
        self.subscribed = []

    async def subscribe(self, channel):
        self.subscribed.append(channel)

    async def get_message(self, **kwargs):
        if self.messages:
            return self.messages.pop(0)
        raise asyncio.TimeoutError()

    async def aclose(self):
        self.closed = True


class FakeRedis:
    def __init__(self, initial_state=None):
        self.initial_state = initial_state
        self.pubsub_instance = FakePubSub()

    async def pubsub(self):
        return self.pubsub_instance

    async def get(self, key):
        return self.initial_state


async def consume_events(response):
    events = []
    async for event in response.body_iterator:
        events.append(event)
    await response.body_iterator.aclose()
    return events


@pytest.mark.anyio
@pytest.mark.parametrize("method, path", PROTECTED_ENDPOINTS)
async def test_protected_endpoints_are_open_when_api_token_is_unset(method, path, monkeypatch):
    seen = []

    async def call_next(_request):
        seen.append((method, path))
        return main.JSONResponse(status_code=200, content={"protected": True})

    monkeypatch.delenv("API_TOKEN", raising=False)
    response = await main.ApiTokenAuthMiddleware(SimpleNamespace()).dispatch(
        request_for(method, path),
        call_next,
    )

    assert response.status_code == 200
    assert seen == [(method, path)]


@pytest.mark.anyio
@pytest.mark.parametrize("method, path", PROTECTED_ENDPOINTS)
async def test_protected_endpoints_accept_the_matching_api_token(method, path, monkeypatch):
    seen = []

    async def call_next(_request):
        seen.append((method, path))
        return main.JSONResponse(status_code=200, content={"protected": True})

    monkeypatch.setenv("API_TOKEN", "correct-token")
    response = await main.ApiTokenAuthMiddleware(SimpleNamespace()).dispatch(
        request_for(method, path, "correct-token"),
        call_next,
    )

    assert response.status_code == 200
    assert seen == [(method, path)]


@pytest.mark.anyio
@pytest.mark.parametrize("method, path", PROTECTED_ENDPOINTS)
@pytest.mark.parametrize("token", [None, "wrong-token"])
async def test_protected_endpoints_reject_missing_and_mismatched_tokens(method, path, token, monkeypatch):
    async def call_next(_request):
        pytest.fail("protected endpoint should not be called")

    monkeypatch.setenv("API_TOKEN", "correct-token")
    response = await main.ApiTokenAuthMiddleware(SimpleNamespace()).dispatch(
        request_for(method, path, token),
        call_next,
    )

    assert response.status_code == 401
    assert json.loads(response.body) == {"detail": "Unauthorized"}


@pytest.mark.anyio
async def test_unrelated_routes_do_not_require_an_api_token(monkeypatch):
    seen = []

    async def call_next(_request):
        seen.append("/health")
        return main.JSONResponse(status_code=200, content={"status": "ok"})

    monkeypatch.setenv("API_TOKEN", "correct-token")
    response = await main.ApiTokenAuthMiddleware(SimpleNamespace()).dispatch(
        request_for("GET", "/health"),
        call_next,
    )

    assert response.status_code == 200
    assert seen == ["/health"]


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


@pytest.mark.parametrize(
    "method,path,configured_token,supplied_token,expected_status",
    [
        *((method, path, None, None, 200) for method, path in PROTECTED_ENDPOINTS),
        *((method, path, "", None, 200) for method, path in PROTECTED_ENDPOINTS),
        *((method, path, "correct-token", "correct-token", 200) for method, path in PROTECTED_ENDPOINTS),
        *((method, path, "correct-token", None, 401) for method, path in PROTECTED_ENDPOINTS),
        *((method, path, "correct-token", "", 401) for method, path in PROTECTED_ENDPOINTS),
        *((method, path, "correct-token", "wrong-token", 401) for method, path in PROTECTED_ENDPOINTS),
    ],
)
@pytest.mark.anyio
async def test_api_token_policy(method, path, configured_token, supplied_token, expected_status, monkeypatch):
    seen = []

    async def call_next(_request):
        seen.append((method, path))
        return main.JSONResponse(status_code=200, content={"protected": True})

    if configured_token is None:
        monkeypatch.delenv("API_TOKEN", raising=False)
    else:
        monkeypatch.setenv("API_TOKEN", configured_token)

    response = await main.ApiTokenAuthMiddleware(SimpleNamespace()).dispatch(
        request_for(method, path, supplied_token),
        call_next,
    )

    assert response.status_code == expected_status
    if expected_status == 401:
        assert seen == []
        assert json.loads(response.body) == {"detail": "Unauthorized"}
    else:
        assert seen == [(method, path)]


@pytest.mark.anyio
async def test_unrelated_routes_do_not_require_an_api_token(monkeypatch):
    seen = []

    async def call_next(_request):
        seen.append("/health")
        return main.JSONResponse(status_code=200, content={"status": "ok"})

    monkeypatch.setenv("API_TOKEN", "correct-token")
    response = await main.ApiTokenAuthMiddleware(SimpleNamespace()).dispatch(
        request_for("GET", "/health"),
        call_next,
    )

    assert response.status_code == 200
    assert seen == ["/health"]


@pytest.mark.parametrize("token,expected_error", [(None, "Task failed"), ("correct-token", "worker failed")])
@pytest.mark.anyio
async def test_task_status_redacts_worker_errors(token, expected_error, monkeypatch):
    async def failed_to_thread(*args, **kwargs):
        return RuntimeError("worker failed")

    logged = []
    monkeypatch.setenv("API_TOKEN", "correct-token")
    monkeypatch.setattr(main.asyncio, "to_thread", failed_to_thread)
    monkeypatch.setattr(
        main.logger,
        "error",
        lambda *args, **kwargs: logged.append((args, kwargs)),
    )

    response = await main.get_task_status.__wrapped__(
        request_for("GET", "/api/v1/tasks/task-id", token),
        "task-id",
    )

    assert json.loads(response.body) == {
        "task_id": "task-id",
        "status": "failed",
        "error": expected_error,
    }
    assert logged[0][0][0] == "task_exception"
    assert logged[0][1]["error"] == "worker failed"


@pytest.mark.anyio
async def test_sse_terminates_at_the_injected_deadline_without_result(monkeypatch):
    redis = FakeRedis()
    monkeypatch.setenv("API_TOKEN", "correct-token")
    monkeypatch.setattr(main, "sse_clock", FakeClock([0.0, 600.0]))
    monkeypatch.setattr(main.huey.storage, "peek_data", lambda key: None)

    response = await main.stream_task_progress.__wrapped__(stream_request(redis), "task-id")
    events = await consume_events(response)

    assert events == []
    assert redis.pubsub_instance.closed


@pytest.mark.anyio
async def test_sse_reports_completion_at_the_injected_deadline(monkeypatch):
    redis = FakeRedis()
    monkeypatch.setenv("API_TOKEN", "correct-token")
    monkeypatch.setattr(main, "sse_clock", FakeClock([0.0, 600.0]))
    monkeypatch.setattr(main.huey.storage, "peek_data", lambda key: None)
    monkeypatch.setattr(
        main.huey,
        "result",
        lambda task_id, blocking=False: {"value": 1},
    )

    response = await main.stream_task_progress.__wrapped__(stream_request(redis), "task-id")
    events = await consume_events(response)

    assert events == [
        {
            "event": "message",
            "data": json.dumps({"status": "complete", "data": {"value": 1}}),
        }
    ]
    assert redis.pubsub_instance.closed


@pytest.mark.parametrize("token,expected_error", [(None, "Task failed"), ("correct-token", "worker failed")])
@pytest.mark.anyio
async def test_sse_redacts_worker_errors(token, expected_error, monkeypatch):
    redis = FakeRedis()
    logged = []
    monkeypatch.setenv("API_TOKEN", "correct-token")
    monkeypatch.setattr(main, "sse_clock", FakeClock([0.0, 0.0]))
    monkeypatch.setattr(main.huey.storage, "peek_data", lambda key: None)
    monkeypatch.setattr(main.huey, "result", lambda task_id, blocking=False: RuntimeError("worker failed"))
    monkeypatch.setattr(
        main.logger,
        "error",
        lambda *args, **kwargs: logged.append((args, kwargs)),
    )

    response = await main.stream_task_progress.__wrapped__(
        stream_request(redis, token),
        "task-id",
    )
    events = await consume_events(response)

    assert json.loads(events[0]["data"]) == {
        "status": "failed",
        "error": expected_error,
    }
    assert logged[0][0][0] == "task_exception"
    assert logged[0][1]["error"] == "worker failed"
    assert redis.pubsub_instance.closed


@pytest.mark.anyio
async def test_sse_stream_exception_is_valid_json(monkeypatch):
    redis = FakeRedis()
    redis.pubsub_instance.get_message = async_fail
    logged = []
    monkeypatch.setenv("API_TOKEN", "correct-token")
    monkeypatch.setattr(main, "sse_clock", FakeClock([0.0, 0.0]))
    monkeypatch.setattr(main.huey.storage, "peek_data", lambda key: None)
    monkeypatch.setattr(main.huey, "result", lambda task_id, blocking=False: None)
    monkeypatch.setattr(
        main.logger,
        "error",
        lambda *args, **kwargs: logged.append((args, kwargs)),
    )

    response = await main.stream_task_progress.__wrapped__(stream_request(redis), "task-id")
    events = await consume_events(response)

    assert json.loads(events[0]["data"]) == {
        "status": "failed",
        "error": "Stream disconnected internally",
        "data": None,
    }
    assert logged[0][0][0] == "sse_stream_exception"
    assert redis.pubsub_instance.closed


async def async_fail():
    raise RuntimeError("pubsub failed")


@pytest.mark.anyio
async def test_task_status_reports_huey_exceptions_as_failed(monkeypatch):
    async def failed_to_thread(*args, **kwargs):
        return RuntimeError("worker failed")

    monkeypatch.setenv("API_TOKEN", "correct-token")
    monkeypatch.setattr(main.asyncio, "to_thread", failed_to_thread)

    request = request_for("GET", "/api/v1/tasks/task-id", "correct-token")
    response = await main.get_task_status.__wrapped__(request, "task-id")

    assert json.loads(response.body) == {
        "task_id": "task-id",
        "status": "failed",
        "error": "worker failed",
    }


@pytest.mark.anyio
async def test_unauthenticated_task_status_reports_a_generic_failure(monkeypatch):
    async def failed_to_thread(*args, **kwargs):
        return RuntimeError("worker failed")

    logged = []
    monkeypatch.setenv("API_TOKEN", "correct-token")
    monkeypatch.setattr(main.asyncio, "to_thread", failed_to_thread)
    monkeypatch.setattr(
        main.logger,
        "error",
        lambda *args, **kwargs: logged.append((args, kwargs)),
    )

    request = request_for("GET", "/api/v1/tasks/task-id")
    response = await main.get_task_status.__wrapped__(request, "task-id")

    assert json.loads(response.body) == {
        "task_id": "task-id",
        "status": "failed",
        "error": "Task failed",
    }
    assert logged[0][0][0] == "task_exception"
    assert logged[0][1]["error"] == "worker failed"


@pytest.mark.anyio
async def test_authenticated_task_status_can_report_the_worker_error(monkeypatch):
    async def failed_to_thread(*args, **kwargs):
        return RuntimeError("worker failed")

    monkeypatch.setenv("API_TOKEN", "correct-token")
    monkeypatch.setattr(main.asyncio, "to_thread", failed_to_thread)

    request = request_for("GET", "/api/v1/tasks/task-id", "correct-token")
    response = await main.get_task_status.__wrapped__(request, "task-id")

    assert json.loads(response.body) == {
        "task_id": "task-id",
        "status": "failed",
        "error": "worker failed",
    }


@pytest.mark.anyio
async def test_sse_terminates_at_the_injected_deadline_without_result(monkeypatch):
    clock = FakeClock([0.0, 600.0])
    redis = FakeRedis()
    monkeypatch.setenv("API_TOKEN", "correct-token")
    monkeypatch.setattr(main, "sse_clock", clock)
    monkeypatch.setattr(main.huey.storage, "peek_data", lambda key: None)
    monkeypatch.setattr(
        main.asyncio,
        "to_thread",
        lambda func, *args, **kwargs: func(*args, **kwargs),
    )

    response = await main.stream_task_progress.__wrapped__(
        stream_request(redis),
        "task-id",
    )
    events = await consume_events(response)

    assert events == []
    assert redis.pubsub_instance.closed


@pytest.mark.anyio
async def test_sse_reports_completion_at_the_injected_deadline(monkeypatch):
    clock = FakeClock([0.0, 600.0])
    redis = FakeRedis()
    monkeypatch.setenv("API_TOKEN", "correct-token")
    monkeypatch.setattr(main, "sse_clock", clock)
    monkeypatch.setattr(main.huey.storage, "peek_data", lambda key: None)
    monkeypatch.setattr(
        main.asyncio,
        "to_thread",
        lambda func, *args, **kwargs: func(*args, **kwargs),
    )
    monkeypatch.setattr(
        main.huey,
        "result",
        lambda task_id, blocking=False: {"value": 1},
    )

    response = await main.stream_task_progress.__wrapped__(
        stream_request(redis),
        "task-id",
    )
    events = await consume_events(response)

    assert events == [
        {
            "event": "message",
            "data": json.dumps({"status": "complete", "data": {"value": 1}}),
        }
    ]
    assert redis.pubsub_instance.closed


@pytest.mark.anyio
async def test_sse_preserves_a_complete_task_termination(monkeypatch):
    redis = FakeRedis()
    monkeypatch.setenv("API_TOKEN", "correct-token")
    monkeypatch.setattr(main, "sse_clock", FakeClock([0.0, 0.0]))
    monkeypatch.setattr(main.huey.storage, "peek_data", lambda key: None)
    monkeypatch.setattr(
        main.asyncio,
        "to_thread",
        lambda func, *args, **kwargs: func(*args, **kwargs),
    )
    monkeypatch.setattr(
        main.huey,
        "result",
        lambda task_id, blocking=False: {"value": 1},
    )

    response = await main.stream_task_progress.__wrapped__(
        stream_request(redis),
        "task-id",
    )
    events = await consume_events(response)

    assert events == [
        {
            "event": "message",
            "data": json.dumps({"status": "complete", "data": {"value": 1}}),
        }
    ]
    assert redis.pubsub_instance.closed


@pytest.mark.anyio
async def test_unauthenticated_sse_reports_a_generic_task_failure(monkeypatch):
    redis = FakeRedis()
    logged = []
    monkeypatch.setenv("API_TOKEN", "correct-token")
    monkeypatch.setattr(main, "sse_clock", FakeClock([0.0, 0.0]))
    monkeypatch.setattr(main.huey.storage, "peek_data", lambda key: None)
    monkeypatch.setattr(
        main.asyncio,
        "to_thread",
        lambda func, *args, **kwargs: func(*args, **kwargs),
    )
    monkeypatch.setattr(main.huey, "result", lambda task_id, blocking=False: RuntimeError("worker failed"))
    monkeypatch.setattr(
        main.logger,
        "error",
        lambda *args, **kwargs: logged.append((args, kwargs)),
    )

    response = await main.stream_task_progress.__wrapped__(
        stream_request(redis),
        "task-id",
    )
    events = await consume_events(response)

    assert events == [
        {
            "event": "message",
            "data": json.dumps({"status": "failed", "error": "Task failed"}),
        }
    ]
    assert logged[0][0][0] == "task_exception"
    assert logged[0][1]["error"] == "worker failed"


@pytest.mark.anyio
async def test_authenticated_sse_can_report_the_worker_error(monkeypatch):
    redis = FakeRedis()
    monkeypatch.setenv("API_TOKEN", "correct-token")
    monkeypatch.setattr(main, "sse_clock", FakeClock([0.0, 0.0]))
    monkeypatch.setattr(main.huey.storage, "peek_data", lambda key: None)
    monkeypatch.setattr(
        main.asyncio,
        "to_thread",
        lambda func, *args, **kwargs: func(*args, **kwargs),
    )
    monkeypatch.setattr(main.huey, "result", lambda task_id, blocking=False: RuntimeError("worker failed"))

    response = await main.stream_task_progress.__wrapped__(
        stream_request(redis, "correct-token"),
        "task-id",
    )
    events = await consume_events(response)

    assert events == [
        {
            "event": "message",
            "data": json.dumps({"status": "failed", "error": "worker failed"}),
        }
    ]
    assert redis.pubsub_instance.closed
