import hashlib
import os
from pathlib import Path

import pytest

os.environ.setdefault("OTEL_SDK_DISABLED", "true")
# src.worker.queue connects to Redis on import; Windows takes 4 s to refuse.
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/?db=0&socket_connect_timeout=0.05")


SHIPPED_DATA = (Path(__file__).resolve().parent.parent / "data" / "input").glob("limitless_model_fit*")


def _fingerprints():
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in SHIPPED}


SHIPPED = sorted(SHIPPED_DATA)


@pytest.fixture(scope="session", autouse=True)
def shipped_model_cache_is_untouched():
    """Fail the run if any test writes into the model cache committed with the shipped snapshot."""
    before = _fingerprints()
    yield
    changed = sorted(name for name, digest in _fingerprints().items() if before.get(name) != digest)
    assert not changed, f"tests modified committed files in data/input: {changed}; pass model_cache_path under tmp_path"
