import json

import pytest

from src.ingestion.best60_research.protocol import ProtocolMismatch, lock_protocol, require_protocol


def test_protocol_lock_is_canonical_immutable_and_idempotent(tmp_path):
    path = tmp_path / "protocol-lock.json"
    original = {"schema_version": 1, "seed": 1312, "gain": 0.0001}
    digest = lock_protocol(path, original)
    assert len(digest) == 64
    assert lock_protocol(path, dict(reversed(list(original.items())))) == digest
    require_protocol(path, original)
    with pytest.raises(ProtocolMismatch):
        require_protocol(path, {**original, "gain": 0.0})
    with pytest.raises(ProtocolMismatch):
        lock_protocol(path, {**original, "gain": 0.0})
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["protocol_hash"] == digest
