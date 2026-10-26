"""Immutable protocol serialisation and run-lock checks."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


class ProtocolMismatch(ValueError):
    """Raised when a run attempts to resume under different protocol values."""


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def protocol_hash(protocol: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(protocol)).hexdigest()


def lock_protocol(path: str | Path, protocol: dict[str, Any]) -> str:
    """Write a protocol and hash once; matching repeated locks are idempotent."""
    target = Path(path)
    digest = protocol_hash(protocol)
    if target.exists():
        stored = json.loads(target.read_text(encoding="utf-8"))
        if stored.get("protocol_hash") != digest or stored.get("protocol") != protocol:
            raise ProtocolMismatch("a protocol lock already exists with different contents")
        return digest
    target.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(target, {"protocol": protocol, "protocol_hash": digest})
    return digest


def require_protocol(path: str | Path, protocol: dict[str, Any]) -> None:
    """Raise ProtocolMismatch when a saved run lock differs from the supplied protocol."""
    target = Path(path)
    if not target.is_file():
        raise ProtocolMismatch("protocol lock does not exist")
    stored = json.loads(target.read_text(encoding="utf-8"))
    expected = protocol_hash(protocol)
    if stored.get("protocol_hash") != expected or stored.get("protocol") != protocol:
        raise ProtocolMismatch("protocol contents do not match the saved run lock")


def _atomic_json(path: Path, value: Any) -> None:
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
