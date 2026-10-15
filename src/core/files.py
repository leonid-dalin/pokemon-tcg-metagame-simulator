import json
import os
from typing import Callable, TextIO


def write_json_atomic(payload: dict, path: str) -> None:
    _write_atomic(path, lambda handle: json.dump(payload, handle, indent=2))


def write_text_atomic(text: str, path: str) -> None:
    _write_atomic(path, lambda handle: handle.write(text))


def relative_path(path: str) -> str:
    """The path as a report should record it, so a shared report carries no local prefix."""
    try:
        return os.path.relpath(path)
    except ValueError:
        return str(path)


def _write_atomic(path: str, write: Callable[[TextIO], object]) -> None:
    temp_path = f"{path}.tmp"
    target_mode = os.stat(path).st_mode & 0o777 if os.path.exists(path) else None
    try:
        with open(temp_path, "w", encoding="utf-8") as handle:
            write(handle)
            handle.flush()
            os.fsync(handle.fileno())
        if target_mode is not None:
            os.chmod(temp_path, target_mode)
        os.replace(temp_path, path)
    except Exception:
        try:
            os.remove(temp_path)
        except FileNotFoundError:
            pass
        raise
