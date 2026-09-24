import json
import os


def write_json_atomic(payload: dict, path: str) -> None:
    temp_path = f"{path}.tmp"
    target_mode = os.stat(path).st_mode & 0o777 if os.path.exists(path) else None
    try:
        with open(temp_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
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
