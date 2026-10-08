import hashlib
import json
import sqlite3
from pathlib import Path


DATABASE = Path("data/limitless.db")
MANIFEST = Path("data/snapshot/manifest.json")


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    digest = hashlib.sha256(DATABASE.read_bytes()).hexdigest()
    expected = manifest["database"]["sha256"]
    if digest != expected:
        raise ValueError(f"Snapshot SHA-256 mismatch: expected {expected}, found {digest}")

    uri = f"{DATABASE.resolve().as_uri()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        counts = {
            table: connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
            for table in ("tournaments", "standings", "pairings")
        }
        decklists = connection.execute(
            "SELECT COUNT(*) FROM standings WHERE decklist_json IS NOT NULL AND decklist_json != 'null'"
        ).fetchone()[0]

    print(json.dumps({"database_sha256": digest, "tables": counts, "decklists": decklists}, indent=2))


if __name__ == "__main__":
    main()
