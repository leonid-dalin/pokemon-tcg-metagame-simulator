from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

MAP_PATH = Path(__file__).resolve().parents[2] / "data" / "input" / "limitless_archetype_map.json"


def load_archetype_map(path: Path = MAP_PATH) -> dict[str, str | None]:
    return {str(key): value for key, value in json.loads(path.read_text(encoding="utf-8")).items()}


def resolve_archetype(deck_id: str, mapping: Mapping[str, str | None] | None = None) -> str | None:
    known = mapping if mapping is not None else load_archetype_map()
    return known.get(deck_id)
