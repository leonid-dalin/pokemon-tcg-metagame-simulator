from __future__ import annotations

import re
from typing import Mapping


def map_archetype(deck_id: str, mapping: Mapping[str, str] | None = None) -> str | None:
    if not deck_id:
        return None
    if mapping and deck_id in mapping:
        return mapping[deck_id]
    cleaned = re.sub(r"[-_]+", " ", deck_id).strip()
    return cleaned or None
