from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

MAP_PATH = Path(__file__).resolve().parents[2] / "data" / "input" / "limitless_archetype_map.json"
DECK_LINK = re.compile(r'<a href="/[^/]+/decks/([^"]+)">([^<]+)</a>')


@dataclass(frozen=True)
class MappingCoverage:
    mapped: list[str]
    unmapped: list[str]


def load_archetype_map(path: Path = MAP_PATH) -> dict[str, str | None]:
    return {str(key): value for key, value in json.loads(path.read_text(encoding="utf-8")).items()}


def extract_recorded_deck_ids(path: Path) -> set[str]:
    text = path.read_text(encoding="utf-8")
    return {match.group(1) for match in DECK_LINK.finditer(text)}


def coverage_report(observed: Iterable[str], mapping: Mapping[str, str | None] | None = None) -> MappingCoverage:
    known = mapping if mapping is not None else load_archetype_map()
    observed_ids = set(observed)
    mapped = sorted(identifier for identifier in observed_ids if known.get(identifier) is not None)
    unmapped = sorted(identifier for identifier in observed_ids if known.get(identifier) is None)
    return MappingCoverage(mapped=mapped, unmapped=unmapped)


def resolve_archetype(deck_id: str, mapping: Mapping[str, str | None] | None = None) -> str | None:
    known = mapping if mapping is not None else load_archetype_map()
    return known.get(deck_id)
