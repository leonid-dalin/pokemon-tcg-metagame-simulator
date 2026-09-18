from typing import Any, Dict, Sequence

RAW_PLAYERS = "Raw Players"

Row = Dict[str, Any]


def _scalar(val: Any) -> float:
    if isinstance(val, (list, tuple)):
        val = val[0] if val else 0.0
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0


def _divisor(players: int, input_mode: str) -> float:
    return float(players) if input_mode == RAW_PLAYERS else 100.0


def locked_share(rows: Sequence[Row], players: int, input_mode: str) -> float:
    divisor = _divisor(players, input_mode)
    if divisor <= 0:
        return 0.0
    total = sum(_scalar(row.get("val")) for row in rows if row.get("spec_type") == "Exact")
    return total / divisor


def locked_exact_spec(rows: Sequence[Row], players: int, input_mode: str) -> Dict[str, float]:
    divisor = _divisor(players, input_mode)
    if divisor <= 0:
        return {}
    spec: Dict[str, float] = {}
    for row in rows:
        if row.get("spec_type") != "Exact":
            continue
        deck = str(row.get("deck", "")).strip()
        if deck:
            spec[deck] = _scalar(row.get("val")) / divisor
    return spec
