"""Pure view helpers for the BDIF parts of the Streamlit app."""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

MICRO_VIEW = "Player (Micro)"


def tier_help_text(thresholds: Mapping[str, float]) -> str:
    return ", ".join(f"{tier} (≥{threshold:.1%})" for tier, threshold in thresholds.items())


def split_by_evidence(decks: Sequence[str], insufficient: Sequence[str]) -> tuple[list[str], list[str]]:
    thin = set(insufficient)
    return [deck for deck in decks if deck not in thin], [deck for deck in decks if deck in thin]


def _percent(value: Any) -> float:
    return round(float(value) * 100, 2)


def interval_columns(
    mc_metrics: Mapping[str, Any],
    posterior: Mapping[str, Any],
    odds_view: str,
    top_cut: int,
    d2_rounds: int,
) -> dict[str, float]:
    if posterior.get("interval_status") != "ok":
        if odds_view == MICRO_VIEW and top_cut > 0 and "win_probability_mc_se" in mc_metrics:
            return {"Win Event % MC SE (approx.)": _percent(mc_metrics["win_probability_mc_se"])}
        return {}
    columns: dict[str, float] = {}
    wanted = []
    if odds_view == MICRO_VIEW:
        if top_cut > 0:
            wanted.append(("Win Event %", "win_probability"))
    else:
        if d2_rounds > 0:
            wanted.append(("Share % (Day 2)", "day2_share"))
        if top_cut > 0:
            wanted.append((f"Share % (Top {top_cut})", "top_cut_share"))
    for label, metric in wanted:
        if f"{metric}_lower" in mc_metrics:
            columns[f"{label} low"] = _percent(mc_metrics[f"{metric}_lower"])
            columns[f"{label} high"] = _percent(mc_metrics[f"{metric}_upper"])
    return columns


def panel_rows(matchup_panel: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for panel_deck, deck_rows in matchup_panel.get("rows", {}).items():
        for row in deck_rows:
            rows.append({
                "Deck": panel_deck,
                "Opponent": row["opponent"],
                "Posterior mean %": _percent(row["mean"]),
                "95% lower %": _percent(row["lower"]),
                "95% upper %": _percent(row["upper"]),
                "Matches": int(row["match_count"]),
                "Reliable": "Yes" if row["reliable"] else "Thin sample",
                "Mirror": "Yes" if row["mirror"] else "",
            })
    return rows


def best60_view(recommendation: Mapping[str, Any]) -> dict[str, Any]:
    cards = recommendation.get("cards", [])
    evidence = recommendation.get("card_evidence", {})
    return {
        "status": recommendation.get("status", "complete"),
        "total_copies": int(recommendation.get("total_copies", sum(int(c["copies"]) for c in cards))),
        "cards": [{"Card": c["card"], "Copies": int(c["copies"])} for c in cards],
        "evidence": [
            {
                "Card": card,
                "Inclusion %": round(float(e["inclusion_rate"]) * 100, 1),
                "Field inclusion %": round(float(e["field_inclusion_rate"]) * 100, 1),
                "Coefficient": round(float(e["coefficient"]), 3),
                "95% low": round(float(e["interval"][0]), 3),
                "95% high": round(float(e["interval"][1]), 3),
                "q-value": None if e.get("q_value") is None else round(float(e["q_value"]), 3),
                "Verdict": e.get("bucket", "not scored"),
            }
            for card, e in sorted(evidence.items(), key=lambda item: -abs(float(item[1]["contribution"])))
        ],
        "no_signal": [row["card"] for row in recommendation.get("no_signal", [])],
    }


def h1_rows(h1_report: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for label, key in (("Without variant control", "without_variant"), ("With variant control", "with_variant")):
        result = h1_report.get(key)
        if not result:
            continue
        low, high = result["interval"]
        rows.append({
            "Model": label,
            "Beta": round(float(result["beta"]), 4),
            "95% low": round(float(low), 4),
            "95% high": round(float(high), 4),
            "Odds ratio": round(math.exp(float(result["beta"])), 3),
        })
    return rows


def field_posterior_rows(field_posterior: Mapping[str, Mapping[str, float]], insufficient: Sequence[str]) -> list[dict[str, Any]]:
    thin = set(insufficient)
    rows = [
        {
            "Deck": deck,
            "Expected win rate %": _percent(metrics["expected_win_rate"]),
            "95% low %": _percent(metrics["expected_win_rate_lower"]),
            "95% high %": _percent(metrics["expected_win_rate_upper"]),
            "P(best pick) %": round(float(metrics["best_pick_probability"]) * 100, 1),
            "Evidence": "Thin" if deck in thin else "OK",
        }
        for deck, metrics in field_posterior.items()
    ]
    return sorted(rows, key=lambda row: -row["P(best pick) %"])


def provenance_rows(provenance: Mapping[str, Any]) -> list[dict[str, str]]:
    rows = []
    for key, value in provenance.items():
        if isinstance(value, Mapping):
            value = ", ".join(f"{nested_key}: {nested_value}" for nested_key, nested_value in value.items())
        rows.append({"Field": key.replace("_", " "), "Value": str(value)})
    return rows