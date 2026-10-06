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
    noisy = []
    for label, metric in wanted:
        if f"{metric}_lower" in mc_metrics:
            columns[f"{label} low"] = _percent(mc_metrics[f"{metric}_lower"])
            columns[f"{label} high"] = _percent(mc_metrics[f"{metric}_upper"])
            if mc_metrics.get(f"{metric}_interval") == "raw":
                noisy.append(f"{label}: includes simulation noise")
    if columns:
        columns["Interval note"] = "; ".join(noisy)
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


def _slots_text(slots: Sequence[Sequence[Any]]) -> str:
    counts: dict[str, list[int]] = {}
    for card, copy in slots:
        counts.setdefault(str(card), []).append(int(copy))
    return ", ".join(
        f"{card} (copy {copies[0]})" if len(copies) == 1 else f"{card} (copies {min(copies)}-{max(copies)})"
        for card, copies in counts.items()
    )


def _swap_rows(swaps: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "Out": _slots_text(swap["removed"]) if "removed" in swap else f"{swap['remove']} (copy {swap['remove_copy']})",
            "In": _slots_text(swap["added"]) if "added" in swap else f"{swap['add']} (copy {swap['add_copy']})",
            "Gain (log-odds)": round(float(swap["gain"]), 3),
            "Chance it helps %": _percent(swap["probability"]),
        }
        for swap in swaps
    ]


def best60_view(recommendation: Mapping[str, Any]) -> dict[str, Any]:
    cards = recommendation.get("cards", [])
    stats = recommendation.get("card_stats", {})
    rates = recommendation.get("match_win_rate", {})
    model = recommendation.get("model", {})
    trend = recommendation.get("trends", {})

    def tier(card: str, name: str) -> float | None:
        value = stats.get(card, {}).get(name, {}).get("with_card")
        return None if value is None else _percent(value)

    return {
        "status": recommendation.get("status", "complete"),
        "deck_ids": list(recommendation.get("deck_ids", [])),
        "legal_list_count": int(recommendation.get("legal_list_count", 0)),
        "ace_spec_choice": recommendation.get("ace_spec_choice"),
        "total_copies": int(recommendation.get("total_copies", sum(int(c["copies"]) for c in cards))),
        "cards": [
            {
                "Card": c["card"],
                "Copies": int(c["copies"]),
                "Consensus": int(c.get("consensus_copies", c["copies"])),
                "Source": c.get("source", "consensus"),
                "Play rate %": None if c["card"] not in stats else _percent(stats[c["card"]]["play_rate"]),
                "Top 25% rate %": tier(c["card"], "top25"),
            }
            for c in [*cards, *recommendation.get("removed_cards", [])]
        ],
        "swaps": _swap_rows(recommendation.get("swaps", [])),
        "proposed_swaps": _swap_rows(recommendation.get("proposed_swaps", [])),
        "leaning_swaps": _swap_rows(recommendation.get("leaning_swaps", [])),
        "win_rates": {
            "Archetype average %": None if "archetype_average" not in rates else _percent(rates["archetype_average"]),
            "With swaps, held out %": None if "with_swaps_held_out" not in rates else _percent(rates["with_swaps_held_out"]),
            "With swaps, in sample %": None if "with_swaps_in_sample" not in rates else _percent(rates["with_swaps_in_sample"]),
        },
        "prior_sd": model.get("prior_sd"),
        "list_mode": recommendation.get("list_mode", "novel"),
        "joint_probability": None if recommendation.get("joint_probability") is None else _percent(recommendation["joint_probability"]),
        "support": [
            {"List": name, **{f"Played lists within {changes} changes": count for changes, count in sorted(profile.items(), key=lambda item: int(item[0]))}}
            for name, profile in (("Consensus", recommendation.get("support", {}).get("consensus")), ("Recommended", recommendation.get("support", {}).get("recommended")))
            if profile
        ],
        "breakthrough": list(trend.get("breakthrough", [])),
        "trends": [
            {
                "Card": row["card"],
                "Start %": _percent(row["start_share"]),
                "End %": _percent(row["end_share"]),
                "Change (points)": _percent(row["change"]),
                "Copies change": round(float(row["copies_change"]), 2),
                "Model effect": None if row.get("model_effect") is None else round(float(row["model_effect"]), 3),
            }
            for row in [*trend.get("rising", []), *trend.get("falling", [])]
        ],
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