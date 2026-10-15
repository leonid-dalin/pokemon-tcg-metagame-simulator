"""Self-contained HTML rendering of a BDIF report, laid out for a phone."""
from __future__ import annotations

import html
from typing import Any, Mapping, Sequence

from src.ui import bdif_view

STYLE = """
:root {
  color-scheme: light dark;
  --bg: #ffffff; --fg: #1b1b1b; --muted: #666; --line: #dcdcdc; --card: #f6f6f6; --flag: #8a4b00;
}
@media (prefers-color-scheme: dark) {
  :root { --bg: #121212; --fg: #e9e9e9; --muted: #9b9b9b; --line: #333; --card: #1d1d1d; --flag: #e0a458; }
}
* { box-sizing: border-box; }
body {
  margin: 0; padding: 16px; background: var(--bg); color: var(--fg);
  font: 16px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  -webkit-text-size-adjust: 100%;
}
main { max-width: 52rem; margin: 0 auto; }
h1 { font-size: 1.4rem; margin: 0 0 4px; }
h2 { font-size: 1.15rem; margin: 28px 0 8px; padding-bottom: 4px; border-bottom: 1px solid var(--line); }
h3 { font-size: 1rem; margin: 20px 0 6px; }
h4 { font-size: .95rem; margin: 14px 0 4px; color: var(--muted); }
p { margin: 0 0 10px; }
p.meta { color: var(--muted); font-size: .9rem; }
p.note { color: var(--flag); font-size: .9rem; }
.scroll { overflow-x: auto; -webkit-overflow-scrolling: touch; margin: 0 0 12px; }
table { border-collapse: collapse; width: 100%; font-size: .9rem; }
th, td { text-align: left; padding: 6px 10px; border-bottom: 1px solid var(--line); white-space: nowrap; }
th { color: var(--muted); font-weight: 600; }
tbody tr:nth-child(even) { background: var(--card); }
dl { display: grid; grid-template-columns: minmax(0, auto) minmax(0, 1fr); gap: 4px 12px; margin: 0 0 12px; font-size: .9rem; }
dt { color: var(--muted); }
dd { margin: 0; overflow-wrap: anywhere; }
ul { padding-left: 20px; margin: 6px 0 12px; }
footer { color: var(--muted); font-size: .85rem; margin-top: 28px; border-top: 1px solid var(--line); padding-top: 8px; }
"""


def _escaped(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def _percent(value: Any) -> float:
    return round(float(value) * 100, 2)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> list[Any]:
    return list(value) if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) else []


def _cell(value: Any) -> str:
    return f"<td>{_escaped(value)}</td>"


def _table(rows: Sequence[Mapping[str, Any]], columns: Sequence[str] | None = None) -> str:
    rows = [row for row in rows if row]
    if not rows:
        return ""
    names = list(columns) if columns else list(rows[0])
    names = [name for name in names if any(row.get(name, "") not in (None, "") for row in rows)] or names
    head = "".join(f"<th>{_escaped(name)}</th>" for name in names)
    body = "".join("<tr>" + "".join(_cell(row.get(name, "")) for name in names) + "</tr>" for row in rows)
    return f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def _details(pairs: Sequence[tuple[str, Any]]) -> str:
    items = "".join(f"<dt>{_escaped(label)}</dt><dd>{_escaped(value)}</dd>" for label, value in pairs if value not in (None, "", []))
    return f"<dl>{items}</dl>" if items else ""


def _section(title: str, body: str, level: int = 2) -> str:
    return f"<h{level}>{_escaped(title)}</h{level}>{body}" if body else ""


def _swap_block(label: str, rows: Sequence[Mapping[str, Any]]) -> str:
    return f"<h4>{_escaped(label)}</h4>{_table(rows)}" if rows else ""


def _significance(by_deck: Mapping[str, Sequence[float]]) -> tuple[list[dict[str, Any]], int]:
    clear, unclear = [], 0
    for deck, values in sorted(by_deck.items()):
        beta, low, high = (float(value) for value in values[:3])
        if low > 0 or high < 0:
            clear.append({"Opponent": deck, "Effect": round(beta, 3), "95% low": round(low, 3), "95% high": round(high, 3)})
        else:
            unclear += 1
    return clear, unclear


def _matchups(recommendation: Mapping[str, Any]) -> str:
    blocks = []
    for slot, by_deck in recommendation.get("card_matchups", {}).items():
        clear, unclear = _significance(by_deck)
        note = ""
        if unclear:
            noun = "opponent" if unclear == 1 else "opponents"
            note = f'<p class="note">No separation from zero against {unclear} other {noun}.</p>'
        if clear:
            blocks.append(f"<h4>{_escaped(slot)}</h4>{_table(clear)}{note}")
    if not blocks:
        return ""
    return _section("Per-opponent card effects", "".join(blocks), level=3)


def _card_columns(cards: Sequence[Mapping[str, Any]]) -> list[str]:
    columns = ["Card", "Copies"]
    if any(row["Consensus"] != row["Copies"] for row in cards):
        columns.append("Consensus")
    if any(row["Source"] != "consensus" for row in cards):
        columns.append("Source")
    return columns + ["Play rate %", "Top 25% rate %", "Same count in resamples %"]


def _best60(name: str, recommendation: Mapping[str, Any]) -> str:
    view = bdif_view.best60_view(recommendation)
    parts = [
        _section(name, "", level=3),
        _details((
            ("Status", view["status"]),
            ("Deck ids", ", ".join(view["deck_ids"])),
            ("Legal lists", view["legal_list_count"]),
            ("List mode", view["list_mode"]),
            ("Total copies", view["total_copies"]),
            ("ACE SPEC", view["ace_spec_choice"]),
            ("Joint probability %", view["joint_probability"]),
        )),
        _table(view["cards"], _card_columns(view["cards"])),
        _swap_block("Applied swaps", view["swaps"]),
        _swap_block("Proposed swaps held back by the held-out gate", view["proposed_swaps"]),
        _swap_block("Leaning swaps", view["leaning_swaps"]),
        "<h4>Win rate</h4>" + _details(list(view["win_rates"].items())),
        "<h4>Model</h4>" + _details(list(view["model_settings"].items())),
        _matchups(recommendation),
        _table(view["support"]),
        _table(view["trends"]),
    ]
    if view["breakthrough"]:
        parts.append("<h4>Breakthrough cards</h4><ul>" + "".join(f"<li>{_escaped(card)}</li>" for card in view["breakthrough"]) + "</ul>")
    if view["field"]:
        parts.append("<h4>Field</h4>" + _table(view["field"]))
    return "".join(part for part in parts if part)


def _ranked_rows(sections: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for deck, metrics in _mapping(sections.get("ranked_metrics")).items():
        metrics = _mapping(metrics)
        rows.append({
            "Deck": deck,
            "Win event %": _percent(metrics.get("win_probability", 0.0)),
            "Win event low %": _percent(metrics.get("win_probability_lower", 0.0)),
            "Win event high %": _percent(metrics.get("win_probability_upper", 0.0)),
            "Top cut share %": _percent(metrics.get("top_cut_share", 0.0)),
            "Day 2 share %": _percent(metrics.get("day2_share", 0.0)),
            "Top cut conversion %": _percent(metrics.get("top_cut_conversion", 0.0)),
        })
    return sorted(rows, key=lambda row: -row["Win event %"])


def _title(report: Mapping[str, Any]) -> str:
    names = list(_mapping(report.get("best60_recommendations")))
    return f"{names[0]} BDIF report" if len(names) == 1 else "BDIF report"


def render_report_html(report: Any, title: str | None = None) -> str:
    """Render a BDIF report as one HTML document with no external assets.

    Accepts either the CLI result, which carries its sections under `mc_results`, or a report
    that already holds them at the top level. A section the report does not carry is left out,
    and a result that is not a mapping renders as an empty report so a caller's exit code stands.
    """
    report = _mapping(report)
    inner = report.get("mc_results")
    sections: Mapping[str, Any] = inner if isinstance(inner, Mapping) else report
    recommendations = {
        name: recommendation
        for name, recommendation in _mapping(sections.get("best60_recommendations")).items()
        if isinstance(recommendation, Mapping)
    }
    heading = title or _title(report)

    body = [
        _section(
            "Best-60 card selection",
            "".join(_best60(name, recommendation) for name, recommendation in recommendations.items()),
        ),
        _section(
            "Field",
            _table(bdif_view.field_posterior_rows(
                _mapping(sections.get("field_posterior")), _sequence(sections.get("insufficient_data")),
            )),
        ),
        _section("Ranked decks", _table(_ranked_rows(sections))),
        _section("Matchup panel", _table(bdif_view.panel_rows(_mapping(sections.get("matchup_panel"))))),
        _section("H1 check", _table(bdif_view.h1_rows(_mapping(sections.get("h1_report"))))),
        _section("Provenance", _table(bdif_view.provenance_rows(_mapping(sections.get("provenance"))))),
    ]
    status = report.get("status") or sections.get("status") or "unknown"
    document = "".join((
        "<!DOCTYPE html>",
        '<html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>{_escaped(heading)}</title>",
        f"<style>{STYLE}</style></head><body><main>",
        f"<h1>{_escaped(heading)}</h1>",
        f'<p class="meta">Status: {_escaped(status)}</p>',
        "".join(body),
        "<footer>Generated from observed tournament results.</footer>",
        "</main></body></html>",
    ))
    return document
