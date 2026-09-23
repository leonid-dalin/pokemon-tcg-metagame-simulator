import pytest

from src.core.config import TIER_THRESHOLDS
from src.ui import bdif_view


@pytest.mark.unit
def test_tier_help_text_is_generated_from_the_thresholds():
    assert bdif_view.tier_help_text(TIER_THRESHOLDS) == (
        "T0 (≥55.0%), T1 (≥52.5%), T2 (≥50.0%), T3 (≥47.5%), T4 (≥45.0%), T5 (≥0.0%)"
    )


@pytest.mark.unit
def test_split_by_evidence_keeps_order_and_labels_thin_decks():
    assert bdif_view.split_by_evidence(["a", "b", "c"], ["b"]) == (["a", "c"], ["b"])


@pytest.mark.unit
@pytest.mark.parametrize(("odds_view", "expected_keys"), [
    ("Player (Micro)", {"Win Event % low", "Win Event % high"}),
    ("Archetype (Macro)", {"Share % (Day 2) low", "Share % (Day 2) high", "Share % (Top 8) low", "Share % (Top 8) high"}),
])
def test_interval_columns_follow_the_view(odds_view, expected_keys):
    metrics = {
        "win_probability_lower": 0.01,
        "win_probability_upper": 0.03,
        "day2_share_lower": 0.1,
        "day2_share_upper": 0.2,
        "top_cut_share_lower": 0.05,
        "top_cut_share_upper": 0.15,
    }

    columns = bdif_view.interval_columns(metrics, {"interval_status": "ok"}, odds_view, top_cut=8, d2_rounds=2)

    assert set(columns) == expected_keys


@pytest.mark.unit
def test_interval_columns_fall_back_to_the_monte_carlo_error():
    columns = bdif_view.interval_columns(
        {"win_probability_mc_se": 0.0025},
        {"interval_status": "too few posterior draws"},
        "Player (Micro)",
        top_cut=8,
        d2_rounds=0,
    )

    assert columns == {"Win Event % MC SE (approx.)": 0.25}


@pytest.mark.unit
def test_field_posterior_rows_rank_by_best_pick_and_label_thin_decks():
    rows = bdif_view.field_posterior_rows({
        "a": {"expected_win_rate": 0.52, "expected_win_rate_lower": 0.5, "expected_win_rate_upper": 0.54, "best_pick_probability": 0.2},
        "b": {"expected_win_rate": 0.55, "expected_win_rate_lower": 0.53, "expected_win_rate_upper": 0.57, "best_pick_probability": 0.8},
    }, insufficient=["a"])

    assert [(row["Deck"], row["Evidence"]) for row in rows] == [("b", "OK"), ("a", "Thin")]


@pytest.mark.unit
def test_best60_view_surfaces_status_evidence_and_no_signal_cards():
    view = bdif_view.best60_view({
        "cards": [],
        "status": "missing observed skeleton",
        "total_copies": 0,
        "no_signal": [{"card": "Weak"}],
        "card_evidence": {
            "Weak": {
                "inclusion_rate": 0.5,
                "field_inclusion_rate": 0.4,
                "coefficient": 0.01,
                "contribution": 0.001,
                "interval": (-1.0, 1.0),
                "q_value": 0.9,
                "bucket": "no signal",
            }
        },
    })

    assert view["status"] == "missing observed skeleton"
    assert view["no_signal"] == ["Weak"]
    assert view["evidence"][0]["Verdict"] == "no signal"
    assert view["evidence"][0]["95% low"] == -1.0


@pytest.mark.unit
def test_h1_rows_unpack_intervals_from_tuples_or_json_lists():
    rows = bdif_view.h1_rows({
        "without_variant": {"beta": 0.5, "interval": [0.1, 0.9]},
        "with_variant": {"beta": 0.4, "interval": (0.0, 0.8)},
    })

    assert [(row["95% low"], row["95% high"]) for row in rows] == [(0.1, 0.9), (0.0, 0.8)]


@pytest.mark.unit
def test_panel_rows_preserve_the_eight_report_columns():
    rows = bdif_view.panel_rows({
        "rows": {"a": [{
            "opponent": "b", "mean": 0.52, "lower": 0.5, "upper": 0.54,
            "match_count": 10, "reliable": True, "mirror": False,
        }]}
    })

    assert list(rows[0]) == [
        "Deck", "Opponent", "Posterior mean %", "95% lower %", "95% upper %",
        "Matches", "Reliable", "Mirror",
    ]