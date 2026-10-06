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
    ("Player (Micro)", {"Win Event % low", "Win Event % high", "Interval note"}),
    ("Archetype (Macro)", {"Share % (Day 2) low", "Share % (Day 2) high", "Share % (Top 8) low", "Share % (Top 8) high", "Interval note"}),
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
@pytest.mark.parametrize(("odds_view", "expected_note"), [
    ("Player (Micro)", "Win Event %: includes simulation noise"),
    ("Archetype (Macro)", "Share % (Top 8): includes simulation noise"),
])
def test_interval_columns_name_intervals_that_keep_simulation_noise(odds_view, expected_note):
    metrics = {
        "win_probability_lower": 0.01, "win_probability_upper": 0.03, "win_probability_interval": "raw",
        "day2_share_lower": 0.1, "day2_share_upper": 0.2, "day2_share_interval": "denoised",
        "top_cut_share_lower": 0.05, "top_cut_share_upper": 0.15, "top_cut_share_interval": "raw",
    }
    metrics_without_noise = {key: "denoised" if key.endswith("_interval") else value for key, value in metrics.items()}

    noisy = bdif_view.interval_columns(metrics, {"interval_status": "ok"}, odds_view, top_cut=8, d2_rounds=2)
    clean = bdif_view.interval_columns(metrics_without_noise, {"interval_status": "ok"}, odds_view, top_cut=8, d2_rounds=2)

    assert noisy["Interval note"] == expected_note
    assert clean["Interval note"] == ""


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
def test_best60_view_shows_sources_swaps_held_out_rates_and_breakthroughs():
    swap = {"remove": "Boss's Orders", "remove_copy": 4, "add": "Judge", "add_copy": 1, "gain": 0.0512, "probability": 0.8123}
    view = bdif_view.best60_view({
        "status": "consensus kept: swaps did not hold up on held-out events",
        "deck_ids": ["crustle-dri"],
        "legal_list_count": 1073,
        "ace_spec_choice": "Hero's Cape",
        "total_copies": 60,
        "cards": [{"card": "Crustle", "copies": 3, "consensus_copies": 3, "source": "consensus"}],
        "removed_cards": [{"card": "Lumiose City", "copies": 0, "consensus_copies": 1, "source": "swap"}],
        "swaps": [],
        "proposed_swaps": [swap],
        "leaning_swaps": [],
        "model": {"prior_sd": 0.05},
        "match_win_rate": {"archetype_average": 0.5032, "with_swaps_in_sample": 0.5592, "with_swaps_held_out": 0.4986},
        "card_stats": {"Crustle": {"play_rate": 1.0, "top25": {"with_card": 0.37}}},
        "trends": {"breakthrough": ["Judge"], "rising": [{"card": "Judge", "start_share": 0.1, "end_share": 0.5, "change": 0.4, "copies_change": 0.5, "model_effect": 0.02}], "falling": []},
    })

    assert view["status"] == "consensus kept: swaps did not hold up on held-out events"
    assert view["deck_ids"] == ["crustle-dri"]
    assert view["cards"] == [
        {"Card": "Crustle", "Copies": 3, "Consensus": 3, "Source": "consensus", "Play rate %": 100.0, "Top 25% rate %": 37.0, "Same count in resamples %": None},
        {"Card": "Lumiose City", "Copies": 0, "Consensus": 1, "Source": "swap", "Play rate %": None, "Top 25% rate %": None, "Same count in resamples %": None},
    ]
    assert view["swaps"] == []
    assert view["proposed_swaps"] == [{"Out": "Boss's Orders (copy 4)", "In": "Judge (copy 1)", "Gain (log-odds)": 0.051, "Chance it helps %": 81.23}]
    assert view["win_rates"] == {"Archetype average %": 50.32, "With swaps, held out %": 49.86, "With swaps, in sample %": 55.92}
    assert view["breakthrough"] == ["Judge"]
    assert view["trends"][0]["Change (points)"] == 40.0


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


@pytest.mark.unit
def test_best60_view_shows_level_moves_support_and_the_whole_list_chance():
    move = {
        "remove": "Transformation Tome", "remove_copy": 4, "add": "Cyrano", "add_copy": 3,
        "removed": [["Transformation Tome", 4], ["Transformation Tome", 3], ["Transformation Tome", 2], ["Transformation Tome", 1]],
        "added": [["Cyrano", 3], ["Judge", 1], ["Team Rocket's Watchtower", 1], ["Team Rocket's Watchtower", 2]],
        "gain": 0.111, "probability": 0.998,
    }
    view = bdif_view.best60_view({
        "cards": [], "swaps": [move], "list_mode": "observed", "joint_probability": 0.9951,
        "support": {"consensus": {2: 489, 4: 1446, 6: 1790}, "recommended": {2: 0, 4: 48, 6: 533}},
    })

    assert view["swaps"][0]["Out"] == "Transformation Tome (copies 1-4)"
    assert view["swaps"][0]["In"] == "Cyrano (copy 3), Judge (copy 1), Team Rocket's Watchtower (copies 1-2)"
    assert view["list_mode"] == "observed"
    assert view["joint_probability"] == 99.51
    assert view["support"] == [
        {"List": "Consensus", "Played lists within 2 changes": 489, "Played lists within 4 changes": 1446, "Played lists within 6 changes": 1790},
        {"List": "Recommended", "Played lists within 2 changes": 0, "Played lists within 4 changes": 48, "Played lists within 6 changes": 533},
    ]


@pytest.mark.unit
def test_best60_view_shows_how_often_resamples_keep_each_change():
    view = bdif_view.best60_view({
        "cards": [{"card": "Cyrano", "copies": 3, "consensus_copies": 2, "source": "swap"}],
        "removed_cards": [{"card": "Transformation Tome", "copies": 0, "consensus_copies": 4, "source": "swap"}],
        "stability": {
            "Cyrano": {"consensus": 2, "recommended": 3, "same_count": 0.9, "same_direction": 0.9, "draws": 30},
            "Transformation Tome": {"consensus": 4, "recommended": 0, "same_count": 1.0, "same_direction": 1.0, "draws": 30},
        },
    })
    assert [(row["Card"], row["Same count in resamples %"]) for row in view["cards"]] == [("Cyrano", 90.0), ("Transformation Tome", 100.0)]
