from src.tournament.reporting import build_bdif_report


def test_build_bdif_report_adds_caller_owned_sections():
    monte_carlo_result = {
        "metrics": {"a": {"win_probability": 0.5}},
        "ranked_metrics": {"a": {"win_probability": 0.5}},
        "insufficient_data": [],
        "posterior": {"draws": 0, "interval_status": "posterior disabled"},
        "field_posterior": {},
        "matchup_panel": {"rows": {}, "unmatched": [], "opponents": []},
    }

    result = build_bdif_report(monte_carlo_result, {"a": {"status": "ready"}}, {"beta": 0.25})

    assert result == {
        **monte_carlo_result,
        "best60_recommendations": {"a": {"status": "ready"}},
        "h1_report": {"beta": 0.25},
        "provenance": {},
    }


def test_build_bdif_report_carries_the_supplied_provenance():
    monte_carlo_result = {
        "metrics": {},
        "ranked_metrics": {},
        "insufficient_data": [],
        "posterior": {"draws": 0, "interval_status": "posterior disabled"},
        "field_posterior": {},
        "matchup_panel": {"rows": {}, "unmatched": [], "opponents": []},
    }

    result = build_bdif_report(monte_carlo_result, {}, {}, {"seed": 7, "input_sha256": "abc"})

    assert result["provenance"] == {"seed": 7, "input_sha256": "abc"}