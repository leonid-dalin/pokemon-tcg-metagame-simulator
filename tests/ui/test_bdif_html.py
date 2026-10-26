import sys
from types import SimpleNamespace

import numpy as np
import pytest

from src.bdif import cli
from src.ui.bdif_html import render_report_html


def _recommendation(**overrides):
    base = {
        "status": "complete",
        "deck_ids": ["dragapult-ex"],
        "legal_list_count": 7842,
        "list_mode": "novel",
        "total_copies": 60,
        "ace_spec_choice": "Unfair Stamp",
        "cards": [{"card": "Boss's Orders", "copies": 3, "consensus_copies": 3, "source": "consensus"}],
        "card_stats": {"Boss's Orders": {"play_rate": 0.9996, "top25": {"with_card": 0.383}}},
        "match_win_rate": {"archetype_average": 0.51775, "with_swaps_held_out": 0.528},
        "model": {"prior_sd": 0.05, "opponent_sd": 0.1, "half_life_days": 42.0},
        "card_matchups": {
            "Munkidori #2": {"Basic Box": [0.225, 0.051, 0.400], "Slowking": [0.010, -0.100, 0.120]},
        },
        "support": {"consensus": {"2": 3418}, "recommended": {"2": 3418}},
    }
    base.update(overrides)
    return base


def _report(**overrides):
    base = {
        "status": "complete",
        "best60_recommendations": {"Dragapult": _recommendation()},
        "h1_report": {},
        "provenance": {"database_sha256": "215f8d50"},
    }
    base.update(overrides)
    return base


@pytest.mark.unit
def test_report_escapes_values_that_carry_markup():
    report = _report(best60_recommendations={"<script>alert(1)</script>": _recommendation()})

    document = render_report_html(report)

    assert "<script>" not in document
    assert "&lt;script&gt;" in document


@pytest.mark.unit
def test_report_is_one_self_contained_document():
    document = render_report_html(_report())

    assert document.startswith("<!DOCTYPE html>")
    assert '<meta name="viewport" content="width=device-width, initial-scale=1">' in document
    assert "<style>" in document
    assert 'src="' not in document
    assert 'href="' not in document


@pytest.mark.unit
def test_report_names_the_archetype_when_it_carries_one():
    assert "<h1>Dragapult BDIF report</h1>" in render_report_html(_report())


@pytest.mark.unit
def test_report_reads_sections_from_the_command_result():
    result = {
        "mc_results": {
            "ranked_metrics": {"Dragapult": {"win_probability": 0.031, "top_cut_share": 0.05}},
            "field_posterior": {},
            "provenance": {"database_sha256": "215f8d50"},
        },
        "solver_results": {},
    }

    document = render_report_html(result)

    assert "<h2>Ranked decks</h2>" in document
    assert "<h1>BDIF report</h1>" in document


@pytest.mark.unit
def test_report_omits_sections_the_report_does_not_carry():
    document = render_report_html(_report())

    assert "<h2>Ranked decks</h2>" not in document
    assert "<h2>Matchup panel</h2>" not in document


@pytest.mark.unit
def test_report_keeps_an_applied_swap():
    report = _report(best60_recommendations={"Dragapult": _recommendation(swaps=[{
        "remove": "Darkness Energy", "remove_copy": 2, "add": "Team Rocket's Watchtower", "add_copy": 1,
        "gain": 0.0542, "probability": 0.803,
    }])})

    document = render_report_html(report)

    assert "Applied swaps" in document
    assert "Darkness Energy (copy 2)" in document
    assert "Team Rocket&#x27;s Watchtower (copy 1)" in document


@pytest.mark.unit
def test_report_separates_the_opponents_a_card_effect_clears():
    document = render_report_html(_report())

    assert "<h4>Munkidori #2</h4>" in document
    assert "Basic Box" in document
    assert "Slowking" not in document
    assert "No separation from zero against 1 other opponent." in document


@pytest.mark.unit
@pytest.mark.parametrize("result", [None, [], "text"])
def test_report_tolerates_a_result_that_is_not_a_mapping(result):
    document = render_report_html(result)

    assert document.startswith("<!DOCTYPE html>")
    assert "Status: unknown" in document


@pytest.mark.unit
def test_report_tolerates_sections_that_are_not_mappings():
    document = render_report_html({"mc_results": {
        "ranked_metrics": None,
        "best60_recommendations": {"status": "failed"},
        "field_posterior": "broken",
        "insufficient_data": None,
        "provenance": {"database_sha256": "215f8d50"},
    }})

    assert "<h2>Ranked decks</h2>" not in document
    assert "<h2>Best-60 card selection</h2>" not in document
    assert "<h2>Provenance</h2>" in document


@pytest.mark.unit
def test_report_hides_the_card_columns_that_say_nothing():
    document = render_report_html(_report())

    assert "<th>Card</th>" in document
    assert "<th>Source</th>" not in document
    assert "<th>Consensus</th>" not in document


@pytest.mark.unit
def test_report_keeps_the_card_columns_that_show_a_swapped_count():
    report = _report(best60_recommendations={"Dragapult": _recommendation(cards=[
        {"card": "Munkidori", "copies": 2, "consensus_copies": 2, "source": "consensus"},
        {"card": "Team Rocket's Watchtower", "copies": 1, "consensus_copies": 0, "source": "swap"},
    ])})

    document = render_report_html(report)

    assert "Consensus</abbr>" in document
    assert "Source</abbr>" in document


@pytest.mark.unit
def test_report_drops_a_column_that_is_empty_in_every_row():
    report = _report(best60_recommendations={"Dragapult": _recommendation(
        cards=[{"card": "Munkidori", "copies": 2, "consensus_copies": 2, "source": "consensus"}],
        stability={},
    )})

    document = render_report_html(report)

    assert "<th>Card</th>" in document
    assert "<th>Same count in resamples %</th>" not in document


@pytest.mark.unit
def test_report_explains_the_h1_check():
    report = _report(h1_report={
        "hypothesis": "H1", "card": "Mist Energy", "target": "Alakazam Dudunsparce",
        "without_variant": {"beta": -0.3366, "interval": [-0.4260, -0.2471]},
        "with_variant": {"beta": -0.3383, "interval": [-0.4286, -0.2479]},
        "status": "rejected", "interpretation": "observational association, not a causal effect",
    })

    document = render_report_html(report)

    assert "<h2>H1 check</h2>" in document
    assert "Mist Energy in Alakazam Dudunsparce decks" in document
    assert "control for the Hammer build" in document


@pytest.mark.unit
def test_report_glosses_the_terms_it_abbreviates():
    document = render_report_html(_report())

    assert '<abbr title="novel lets the builder propose a list nobody has played' in document
    assert '<abbr title="Share of the archetype&#x27;s lists that run the card.">Play rate %</abbr>' in document


@pytest.mark.unit
def test_report_says_why_the_consensus_and_recommended_rows_match():
    document = render_report_html(_report())

    assert "the recommended list is the observed list the builder started from" in document


@pytest.mark.unit
def test_report_leaves_out_the_consensus_note_when_a_swap_was_applied():
    report = _report(best60_recommendations={"Dragapult": _recommendation(swaps=[{
        "remove": "Darkness Energy", "remove_copy": 2, "add": "Team Rocket's Watchtower", "add_copy": 1,
        "gain": 0.0542, "probability": 0.803,
    }])})

    assert "the recommended list is the consensus list" not in render_report_html(report)


@pytest.mark.unit
def test_report_command_writes_the_html_alongside_the_json(monkeypatch, tmp_path):
    input_path = tmp_path / "input.json"
    input_path.write_text("{}", encoding="utf-8")
    calls = []

    service = SimpleNamespace(
        simulation_input_path=lambda: str(input_path),
        run_prediction=lambda request, **kwargs: {
            "mc_results": {"ranked_metrics": {"Dragapult": {"win_probability": 0.031}}, "provenance": {}},
            "solver_results": {},
        },
    )
    monkeypatch.setattr("sys.argv", ["tcg-bdif", "report", "-i", str(input_path), "-o", str(tmp_path)])
    monkeypatch.setattr(cli, "setup_structured_logging", lambda: None)
    monkeypatch.setattr(cli, "setup_telemetry", lambda name: None)
    monkeypatch.setitem(sys.modules, "src.bdif.service", service)
    monkeypatch.setattr("src.core.data.load_matchup_data", lambda path: (
        ["Dragapult", "Slowking"], np.array([[0.5, 0.55], [0.45, 0.5]]), calls.append(path),
    ))

    assert cli.main() == 0
    assert calls == [str(input_path)]
    assert (tmp_path / "bdif_report.json").exists()
    document = (tmp_path / "bdif_report.html").read_text(encoding="utf-8")
    assert document.startswith("<!DOCTYPE html>")
    assert "<h2>Ranked decks</h2>" in document
