import pytest
from bs4 import BeautifulSoup

from src.core.scraper import scrape_matchup_soup


def _soup(rows: str) -> BeautifulSoup:
    return BeautifulSoup(
        f'<table class="striped"><tr><th>h</th></tr>{rows}</table>', "html.parser"
    )


def _row(name: str, matches: int, record: str = "60 - 30 - 10") -> str:
    return (
        f'<tr data-name="{name}" data-matches="{matches}">'
        f"<td>a</td><td>b</td><td>c</td><td>{record}</td></tr>"
    )


@pytest.mark.unit
def test_a_known_archetype_is_always_parsed():
    canonical = {"known deck": "Known Deck"}
    result = scrape_matchup_soup(_soup(_row("Known Deck", 5)), "Mine", "Standard", canonical)
    assert [m["opponent_archetype"] for m in result] == ["Known Deck"]


@pytest.mark.unit
def test_a_high_volume_unknown_archetype_is_admitted():
    canonical = {}
    result = scrape_matchup_soup(_soup(_row("Rogue Deck", 500)), "Mine", "Standard", canonical)
    assert [m["opponent_archetype"] for m in result] == ["Rogue Deck"]
    assert "rogue deck" in canonical


@pytest.mark.unit
def test_a_low_volume_unknown_archetype_is_rejected():
    canonical = {}
    result = scrape_matchup_soup(_soup(_row("Fringe Deck", 2)), "Mine", "Standard", canonical)
    assert result == []
    assert canonical == {}


@pytest.mark.unit
def test_rejecting_a_stranger_does_not_affect_a_known_deck_in_the_same_table():
    canonical = {"known deck": "Known Deck"}
    soup = _soup(_row("Fringe Deck", 1) + _row("Known Deck", 1))
    result = scrape_matchup_soup(soup, "Mine", "Standard", canonical)
    assert [m["opponent_archetype"] for m in result] == ["Known Deck"]


@pytest.mark.unit
def test_excluded_opponents_are_skipped_regardless_of_volume():
    canonical = {}
    result = scrape_matchup_soup(_soup(_row("Bye", 9999)), "Mine", "Standard", canonical)
    assert result == []


@pytest.mark.unit
def test_win_rate_counts_ties_as_half():
    canonical = {"known deck": "Known Deck"}
    result = scrape_matchup_soup(
        _soup(_row("Known Deck", 100, "60 - 30 - 10")), "Mine", "Standard", canonical
    )
    assert result[0]["win_rate"] == pytest.approx(0.65)
