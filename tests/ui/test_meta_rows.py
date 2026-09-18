import pytest

from src.ui.meta_rows import RAW_PLAYERS, locked_exact_spec, locked_share


def _row(deck="alpha", spec_type="Exact", val=25.0):
    return {"id": "r1", "deck": deck, "spec_type": spec_type, "val": val, "mode": "Percentage"}


@pytest.mark.unit
def test_percentage_mode_divides_by_one_hundred():
    rows = [_row(val=25.0), _row(deck="beta", val=15.0)]
    assert locked_share(rows, players=256, input_mode="Percentage") == pytest.approx(0.40)


@pytest.mark.unit
def test_raw_players_mode_divides_by_the_player_count():
    rows = [_row(val=64), _row(deck="beta", val=64)]
    assert locked_share(rows, players=256, input_mode=RAW_PLAYERS) == pytest.approx(0.50)


@pytest.mark.unit
def test_range_rows_are_excluded():
    rows = [_row(val=25.0), _row(deck="beta", spec_type="Range", val=(10.0, 40.0))]
    assert locked_share(rows, players=256, input_mode="Percentage") == pytest.approx(0.25)


@pytest.mark.unit
def test_a_tuple_value_on_an_exact_row_uses_its_first_element():
    rows = [_row(val=(30.0, 60.0))]
    assert locked_share(rows, players=256, input_mode="Percentage") == pytest.approx(0.30)


@pytest.mark.unit
def test_zero_players_in_raw_mode_returns_zero_rather_than_dividing_by_zero():
    assert locked_share([_row(val=10)], players=0, input_mode=RAW_PLAYERS) == 0.0


@pytest.mark.unit
def test_no_rows_returns_zero():
    assert locked_share([], players=256, input_mode="Percentage") == 0.0


@pytest.mark.unit
def test_unparseable_values_are_skipped():
    assert locked_share([_row(val=None)], players=256, input_mode="Percentage") == 0.0


@pytest.mark.unit
def test_spec_maps_deck_names_to_proportions():
    rows = [_row(deck="alpha", val=25.0), _row(deck="beta", val=15.0)]
    spec = locked_exact_spec(rows, players=256, input_mode="Percentage")
    assert spec == pytest.approx({"alpha": 0.25, "beta": 0.15})


@pytest.mark.unit
def test_spec_omits_rows_with_no_deck_selected():
    rows = [_row(deck="", val=10.0), _row(deck="alpha", val=25.0)]
    assert set(locked_exact_spec(rows, players=256, input_mode="Percentage")) == {"alpha"}


@pytest.mark.unit
def test_spec_omits_range_rows():
    rows = [_row(deck="alpha", spec_type="Range", val=(10.0, 40.0))]
    assert locked_exact_spec(rows, players=256, input_mode="Percentage") == {}
