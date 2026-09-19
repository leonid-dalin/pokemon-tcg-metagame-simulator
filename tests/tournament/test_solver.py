import numpy as np
import pytest

from src.api.models import ExactSpec, RangeSpec
from src.tournament.solver import calculate_empirical_baseline, resolve_meta_constraints

DECKS = ["a", "b", "c", "d"]
IDX = {name: i for i, name in enumerate(DECKS)}
BASELINE = np.array([0.4, 0.3, 0.2, 0.1], dtype=float)


@pytest.mark.unit
def test_a_bare_float_pins_an_exact_share():
    result = resolve_meta_constraints(BASELINE, {"a": 0.5}, IDX)
    assert result[IDX["a"]] == pytest.approx(0.5)
    assert result.sum() == pytest.approx(1.0)


@pytest.mark.unit
def test_an_exact_spec_model_pins_an_exact_share():
    result = resolve_meta_constraints(BASELINE, {"a": ExactSpec(exact=0.5)}, IDX)
    assert result[IDX["a"]] == pytest.approx(0.5)
    assert result.sum() == pytest.approx(1.0)


@pytest.mark.unit
def test_unconstrained_decks_split_the_remainder_by_baseline_weight():
    result = resolve_meta_constraints(BASELINE, {"a": 0.4}, IDX)
    assert result[IDX["b"]] == pytest.approx(0.3)
    assert result[IDX["c"]] == pytest.approx(0.2)
    assert result[IDX["d"]] == pytest.approx(0.1)


@pytest.mark.unit
def test_a_range_spec_caps_allocation_at_its_max():
    result = resolve_meta_constraints(BASELINE, {"b": RangeSpec(min=0.0, max=0.05)}, IDX)
    assert result[IDX["b"]] <= 0.05 + 1e-6
    assert result.sum() == pytest.approx(1.0)


@pytest.mark.unit
def test_a_range_spec_floor_is_honoured():
    result = resolve_meta_constraints(BASELINE, {"d": RangeSpec(min=0.4, max=1.0)}, IDX)
    assert result[IDX["d"]] >= 0.4 - 1e-6
    assert result.sum() == pytest.approx(1.0)


@pytest.mark.unit
def test_overallocated_minimum_constraints_are_rejected():
    with pytest.raises(ValueError, match="cannot satisfy minimum constraints"):
        resolve_meta_constraints(BASELINE, {"a": 0.8, "b": 0.8}, IDX)


@pytest.mark.unit
def test_unknown_deck_names_are_ignored():
    result = resolve_meta_constraints(BASELINE, {"not_a_deck": 0.9}, IDX)
    assert result.sum() == pytest.approx(1.0)


@pytest.mark.unit
def test_an_empty_spec_returns_the_normalised_baseline():
    result = resolve_meta_constraints(BASELINE, {}, IDX)
    assert result.tolist() == pytest.approx(BASELINE.tolist())


@pytest.mark.unit
def test_every_deck_capped_below_one_is_rejected_as_infeasible():
    spec = {name: RangeSpec(min=0.0, max=0.1) for name in DECKS}
    with pytest.raises(ValueError, match="cannot satisfy maximum constraints"):
        resolve_meta_constraints(BASELINE, spec, IDX)


@pytest.mark.unit
def test_minimum_constraints_cannot_exceed_field_capacity():
    with pytest.raises(ValueError, match="cannot satisfy minimum constraints"):
        resolve_meta_constraints(BASELINE, {"a": 0.8, "b": 0.8}, IDX)


@pytest.mark.unit
def test_empirical_baseline_weights_by_match_volume():
    details = {
        ("a", "b"): {"match_count": 30.0},
        ("b", "a"): {"match_count": 10.0},
    }
    result = calculate_empirical_baseline(["a", "b"], details)
    assert result.tolist() == pytest.approx([0.75, 0.25])
