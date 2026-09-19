import pytest
from pydantic import ValidationError

from src.api.models import PredictionRequest, RangeSpec


def _request_payload(**overrides):
    payload = {
        "deck_names": ["a", "b"],
        "matchup_matrix": [[0.5, 0.6], [0.4, 0.5]],
    }
    payload.update(overrides)
    return payload


@pytest.mark.unit
def test_dead_meta_constraints_field_is_rejected():
    with pytest.raises(ValidationError):
        PredictionRequest(**_request_payload(meta_constraints={"a": {"exact": 0.5}}))


@pytest.mark.unit
def test_range_spec_rejects_a_minimum_above_the_maximum():
    with pytest.raises(ValidationError, match="min must be less than or equal to max"):
        RangeSpec(min=0.8, max=0.2)


@pytest.mark.unit
def test_tournament_style_accepts_only_supported_values():
    with pytest.raises(ValidationError):
        PredictionRequest(**_request_payload(tournament_style="single_elimination"))