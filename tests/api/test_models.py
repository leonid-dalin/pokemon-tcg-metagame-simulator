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


@pytest.mark.unit
def test_prediction_request_rejects_job_id_longer_than_sixty_four_characters():
    with pytest.raises(ValidationError):
        PredictionRequest(**_request_payload(job_id="x" * 65))


@pytest.mark.unit
def test_prediction_request_rejects_matrix_with_more_than_sixty_four_decks():
    deck_names = [f"deck-{index}" for index in range(65)]
    matchup_matrix = [
        [0.5 if row == column else 0.0 for column in range(65)]
        for row in range(65)
    ]

    with pytest.raises(ValidationError):
        PredictionRequest(deck_names=deck_names, matchup_matrix=matchup_matrix)


@pytest.mark.unit
def test_prediction_request_rejects_duplicate_deck_names():
    with pytest.raises(ValidationError):
        PredictionRequest(**_request_payload(deck_names=["dragon", "dragon"]))


@pytest.mark.unit
def test_prediction_request_rejects_asymmetric_matchup_matrix():
    with pytest.raises(ValidationError):
        PredictionRequest(**_request_payload(matchup_matrix=[[0.5, 0.7], [0.5, 0.5]]))


@pytest.mark.unit
def test_prediction_request_accepts_tiny_floating_point_symmetry_error():
    request = PredictionRequest(
        **_request_payload(matchup_matrix=[[0.5, 0.1 + 0.2], [0.7, 0.5]])
    )

    assert request.matchup_matrix[0][1] == pytest.approx(0.3)


@pytest.mark.unit
@pytest.mark.parametrize(("panel", "message"), [
    (["c"], "bdif_panel_decks"),
    (["a", "a"], "unique"),
])
def test_panel_decks_must_be_unique_known_decks(panel, message):
    with pytest.raises(ValidationError, match=message):
        PredictionRequest(**_request_payload(bdif_panel_decks=panel))


@pytest.mark.unit
def test_panel_decks_are_capped_at_ten():
    names = [f"d{i}" for i in range(11)]
    matrix = [[0.5] * 11 for _ in range(11)]
    with pytest.raises(ValidationError):
        PredictionRequest(deck_names=names, matchup_matrix=matrix, bdif_panel_decks=names)
