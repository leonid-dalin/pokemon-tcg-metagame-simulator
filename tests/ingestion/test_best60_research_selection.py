import numpy as np
import pytest

from src.ingestion.best60_research.selection import ExpectedScoreFailure, field_scores, select_catalogue


def test_field_mixture_averages_probabilities_not_logits():
    logits = np.array([[4.0, -1.0], [1.0, 1.0]])
    scores = field_scores(logits, [0.5, 0.5])
    assert scores[0] == pytest.approx(0.625477607962246, abs=1e-8)
    assert scores[1] > scores[0]


def test_field_scores_normalise_nonnegative_weights():
    np.testing.assert_allclose(field_scores(np.zeros((2, 2)), [2, 1]), [0.5, 0.5])
    for weights in ([], [0, 0], [-1, 2], [1, np.nan]):
        with pytest.raises(ValueError):
            field_scores(np.zeros((1, 2)), weights)


def test_catalogue_returns_legal_global_maximum_and_deduplicates_provenance():
    candidates = [
        {"counts": {"Core": 1, "Grass Energy": 59}, "support": 2, "source_ids": ["event-a"]},
        {"counts": {"Core": 1, "Grass Energy": 59}, "support": 3, "source_ids": ["event-b"]},
        {"counts": {"Core": 2, "Grass Energy": 58}, "support": 1, "source_ids": ["event-c"]},
    ]
    result = select_catalogue(candidates, score=lambda counts: (counts["Core"] - 1) / 2)
    assert result["status"] == "selected"
    assert result["counts"] == candidates[2]["counts"]
    assert result["candidate_count"] == 2
    assert result["counts"] in [row["counts"] for row in candidates]
    assert result["scope"] == "exact_observed_catalogue"
    duplicate = select_catalogue(candidates[:2], score=lambda _: 0.5)
    assert duplicate["support"] == 5
    assert duplicate["source_ids"] == ["event-a", "event-b"]


@pytest.mark.parametrize("counts", [
    {"Core": 5, "Grass Energy": 55},
    {"Core": 59, "Grass Energy": 1},
    {"Core": 1, "Prime Catcher": 2, "Grass Energy": 57},
])
def test_validator_enforces_copy_and_energy_rules(counts):
    result = select_catalogue([counts], score=lambda _: 0.5)
    assert result["status"] == "invalid_catalogue"


def test_validator_enforces_ace_spec_count_and_declared_companions():
    ace = {"Core": 4, "Grass Energy": 55, "Prime Catcher": 1}
    too_many = {"Core": 4, "Grass Energy": 54, "Prime Catcher": 1, "Master Ball": 1}
    result = select_catalogue([ace, too_many], score=lambda _: 0.5)
    assert result["status"] == "selected"
    assert result["counts"] == ace
    dependent = {"counts": {"Core": 2, "Companion": 1, "Grass Energy": 57},
                 "companions": {"Core": {"Companion": 1}}}
    assert select_catalogue([dependent], score=lambda _: 0.5)["status"] == "invalid_catalogue"
    presence = {"counts": {"Core": 1, "Grass Energy": 59}, "companions": {"Core": {"Companion": 0}}}
    assert select_catalogue([presence], score=lambda _: 0.5)["status"] == "invalid_catalogue"


def test_basic_energy_exception_allows_more_than_four_copies():
    result = select_catalogue([{"Core": 1, "Grass Energy": 59}], score=lambda _: 0.5)
    assert result["status"] == "selected"


def test_invalid_scores_and_empty_catalogue_are_explicit():
    valid = {"Core": 1, "Grass Energy": 59}
    assert select_catalogue([], score=lambda _: 1)["status"] == "empty_catalogue"
    assert select_catalogue([valid], score=lambda _: float("nan"))["status"] == "invalid_scores"
    assert select_catalogue([valid], score=lambda _: 1.1)["status"] == "invalid_scores"


def test_expected_score_failure_is_reported_and_programming_errors_propagate():
    valid = {"Core": 1, "Grass Energy": 59}
    expected = select_catalogue(
        [valid], score=lambda _: (_ for _ in ()).throw(ExpectedScoreFailure("nonconverged model"))
    )
    assert expected["status"] == "invalid_scores"
    assert expected["score_failure_messages"] == {"nonconverged model": 1}
    with pytest.raises(ValueError, match="shape bug"):
        select_catalogue([valid], score=lambda _: (_ for _ in ()).throw(ValueError("shape bug")))


def test_equal_scores_use_stable_lexical_tie_break():
    first = {"Zed": 1, "Grass Energy": 59}
    second = {"Alpha": 1, "Grass Energy": 59}
    result = select_catalogue([first, second], score=lambda _: 0.7)
    assert result["counts"] == second
    assert result["tie_count"] == 2
