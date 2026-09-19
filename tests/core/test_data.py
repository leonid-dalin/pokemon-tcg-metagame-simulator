import numpy as np
import pytest

from src.core.data import cluster_decks_by_matchup_profile, safe_normalize


@pytest.mark.unit
def test_safe_normalize_scales_a_positive_vector_to_sum_one():
    result = safe_normalize(np.array([1.0, 3.0], dtype=float))
    assert result.sum() == pytest.approx(1.0)
    assert result.tolist() == pytest.approx([0.25, 0.75])


@pytest.mark.unit
def test_safe_normalize_returns_uniform_for_an_all_zero_vector():
    result = safe_normalize(np.zeros(4, dtype=float))
    assert result.tolist() == pytest.approx([0.25, 0.25, 0.25, 0.25])


@pytest.mark.unit
def test_safe_normalize_returns_uniform_for_a_negative_sum():
    result = safe_normalize(np.array([-2.0, 1.0], dtype=float))
    assert result.tolist() == pytest.approx([0.5, 0.5])


@pytest.mark.unit
def test_identical_rows_collapse_to_a_single_reported_cluster():
    names = ["a", "b", "c", "d"]
    win_matrix = np.tile(np.array([0.5, 0.5, 0.5, 0.5], dtype=float), (4, 1))

    result = cluster_decks_by_matchup_profile(
        win_matrix, names, method="kmeans", n_clusters="auto"
    )

    assert result["n_clusters"] == len(set(result["labels"]))


@pytest.mark.unit
def test_distinct_profiles_still_cluster():
    names = ["a", "b", "c", "d"]
    win_matrix = np.array([
        [0.5, 0.9, 0.9, 0.9],
        [0.1, 0.5, 0.9, 0.9],
        [0.1, 0.1, 0.5, 0.9],
        [0.1, 0.1, 0.1, 0.5],
    ], dtype=float)

    result = cluster_decks_by_matchup_profile(
        win_matrix, names, method="kmeans", n_clusters="auto"
    )

    assert len(result["labels"]) == 4
    assert result["n_clusters"] >= 2
