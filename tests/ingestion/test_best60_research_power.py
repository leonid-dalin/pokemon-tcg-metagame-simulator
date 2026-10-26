import numpy as np
import pytest
from scipy import sparse

from src.ingestion.best60_research.power import interaction_information


def test_balanced_pair_has_analytical_information_n_over_64():
    X = np.tile(np.array([[0, 0], [0, 1], [1, 0], [1, 1]], float), (100, 1))
    nuisance = np.column_stack([np.ones(len(X)), X])
    pair = X[:, 0] * X[:, 1]
    result = interaction_information(nuisance, pair, probabilities=np.full(len(X), 0.5))
    assert result["status"] == "identified"
    assert result["information"] == pytest.approx(len(X) / 64)
    assert result["mde80"] is not None


def test_exact_alias_has_zero_information_and_no_mde():
    X = np.tile(np.array([[0, 0], [0, 1], [1, 0], [1, 1]], float), (20, 1))
    nuisance = np.column_stack([np.ones(len(X)), X])
    result = interaction_information(nuisance, X[:, 0], probabilities=np.full(len(X), 0.5))
    assert result["status"] == "not_identified"
    assert result["information"] == pytest.approx(0.0, abs=1e-10)
    assert result["mde80"] is None


def test_zero_weights_and_extreme_probabilities_have_zero_information():
    nuisance = np.ones((5, 1))
    exposure = np.array([0, 1, 0, 1, 1], dtype=float)
    result = interaction_information(
        nuisance,
        exposure,
        probabilities=np.array([0.0, 1.0, 0.5, 0.5, 0.5]),
        sample_weight=np.array([1.0, 1.0, 0.0, 0.0, 0.0]),
    )
    assert result["status"] == "not_identified"
    assert result["information"] == 0.0
    assert result["exposure_count"] == 0


def test_cluster_count_and_weighted_design_are_reported():
    nuisance = np.ones((8, 1))
    exposure = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=float)
    clusters = np.repeat(np.arange(4), 2)
    result = interaction_information(
        nuisance,
        exposure,
        probabilities=np.full(8, 0.5),
        sample_weight=np.array([1.0, 0.5] * 4),
        cluster_ids=clusters,
    )
    assert result["cluster_count"] == 4
    assert result["exposure_count"] == 4
    assert result["information"] > 0


def test_sparse_exposure_vector_uses_same_information_as_dense_vector():
    nuisance = sparse.csr_matrix(np.column_stack([np.ones(8), [0, 0, 1, 1] * 2]))
    exposure = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=float)
    kwargs = {"probabilities": np.full(8, 0.5), "sample_weight": np.arange(1, 9)}
    dense = interaction_information(nuisance, exposure, **kwargs)
    csr = interaction_information(nuisance, sparse.csr_matrix(exposure[:, None]), **kwargs)
    assert csr["information"] == pytest.approx(dense["information"])
    assert csr["mde80"] == pytest.approx(dense["mde80"])
