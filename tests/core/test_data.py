import numpy as np
import pytest

from src.core.data import safe_normalize


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
