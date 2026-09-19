import numpy as np
import pytest

from src.core import data as data_module
from src.core.data import cluster_decks_by_matchup_profile, load_matchup_data, safe_normalize


class _CapturingLogger:
    def __init__(self):
        self.events = []

    def warning(self, event, **kwargs):
        self.events.append((event, kwargs))

    def info(self, event, **kwargs):
        self.events.append((event, kwargs))


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
def test_missing_matchup_file_returns_empty_result_and_logs_failure(tmp_path, monkeypatch):
    logger = _CapturingLogger()
    monkeypatch.setattr(data_module, "logger", logger)
    path = tmp_path / "missing.json"

    result = load_matchup_data(str(path))

    assert result[0] == []
    assert result[1].shape == (0, 0)
    assert result[2] == {}
    assert any(event == "matchup_data_load_failed" for event, _ in logger.events)
    failed_event, failure = next(item for item in logger.events if item[0] == "matchup_data_load_failed")
    assert failed_event == "matchup_data_load_failed"
    assert failure["path"] == str(path)
    assert failure["error_type"] == "FileNotFoundError"


@pytest.mark.unit
def test_undecodable_matchup_file_returns_empty_result_and_logs_failure(tmp_path, monkeypatch):
    logger = _CapturingLogger()
    monkeypatch.setattr(data_module, "logger", logger)
    path = tmp_path / "undecodable.json"
    path.write_bytes(b"\xff\xfe\x00")

    result = load_matchup_data(str(path))

    assert result[0] == []
    assert result[1].shape == (0, 0)
    assert result[2] == {}
    assert any(event == "matchup_data_load_failed" for event, _ in logger.events)
    failed_event, failure = next(item for item in logger.events if item[0] == "matchup_data_load_failed")
    assert failed_event == "matchup_data_load_failed"
    assert failure["path"] == str(path)
    assert failure["error_type"] == "UnicodeDecodeError"
    assert "invalid start byte" in failure["error"]


@pytest.mark.unit
def test_malformed_matchup_json_returns_empty_result_and_logs_failure(tmp_path, monkeypatch):
    logger = _CapturingLogger()
    monkeypatch.setattr(data_module, "logger", logger)
    path = tmp_path / "malformed.json"
    path.write_text('{"archetypes":', encoding="utf-8")

    result = load_matchup_data(str(path))

    assert result[0] == []
    assert result[1].shape == (0, 0)
    assert result[2] == {}
    assert any(event == "matchup_data_load_failed" for event, _ in logger.events)
    failed_event, failure = next(item for item in logger.events if item[0] == "matchup_data_load_failed")
    assert failed_event == "matchup_data_load_failed"
    assert failure["path"] == str(path)
    assert failure["error_type"] == "JSONDecodeError"
    assert "Expecting value" in failure["error"]


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
