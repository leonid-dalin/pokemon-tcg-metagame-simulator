import numpy as np
import json

import pytest

from src.ingestion.best60_research.estimators import FitBudgetExceeded
from src.ingestion.best60_research.uncertainty import RefitFailure, fixed_list_interval


def test_fixed_list_interval_resamples_whole_utc_date_blocks_and_keeps_list_fixed():
    events = [f"e{i}" for i in range(60)]
    block_ids = {event: f"utc-date:{i // 2:02d}" for i, event in enumerate(events)}
    context = {"training_event_ids": events, "event_block_ids": block_ids, "unchanged": [1, 2]}
    original = dict(context)
    fixed = {"Core": 1, "Grass Energy": 59}
    seen = []

    def fit_and_score(sample, counts):
        assert counts == fixed
        counts["Callback mutation"] = 1
        chosen_events = sample["resample_event_ids"]
        chosen_blocks = sample["resample_block_ids"]
        assert len(chosen_blocks) == 30
        assert len(chosen_events) == 60
        seen.append((chosen_blocks, chosen_events))
        odd_blocks = sum(int(block.rsplit(":", 1)[1]) % 2 for block in chosen_blocks)
        return 0.3 + 0.4 * odd_blocks / len(chosen_blocks)

    result = fixed_list_interval(context, fixed, fit_and_score=fit_and_score, draws=20, seed=1312)
    assert len(seen) == 20
    assert any(len(set(blocks)) < 30 for blocks, _ in seen)
    for blocks, chosen_events in seen:
        expected = [event for block in blocks for event in events if block_ids[event] == block]
        assert list(chosen_events) == expected
    assert result["scope"] == "model_conditional"
    assert result["target"] == "expected_decided_pairing_win_probability"
    assert result["resampling_unit"] == "atomic_date_block"
    assert result["status"] == "complete"
    assert context == original
    assert "resample_event_ids" not in context
    assert fixed == {"Core": 1, "Grass Energy": 59}


def test_fixed_list_interval_counts_failed_draws_and_returns_incomplete():
    context = {"training_event_ids": [f"e{i}" for i in range(5)]}
    counter = iter([0.4, RefitFailure("fit did not converge"), 0.5, RefitFailure("singular design"), 0.6])

    def fit_and_score(sample, counts):
        value = next(counter)
        if isinstance(value, Exception):
            raise value
        return value

    result = fixed_list_interval(context, {"Core": 1}, fit_and_score=fit_and_score, draws=5)
    assert result["status"] == "incomplete"
    assert result["effective_draws"] == 3
    assert result["failed_draws"] == 2
    assert result["failure_reasons"] == {"fit did not converge": 1, "singular design": 1}
    assert any("does not establish causal or post-selection coverage" in item for item in result["assumptions"])


def test_fixed_list_interval_resume_repeats_pending_draw_and_matches_uninterrupted(tmp_path):
    events = [f"e{i}" for i in range(24)]
    context = {
        "training_event_ids": events,
        "event_block_ids": {event: f"utc-date:{i // 2:02d}" for i, event in enumerate(events)},
        "provenance": {"source_sha256": "source-test", "cutoff": "2026-09-01T00:00:00+00:00"},
    }
    counts = {"Core": 1, "Grass Energy": 59}
    checkpoint = tmp_path / "interval.json"
    attempted_blocks = []
    resumed_scores = []

    def interrupted(sample, _counts):
        blocks = tuple(sample["resample_block_ids"])
        attempted_blocks.append(blocks)
        if len(attempted_blocks) == 3:
            raise FitBudgetExceeded("refit time slice expired")
        score = 0.3 + (sum(int(block[-2:]) for block in blocks) % 10) / 100
        resumed_scores.append(score)
        return score

    first = fixed_list_interval(context, counts, fit_and_score=interrupted, draws=12, seed=71,
                                checkpoint_path=checkpoint, context_key="model-cache-key",
                                max_seconds=30)
    assert first["status"] == "incomplete"
    assert first["next_index"] == 2
    assert first["pending_draws"] == 10
    assert first["failed_draws"] == 0
    assert first["incomplete_reason"] == "refit time slice expired"
    saved = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert saved["next_index"] == 2
    assert len(saved["scores"]) == 2
    assert saved["failures"] == []

    retry_blocks = []

    def resumed(sample, _counts):
        blocks = tuple(sample["resample_block_ids"])
        retry_blocks.append(blocks)
        score = 0.3 + (sum(int(block[-2:]) for block in blocks) % 10) / 100
        resumed_scores.append(score)
        return score

    final = fixed_list_interval(context, counts, fit_and_score=resumed, draws=12, seed=71,
                                checkpoint_path=checkpoint, context_key="model-cache-key",
                                max_seconds=30)
    uninterrupted_scores = []
    uninterrupted_blocks = []

    def uninterrupted(sample, _counts):
        blocks = tuple(sample["resample_block_ids"])
        uninterrupted_blocks.append(blocks)
        score = 0.3 + (sum(int(block[-2:]) for block in blocks) % 10) / 100
        uninterrupted_scores.append(score)
        return score

    baseline = fixed_list_interval(context, counts, fit_and_score=uninterrupted, draws=12, seed=71,
                                   context_key="model-cache-key", max_seconds=30)
    assert final["status"] == baseline["status"] == "complete"
    assert final["next_index"] == baseline["next_index"] == 12
    assert retry_blocks[0] == attempted_blocks[2] == uninterrupted_blocks[2]
    assert resumed_scores == uninterrupted_scores
    assert final["interval"] == baseline["interval"]
    assert final["estimate"] == baseline["estimate"]


def test_fixed_list_interval_checkpoint_rejects_changed_context_or_fixed_list(tmp_path):
    context = {"training_event_ids": [f"e{i}" for i in range(8)]}
    checkpoint = tmp_path / "interval.json"
    fixed_list_interval(context, {"Core": 1}, fit_and_score=lambda *_: 0.5, draws=4,
                        checkpoint_path=checkpoint, context_key="model-a")
    with pytest.raises(ValueError, match="does not match context"):
        fixed_list_interval(context, {"Core": 2}, fit_and_score=lambda *_: 0.5, draws=4,
                            checkpoint_path=checkpoint, context_key="model-a")


def test_fixed_list_interval_propagates_programming_errors_from_callback():
    context = {"training_event_ids": [f"e{i}" for i in range(5)]}
    with pytest.raises(ValueError, match="callback bug"):
        fixed_list_interval(
            context,
            {"Core": 1},
            fit_and_score=lambda *_: (_ for _ in ()).throw(ValueError("callback bug")),
            draws=3,
        )


def test_fixed_list_interval_is_inconclusive_with_one_block_or_degenerate_draws():
    context = {"training_event_ids": ["e1", "e2"], "event_block_ids": {"e1": "d1", "e2": "d1"}}
    no_resample = fixed_list_interval(context, {"Core": 1}, fit_and_score=lambda *_: 0.5, draws=10)
    assert no_resample["status"] == "inconclusive"
    assert no_resample["reason"] == "fewer_than_two_resampling_blocks"

    context = {"training_event_ids": ["e1", "e2"]}
    degenerate = fixed_list_interval(context, {"Core": 1}, fit_and_score=lambda *_: 0.5, draws=10)
    assert degenerate["status"] == "inconclusive"
    assert degenerate["interval_width"] == 0.0


def _probability_range(values, *, seed=1312, draws=120):
    events = list(range(len(values)))
    context = {"training_event_ids": events, "event_block_ids": {event: event for event in events}}
    return fixed_list_interval(
        context, {"Core": 1, "Grass Energy": 59},
        fit_and_score=lambda sample, _: float(np.mean([values[event] for event in sample["resample_event_ids"]])),
        draws=draws, seed=seed,
    )


@pytest.fixture
def probability_range():
    return _probability_range


def test_coverage_summary_keeps_latent_and_future_targets_separate(probability_range):
    result = probability_range([0.4, 0.6] * 25)
    assert result["scope"] == "model_conditional"
    assert result["interval"][0] < 0.5 < result["interval"][1]
    assert "predictive_coverage" not in result
    assert "policy_value_coverage" not in result


def test_clustered_coverage_with_small_or_zero_variance_is_inconclusive(probability_range):
    small = probability_range([0.5], draws=20)
    assert small["status"] == "inconclusive"
    assert small["reason"] == "fewer_than_two_resampling_blocks"
    zero_variance = probability_range([0.5] * 20, draws=20)
    assert zero_variance["status"] == "inconclusive"
    assert zero_variance["interval_width"] == 0.0


def test_coverage_validation_uses_known_latent_mean_and_future_match_noise(probability_range):
    rng = np.random.default_rng(1312)
    probability = 0.65
    replicates = 200
    latent_covered = 0
    future_covered = 0
    for replicate in range(replicates):
        observed = rng.binomial(1, probability, size=100)
        result = probability_range(observed, seed=replicate, draws=120)
        lower, upper = result["interval"]
        latent_covered += lower <= probability <= upper
        future_average = rng.binomial(60, probability) / 60
        future_covered += lower <= future_average <= upper
    # A known latent truth checks the actual resampler. Future match noise is a
    # different target, so its lower coverage must never be called calibration.
    assert 0.88 <= latent_covered / replicates <= 0.99
    assert future_covered / replicates < latent_covered / replicates - 0.1
