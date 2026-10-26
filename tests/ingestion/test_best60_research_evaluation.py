import numpy as np
import pytest

from src.ingestion.best60_research.evaluation import (
    balanced_h1_summary,
    decide_h1,
    player_aware_interval,
    policy_value,
    ranking_agreement,
)


@pytest.mark.parametrize("difference,expected", [(0.02, "accepted"), (-0.01, "rejected")])
def test_h1_uses_event_cluster_interval(difference, expected):
    values = np.full(600, difference)
    events = np.repeat(np.arange(60), 10)
    result = decide_h1(values, events, seed=1312, draws=300)
    assert result["status"] == expected
    assert result["estimate"] == pytest.approx(difference)
    assert result["cluster_count"] == 60
    assert result["scope"] == "predictive_h1"


def test_h1_null_does_not_pass_positive_lower_bound():
    rng = np.random.default_rng(1313)
    event_ids = np.repeat(np.arange(80), 8)
    differences = rng.normal(0, 0.01, len(event_ids))
    result = decide_h1(differences, event_ids, draws=500)
    assert result["status"] != "accepted"


def test_h1_reports_small_cluster_warning_with_interval():
    result = decide_h1(np.ones(190), np.repeat(np.arange(19), 10))
    assert result["status"] == "inconclusive"
    assert result["interval"] is not None
    assert result["warning"] == "fewer_than_20_independent_clusters"


def test_h1_balanced_summary_reports_event_and_archetype_views():
    result = balanced_h1_summary([0.1, 0.3, -0.2], ["e1", "e1", "e2"], ["A", "B", "A"])
    assert result["pairing_weighted"] == pytest.approx(0.0666667)
    assert result["event_balanced"] == pytest.approx(0.0)
    assert result["archetype_balanced"] == pytest.approx(0.125)


def test_ranking_agreement_uses_within_event_archetype_pairs_and_match_weights():
    result = ranking_agreement(
        [0.8, 0.2, 0.9], [0.5, 0.1, 0.0], [8, 5, 10],
        ["e1", "e1", "e2"], ["A", "A", "A"],
    )
    assert result["status"] == "measured"
    assert result["eligible_pair_count"] == 1
    assert result["weighted_agreement"] == 1.0
    sparse = ranking_agreement([0.5], [0.2], [4], ["e1"], ["A"])
    assert sparse["status"] == "inconclusive_sparse_support"


def test_player_aware_interval_resamples_events_and_repeated_players():
    n = 120
    events = np.repeat(np.arange(30), 4)
    first = np.tile(np.repeat(np.arange(20), 2), 3)
    second = (first + np.tile([1, 2, 3, 4], 30)) % 20
    values = np.linspace(-0.1, 0.1, n)
    result = player_aware_interval(values, events, first, second, draws=200)
    assert result["status"] == "measured"
    assert result["cluster_count"] == 30
    assert result["player_count"] == 20
    assert result["resampling_unit"] == "event_and_both_player_ids"
    assert result["scope"] == "dependence_sensitivity"


def test_player_aware_interval_keeps_same_date_events_in_one_block():
    events = [f"e{i}" for i in range(60)]
    date_blocks = [f"utc-date:{i // 2:02d}" for i in range(60)]
    first = np.tile(np.arange(20), 3)
    second = (first + 1) % 20
    result = player_aware_interval(
        np.linspace(-0.1, 0.1, 60),
        events,
        first,
        second,
        date_block_ids=date_blocks,
        draws=200,
    )
    assert result["status"] == "measured"
    assert result["cluster_count"] == 30
    assert result["event_count"] == 60
    assert result["date_block_count"] == 30
    assert result["resampling_unit"] == "date_block_and_both_player_ids"


def test_policy_value_refuses_zero_or_rare_exact_action_overlap():
    n = 600
    events = np.repeat(np.arange(60), 10)
    mu = np.full((n, 2), 0.5)
    target = np.tile([0.0, 1.0], (n, 1))
    zero = np.tile([1.0, 0.0], (n, 1))
    result = policy_value(np.zeros(n), np.zeros(n, dtype=int), target, mu, zero, event_ids=events)
    assert result["status"] == "not_identified"
    assert result["estimate"] is None
    rare = np.tile([0.995, 0.005], (n, 1))
    result = policy_value(np.zeros(n), np.zeros(n, dtype=int), target, mu, rare, event_ids=events)
    assert result["status"] == "not_identified"
    assert result["low_target_propensity_rows"] == n


def test_policy_value_accepts_deterministic_target_action_ids():
    n = 600
    result = policy_value(
        np.zeros(n), np.zeros(n, dtype=int), np.ones(n, dtype=int),
        np.full((n, 2), 0.5), np.tile([1.0, 0.0], (n, 1)),
        event_ids=np.repeat(np.arange(60), 10),
    )
    assert result["status"] == "not_identified"
    assert result["estimate"] is None


def test_policy_value_with_low_ess_is_not_identified():
    rng = np.random.default_rng(10)
    n = 600
    action = rng.choice(2, size=n, p=[0.9, 0.1])
    result = policy_value(
        np.zeros(n), action, np.tile([0.0, 1.0], (n, 1)), np.full((n, 2), 0.5),
        np.tile([0.9, 0.1], (n, 1)), event_ids=np.repeat(np.arange(60), 10),
    )
    assert result["status"] == "not_identified"
    assert result["reason"] == "insufficient_operational_overlap"
    assert result["importance_weight_ess"] < 200


def test_policy_value_recovers_known_randomised_logging_value():
    rng = np.random.default_rng(1312)
    n = 8000
    action = rng.integers(0, 2, n)
    outcome_mean = np.tile([0.2, 0.7], (n, 1))
    y = rng.binomial(1, outcome_mean[np.arange(n), action])
    result = policy_value(
        y,
        action,
        np.tile([0.0, 1.0], (n, 1)),
        outcome_mean,
        np.full((n, 2), 0.5),
        event_ids=np.repeat(np.arange(200), 40),
    )
    assert result["status"] == "identified"
    assert result["estimate"] == pytest.approx(0.7, abs=0.04)
    assert result["clipped_at_20_estimate"] == pytest.approx(result["estimate"])
    assert result["interval"][0] < 0.7 < result["interval"][1]


def test_policy_value_with_correct_outcome_model_recovers_value():
    # The outcome model is the conditional truth; logged residuals have zero mean.
    rng = np.random.default_rng(77)
    n = 6000
    action = rng.integers(0, 2, n)
    mu = np.tile([0.25, 0.65], (n, 1))
    y = rng.binomial(1, mu[np.arange(n), action])
    target = np.tile([0.0, 1.0], (n, 1))
    result = policy_value(y, action, target, mu, np.full((n, 2), 0.5), event_ids=np.repeat(np.arange(200), 30))
    assert result["status"] == "identified"
    assert result["estimate"] == pytest.approx(0.65, abs=0.025)


def test_policy_value_with_correct_propensity_recovers_value_under_wrong_mu():
    rng = np.random.default_rng(78)
    n = 8000
    action = rng.integers(0, 2, n)
    true_mean = np.tile([0.15, 0.75], (n, 1))
    y = rng.binomial(1, true_mean[np.arange(n), action])
    target = np.tile([0.0, 1.0], (n, 1))
    wrong_mu = np.full((n, 2), 0.5)
    result = policy_value(y, action, target, wrong_mu, np.full((n, 2), 0.5), event_ids=np.repeat(np.arange(200), 40))
    assert result["status"] == "identified"
    assert result["estimate"] == pytest.approx(0.75, abs=0.04)


def test_policy_value_rejects_bad_probabilities():
    with pytest.raises(ValueError):
        policy_value([0], [0], [[0.2, 0.2]], [[0.5, 0.5]], [[0.5, 0.5]], event_ids=["e"])
