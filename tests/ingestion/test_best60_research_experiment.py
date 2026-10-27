from __future__ import annotations

import copy
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from src.ingestion.best60_research import experiment


def _views(prefix: str, start: int, events: int, *, pairs_per_event: int = 6):
    rows = []
    training_lists = []
    rng = np.random.default_rng(start + 100)
    for event_offset in range(events):
        event_index = start + event_offset
        event_id = f"{prefix}{event_index}"
        day = datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(days=7 * event_index)
        date_value = day.isoformat()
        block = f"utc-date:{day.date().isoformat()}"
        for match_index in range(pairs_per_event):
            card_a = int(match_index in {0, 1, 3})
            card_b = int(match_index in {0, 2, 3})
            counts = {"Card A": card_a, "Card B": card_b, "Grass Energy": 58 + 1 - card_a - card_b, "Core": 1}
            player1 = f"p-{event_id}-{match_index}"
            player2 = f"o-{event_id}-{match_index}"
            key = (event_id, 1, 1, str(match_index), player1, player2)
            won = int(rng.binomial(1, 0.25 + 0.5 * card_a))
            base = {
                "pairing_key": key,
                "event": event_id,
                "round": 1,
                "date": date_value,
                "date_block_id": block,
                "source": "official",
                "platform": "live",
                "is_online": True,
                "delivery_mode": "online",
                "match_mode": "best-of-three",
                "game": "PTCG",
                "format": "standard",
                "player1": player1,
                "player2": player2,
                "counts": counts,
                "weight": 0.5,
            }
            rows.append({
                **base, "player": player1, "opponent": player2,
                "archetype_id": "Kanga", "opponent_archetype_id": "Other",
                "won": won, "side": 1,
            })
            rows.append({
                **base, "player": player2, "opponent": player1,
                "archetype_id": "Other", "opponent_archetype_id": "Kanga",
                "won": 1 - won, "side": 2,
            })
        for archetype, player_prefix in (("Kanga", "p"), ("Other", "o")):
            for match_index in range(pairs_per_event):
                if archetype == "Kanga":
                    counts = {"Card A": int(match_index in {0, 1, 3}), "Card B": int(match_index in {0, 2, 3}),
                              "Grass Energy": 58 + 1 - int(match_index in {0, 1, 3}) - int(match_index in {0, 2, 3}), "Core": 1}
                else:
                    counts = {"Card A": 1, "Card B": 0, "Grass Energy": 59, "Core": 1}
                training_lists.append({
                    "event": event_id, "player": f"{player_prefix}-{event_id}-{match_index}",
                    "archetype_id": archetype, "date": date_value, "players": 32,
                    "placing": match_index + 1, "wins": 2, "losses": 1,
                    "counts": counts, "groups": {},
                })
    return rows, training_lists


def _context():
    train_rows, training_lists = _views("d", 0, 40)
    validation_rows, _ = _views("v", 40, 12)
    test_rows, _ = _views("t", 52, 24)
    training_events = sorted({row["event"] for row in train_rows})
    dates = {row["event"]: row["date"][:10] for row in train_rows}
    block_ids = {event: f"utc-date:{dates[event]}" for event in training_events}
    strength = {}
    strength_at_cutoff = {}
    for row in train_rows + validation_rows + test_rows:
        strength[(row["event"], row["player"])] = 0.05
        strength[(row["event"], row["opponent"])] = -0.05
        strength_at_cutoff[row["player"]] = 0.02
        strength_at_cutoff[row["opponent"]] = -0.02
    legal_a = {"Card A": 1, "Grass Energy": 58, "Core": 1}
    legal_b = {"Card B": 1, "Grass Energy": 58, "Core": 1}
    catalogue = [
        {"counts": legal_a, "list_count": 5, "events": ["d0"], "archetype_ids": ["Kanga"],
         "companions": {"Card A": {}, "Grass Energy": {}, "Core": {}}},
        {"counts": legal_b, "list_count": 4, "events": ["d1"], "archetype_ids": ["Kanga"],
         "companions": {"Card B": {}, "Grass Energy": {}, "Core": {}}},
    ]
    context = {
        "training_event_ids": training_events,
        "cutoff": "2024-12-31T00:00:00+00:00",
        "eligible_views": train_rows,
        "training_lists": training_lists,
        "slot_names": [("Card A", 1), ("Card B", 1)],
        "strength": strength,
        "strength_at_cutoff": strength_at_cutoff,
        "event_block_ids": block_ids,
        "field": {"Other": 0.7, "Kanga": 0.3},
        "catalogue": catalogue,
        "companions": {},
        "target_archetype_id": "Kanga",
    }
    protocol = {
        "features": {"slot_minimum_lists": 1},
        "estimators": {
            "rolling_folds": 1,
            "max_iterations": 5000,
            "max_iterations_by_model": {"nuisance": 5000, "per_archetype_additive": 5000,
                                        "pooled_additive": 5000, "regularised_selected_pairs": 5000,
                                        "pooled_factorisation_machine": 5000},
            "fm_ranks": [2], "fm_seeds": [1312], "interaction_penalties": [0.001],
        },
        "simulation": {"quick_replicates_per_effect": 0, "quick_rank": 2, "quick_seed": 1312, "quick_penalty": 0.001},
        "h1": {"minimum_gain_nats_per_pairing": 0.0001,
               "ranking_minimum_decided_results_per_list": 5},
        "h3": {"minimum_event_clusters": 50, "minimum_importance_weight_ess": 5000},
        "h4": {"bootstrap_draws": 3, "mean_interval_width_target": 0.1},
        "inference": {"seed": 1312, "h1_bootstrap_draws": 100},
    }
    return context, validation_rows, test_rows, protocol


def test_design_uses_event_pre_event_strength_for_outer_training_and_frozen_holdout():
    context, rows, _, _ = _context()
    row = dict(rows[0])
    context["strength"][(row["event"], row["player"])] = 0.4
    context["strength"][(row["event"], row["opponent"])] = -0.2
    context["strength_at_cutoff"][row["player"]] = 90.0
    context["strength_at_cutoff"][row["opponent"]] = -90.0
    _, _, nuisance, _, schema = experiment._design([row], context, training=True)
    assert nuisance[0, 1] == pytest.approx(0.6)
    _, _, holdout_nuisance, _, _ = experiment._design([row], context, training=False, schema=schema)
    assert holdout_nuisance[0, 1] == pytest.approx(180.0)


def test_inner_validation_strength_is_frozen_before_first_validation_date_and_embargo():
    context, validation_rows, _, _ = _context()
    context["embargo_days"] = 7
    context["training_event_dates"] = {
        "jan-01": "2024-01-01", "jan-02": "2024-01-02",
        "jan-10": "2024-01-10", "jan-20": "2024-01-20",
    }
    context["eligible_records"] = [
        {"event": "jan-01", "player": "shared", "wins": 1, "losses": 1},
        {"event": "jan-02", "player": "shared", "wins": 2, "losses": 0},
        # These January 10 outcomes are outside the inner fit and must not
        # influence any validation row, including a later January 20 row.
        {"event": "jan-10", "player": "shared", "wins": 500, "losses": 0},
        # A different archetype is represented by the record too; history is
        # pooled across archetypes when it is genuinely before the boundary.
        {"event": "jan-02", "player": "other-archetype-player", "wins": 3, "losses": 1},
    ]
    inner_rows = [
        {**validation_rows[0], "event": "jan-10", "date": "2024-01-10T00:00:00Z"},
        {**validation_rows[1], "event": "jan-20", "date": "2024-01-20T00:00:00Z"},
    ]
    first_context, boundary = experiment._inner_validation_context(
        context, inner_rows, embargo_days=7
    )
    assert boundary == "2024-01-03"
    assert "other-archetype-player" in first_context["strength_at_cutoff"]
    first_strength = first_context["strength_at_cutoff"]["shared"]
    expected = np.log((3 + 10) / (1 + 10))
    assert first_strength == pytest.approx(expected)

    poisoned = copy.deepcopy(context)
    poisoned["eligible_records"][2]["wins"] = 500_000
    second_context, _ = experiment._inner_validation_context(
        poisoned, inner_rows, embargo_days=7
    )
    assert second_context["strength_at_cutoff"]["shared"] == pytest.approx(first_strength)

    _, _, _, _, schema = experiment._design(context["eligible_views"], context, training=True)
    _, _, first_nuisance, _, _ = experiment._design(
        [inner_rows[1]], first_context, training=False, schema=schema
    )
    _, _, second_nuisance, _, _ = experiment._design(
        [inner_rows[1]], second_context, training=False, schema=schema
    )
    assert first_nuisance[0, 1] == pytest.approx(second_nuisance[0, 1])


def test_focal_view_weights_renormalise_within_pairing():
    rows, _ = _views("e", 0, 1, pairs_per_event=1)
    one_side = [row for row in rows if row["archetype_id"] == "Kanga"]
    np.testing.assert_allclose(experiment._view_weights(rows, focal=False), [0.5, 0.5])
    np.testing.assert_allclose(experiment._view_weights(one_side, focal=True), [1.0])


def test_simulation_context_preserves_pair_ids_player_blocks_and_nonalised_pair():
    context, _, _, _ = _context()
    prepared = experiment.prepare_simulation_context(context)
    assert prepared["X"].shape[0] == len(context["eligible_views"])
    assert len(prepared["pairing_ids"]) == len(context["eligible_views"])
    assert len(prepared["list_ids"]) == len(context["eligible_views"])
    assert prepared["synthetic_design"] is False
    assert prepared["feature_names"] == ["Card A", "Card B"]
    assert prepared["feature_pair"] is not None
    focal = experiment.prepare_simulation_context(context, archetype_id="Kanga")
    assert focal["simulation_scope"] == "Kanga"
    assert len(focal["X"].toarray()) == sum(row["archetype_id"] == "Kanga" for row in context["eligible_views"])


def test_rank_lists_uses_actual_list_match_counts_and_fixed_score_callback():
    rows = []
    for player_index in range(3):
        for match_index in range(5):
            rows.append({
                "event": "event-1", "date_block_id": "utc-date:2024-01-01",
                "player": f"player-{player_index}", "archetype_id": "Kanga",
                "counts": {"Card A": player_index}, "won": player_index / 2,
                "pairing_key": ("event-1", match_index, player_index),
            })
    nuisance_logits = np.zeros(len(rows))
    weights = np.full(len(rows), 0.5)
    scored = experiment._rank_lists(
        rows, nuisance_logits, weights,
        lambda counts, row: (float(counts["Card A"]), float(counts["Card A"]) / 2),
    )
    assert scored["matches"] == [5, 5, 5]
    assert scored["model"] == [0.0, 1.0, 2.0]
    # The heldout proxy is the outcome residual Y - p(nuisance), not Y itself.
    assert scored["proxy"] == pytest.approx([-0.5, 0.0, 0.5])


def test_validation_checkpoint_freezes_models_and_evaluation_does_not_retune(tmp_path, monkeypatch):
    context, validation_rows, test_rows, protocol = _context()
    # Exercise the real production iteration allowance so the pooled FM is not
    # spuriously marked failed by this deliberately tiny fixture budget.
    protocol["estimators"]["max_iterations"] = 5000
    protocol["estimators"]["max_iterations_by_model"] = {
        key: 5000 for key in protocol["estimators"]["max_iterations_by_model"]
    }
    checkpoint = tmp_path / "validation-checkpoint.json"
    validation = experiment.validate_experiment(
        context, validation_rows, protocol, archetype_id="Kanga", checkpoint_path=checkpoint, max_seconds=90,
    )
    assert validation["status"] == "complete"
    assert validation["test_outcomes_read"] is False
    assert validation["benchmarks"]["pooled"]["rolling_development_folds"]
    assert validation["benchmarks"]["pooled"]["selected_models"]["best_interaction"]
    first_checkpoint = checkpoint.read_bytes()
    resumed = experiment.validate_experiment(
        context, validation_rows, protocol, archetype_id="Kanga", checkpoint_path=checkpoint, max_seconds=90,
    )
    assert resumed["benchmarks"]["pooled"]["selected_models"] == validation["benchmarks"]["pooled"]["selected_models"]
    assert checkpoint.read_bytes() == first_checkpoint

    calls = []
    original_evaluation_model = experiment._evaluation_model

    def tracked(*args, **kwargs):
        calls.append(dict(args[2]))
        return original_evaluation_model(*args, **kwargs)

    monkeypatch.setattr(experiment, "_evaluation_model", tracked)
    result = experiment.evaluate_experiment(
        context, test_rows, validation, protocol, archetype_id="Kanga",
        field_override={"Other": 1.0}, output_path=tmp_path / "evaluation-progress.json", max_seconds=90,
    )
    assert result["test_outcomes_read"] is True
    assert result["status"] in {"complete", "incomplete"}
    assert result["benchmarks"]["pooled"]["h3_policy_value"]["status"] == "not_identified"
    assert result["benchmarks"]["pooled"]["h3_policy_value"]["row_count"] == len(test_rows)
    assert result["benchmarks"]["pooled"]["h3_policy_value"]["missing_inputs"] == [
        "cross_fitted_outcome_model", "exact_list_logging_propensities",
    ]
    assert result["benchmarks"]["pooled"]["h1_balanced_unweighted_diagnostic"]["archetype_count"] == 2
    assert result["benchmarks"]["pooled"]["h4_predictive_coverage"]["status"] == "inconclusive"
    coverage = result["benchmarks"]["pooled"]["h4_predictive_coverage"]
    assert coverage["status"] in {"complete", "inconclusive"}
    assert coverage["latent_mean_coverage"] is None or 0.0 <= coverage["latent_mean_coverage"] <= 1.0
    assert calls
    assert len(calls) >= 4
    assert calls[:2] == [
        validation["benchmarks"]["pooled"]["selected_models"]["primary_additive"],
        validation["benchmarks"]["pooled"]["selected_models"]["best_interaction"],
    ]


@pytest.mark.parametrize("failed_role", ["primary_additive", "best_interaction", "nuisance"])
def test_nonconverged_frozen_refit_is_inconclusive_not_evidence(tmp_path, monkeypatch, failed_role):
    context, _, test_rows, protocol = _context()
    selected = {
        "primary_additive": {"kind": "additive", "group_scope": "pooled", "penalty": 0.01},
        "best_interaction": {"kind": "fm", "group_scope": "pooled", "rank": 2, "seed": 1312, "penalty": 0.01},
    }
    validation = {
        "status": "complete",
        "benchmarks": {"pooled": {"status": "complete", "selected_models": selected}},
    }
    _, _, _, _, schema = experiment._design(context["eligible_views"], context, training=True)

    class DeterministicModel:
        def __init__(self, converged):
            self.converged = converged

        def predict_logit(self, X, *, nuisance, groups):
            return np.zeros(X.shape[0], dtype=float)

    def evaluation_refit(rows, context_arg, config, **kwargs):
        role = kwargs["role"]
        return DeterministicModel(role != failed_role), schema, False

    monkeypatch.setattr(experiment, "_evaluation_model", evaluation_refit)
    original_model_for_config = experiment._model_for_config

    def nuisance_refit(rows, context_arg, config, *args, **kwargs):
        if config.get("kind") == "nuisance":
            return DeterministicModel(failed_role != "nuisance"), schema
        return original_model_for_config(rows, context_arg, config, *args, **kwargs)

    monkeypatch.setattr(experiment, "_model_for_config", nuisance_refit)
    result = experiment.evaluate_experiment(
        context, test_rows, validation, protocol, archetype_id="Kanga",
        field_override=None, output_path=tmp_path / f"evaluation-{failed_role}.json", max_seconds=30,
    )

    benchmark_result = result["benchmarks"]["pooled"]
    assert result["status"] == "inconclusive"
    assert benchmark_result["status"] == "inconclusive"
    assert benchmark_result["reason"] == "frozen_model_refit_did_not_converge"
    assert benchmark_result["failed_model_role"] == ("nuisance" if failed_role == "nuisance" else failed_role)
    assert benchmark_result["converged"] is False
    assert "test_log_loss" not in benchmark_result


def test_final_interaction_main_penalty_matches_additive_at_refit_size_without_mutation(tmp_path, monkeypatch):
    context, _, test_rows, protocol = _context()
    train_rows = context["eligible_views"]
    validation_interaction = {
        "kind": "fm", "group_scope": "pooled", "rank": 2, "seed": 1312,
        "penalty": 0.003, "main_effect_penalty": 0.75,
    }
    ref_config = {
        "kind": "additive", "group_scope": "pooled", "prior_sd": 0.4,
        "penalty": 0.0,
    }
    validation = {
        "benchmarks": {"pooled": {
            "status": "complete",
            "selected_models": {
                "primary_additive": dict(ref_config),
                "best_interaction": dict(validation_interaction),
            },
        }},
    }
    locked_choices = copy.deepcopy(validation)
    _, _, _, _, schema = experiment._design(train_rows, context, training=True)
    captured_configs = []

    class DeterministicModel:
        converged = True

        def predict_logit(self, X, *, nuisance, groups):
            return np.zeros(X.shape[0], dtype=float)

    def evaluation_refit(rows, context_arg, config, **kwargs):
        captured_configs.append(dict(config))
        return DeterministicModel(), schema, False

    def model_for_config(rows, context_arg, config, *args, **kwargs):
        return DeterministicModel(), schema

    monkeypatch.setattr(experiment, "_evaluation_model", evaluation_refit)
    monkeypatch.setattr(experiment, "_model_for_config", model_for_config)
    result = experiment._evaluate_benchmark(
        context, test_rows, validation, protocol, "pooled", archetype_id="Kanga",
        field_override=None, max_seconds=30, output_path=tmp_path / "penalty-eval.json",
    )

    expected_main_penalty = experiment._effective_penalty(
        ref_config, experiment._view_weights(train_rows, focal=False)
    )
    assert len(captured_configs) == 2
    assert captured_configs[0] == ref_config
    assert captured_configs[1]["main_effect_penalty"] == pytest.approx(expected_main_penalty)
    assert captured_configs[1]["main_effect_penalty"] != validation_interaction["main_effect_penalty"]
    assert {key: captured_configs[1][key] for key in ("penalty", "rank", "seed")} == {
        key: validation_interaction[key] for key in ("penalty", "rank", "seed")
    }
    assert result["best_interaction"]["config"] == captured_configs[1]
    assert result["best_interaction"]["validation_config"] == validation_interaction
    assert validation == locked_choices


def test_evaluation_refit_cache_reloads_npz_without_refitting(tmp_path):
    context, _, _, protocol = _context()
    train_rows = context["eligible_views"]
    config = {"kind": "nuisance", "group_scope": "pooled", "penalty": 0.0}
    first_model, first_schema, cache_hit = experiment._evaluation_model(
        train_rows, context, config, benchmark="pooled", role="primary_additive",
        cache_dir=tmp_path / "model-cache", max_iterations=200, max_seconds=20,
    )
    assert cache_hit is False
    cache_token = experiment.hashlib.sha256(b"pooled\0primary_additive").hexdigest()[:20]
    cache_path = tmp_path / "model-cache" / f"{cache_token}.npz"
    assert cache_path.is_file()
    second_model, second_schema, cache_hit = experiment._evaluation_model(
        train_rows, context, config, benchmark="pooled", role="primary_additive",
        cache_dir=tmp_path / "model-cache", max_iterations=200, max_seconds=0,
    )
    assert cache_hit is True
    assert second_schema == first_schema
    np.testing.assert_allclose(second_model.nuisance_coef, first_model.nuisance_coef)


def test_field_override_changes_recommendation_score_and_exact_choice():
    context, _, _, _ = _context()
    # A model with opponent-specific card effects exercises the field mixture.
    candidates = context["catalogue"]
    _, _, _, _, schema = experiment._design(context["eligible_views"], context, training=True)
    levels = schema["vocabulary"]["opponent_archetype_id"]
    other_column = None
    column = 2  # intercept and skill gap
    for category in experiment._CATEGORIES:
        for level in schema["vocabulary"][category][1:]:
            if category == "opponent_archetype_id" and level == "Other":
                other_column = column
            column += 1
    assert other_column is not None

    class OpponentSpecificModel:
        converged = True

        def predict_logit(self, X, *, nuisance, groups):
            features = X.toarray()
            controls = nuisance.toarray()
            other = controls[:, other_column]
            # Card A helps against Other; Card B helps against Kanga.
            return 4.0 * (features[:, 0] * other + features[:, 1] * (1.0 - other))

    model = OpponentSpecificModel()
    result_a = experiment.select_catalogue(
        candidates,
        score=lambda counts: experiment._score_count_mapping(
            counts, model, schema, context, "Kanga", {"Other": 1, "Kanga": 0}, "pooled"
        ),
    )
    result_b = experiment.select_catalogue(
        candidates,
        score=lambda counts: experiment._score_count_mapping(
            counts, model, schema, context, "Kanga", {"Other": 0, "Kanga": 1}, "pooled"
        ),
    )
    assert result_a["counts"] != result_b["counts"]


def test_fixed_list_refits_keep_the_frozen_nuisance_schema(monkeypatch, tmp_path):
    context, _, _, protocol = _context()
    context["eligible_views"] = copy.deepcopy(context["eligible_views"])
    context["eligible_views"] = [
        row for row in context["eligible_views"] if row["archetype_id"] == "Kanga"
    ]
    context["training_event_ids"] = sorted({row["event"] for row in context["eligible_views"]})
    # This category occurs in training, but a bootstrap sample may omit it.
    context["eligible_views"][0]["platform"] = "rare-platform"
    _, _, _, _, frozen_schema = experiment._design(
        context["eligible_views"], context, training=True
    )
    schema_inputs = []
    rare_category_in_sample = []

    class Converged:
        converged = True

    def fake_fit(rows, context_arg, config, max_iterations, max_seconds, schema=None, focal=False):
        schema_inputs.append(schema)
        rare_category_in_sample.append(any(row["platform"] == "rare-platform" for row in rows))
        assert {row["archetype_id"] for row in rows} == {"Kanga"}
        assert schema is frozen_schema
        return Converged(), schema

    monkeypatch.setattr(experiment, "_model_for_config", fake_fit)
    monkeypatch.setattr(experiment, "_score_count_mapping", lambda *args, **kwargs: 0.61)
    result = experiment._fixed_interval(
        context,
        {"Card A": 1, "Grass Energy": 58, "Core": 1},
        {"kind": "additive", "prior_sd": 0.1, "group_scope": "pooled"},
        {"Other": 0.5, "Kanga": 0.5},
        "Kanga",
        frozen_schema,
        protocol,
        max_seconds=30,
        checkpoint_path=tmp_path / "h4-checkpoint.json",
        benchmark="Kanga",
    )
    assert len(schema_inputs) == protocol["h4"]["bootstrap_draws"]
    assert not all(rare_category_in_sample)
    assert result["scope"] == "model_conditional"
    assert result["status"] == "inconclusive"  # Identical synthetic scores have zero width.
    assert result["checkpoint_path"] == str(tmp_path / "h4-checkpoint.json")
    assert (tmp_path / "h4-checkpoint.json").is_file()
    resumed = experiment._fixed_interval(
        context,
        {"Card A": 1, "Grass Energy": 58, "Core": 1},
        {"kind": "additive", "prior_sd": 0.1, "group_scope": "pooled"},
        {"Other": 0.5, "Kanga": 0.5},
        "Kanga",
        frozen_schema,
        protocol,
        max_seconds=30,
        checkpoint_path=tmp_path / "h4-checkpoint.json",
        benchmark="Kanga",
    )
    assert len(schema_inputs) == protocol["h4"]["bootstrap_draws"]
    assert resumed["next_index"] == protocol["h4"]["bootstrap_draws"]


def test_policy_value_missing_nuisances_is_an_explicit_nonidentification():
    context, _, test_rows, _ = _context()
    action_ids = np.arange(len(test_rows), dtype=int) % 3
    target_actions = np.zeros(len(test_rows), dtype=int)
    result = experiment.policy_value(
        [row["won"] for row in test_rows], action_ids, target_actions, None, None,
        event_ids=[row["date_block_id"] for row in test_rows],
    )
    assert result["status"] == "not_identified"
    assert result["estimate"] is None
    assert set(result["missing_inputs"]) == {
        "cross_fitted_outcome_model", "exact_list_logging_propensities",
    }


def test_balanced_h1_view_weights_count_each_pair_once_with_mixed_orientations():
    summary = experiment.balanced_h1_summary(
        [0.2, 0.4, -0.6],
        ["block-1", "block-1", "block-2"],
        ["Kanga", "Other", "Kanga"],
        sample_weight=[0.5, 0.5, 1.0],
    )
    # Pair one has two orientations averaging 0.3; pair two has one view at -0.6.
    assert summary["pairing_weighted"] == pytest.approx(-0.15)
    assert summary["event_balanced"] == pytest.approx(-0.15)
    assert summary["archetype_balanced"] == pytest.approx(((-0.5 / 1.5) + 0.4) / 2)
    assert summary["view_count"] == 3
    assert summary["unit"].startswith("test_view")


def test_candidate_timeout_is_checkpointed_but_global_timeout_stays_pending(tmp_path, monkeypatch):
    budgets = []

    def timed_out(*args, **kwargs):
        budgets.append(kwargs["max_seconds"])
        raise experiment.FitBudgetExceeded("synthetic optimizer timeout")

    monkeypatch.setattr(experiment, "_score_config", timed_out)
    empty = np.zeros((1, 1))
    y = np.asarray([0.0])
    weights = np.asarray([1.0])
    labels = np.asarray(["Kanga"], dtype=object)
    checkpoint = {"schema_version": 1, "fit_records": {}, "pilot": None}
    config = {"kind": "nuisance", "group_scope": "pooled", "penalty": 0.0}
    candidate_record = experiment._cached_score(
        "candidate-limited", empty, y, empty, labels, weights,
        empty, y, empty, labels, weights, config,
        checkpoint=checkpoint, checkpoint_path=tmp_path / "candidate.json",
        max_iterations=10, max_seconds=10.0, candidate_fit_seconds=0.5,
    )
    assert candidate_record["status"] == "fit_timeout"
    assert candidate_record["timeout_scope"] == "candidate_budget"
    assert "candidate-limited" in checkpoint["fit_records"]
    assert budgets[-1] == 0.5

    stage_record = experiment._cached_score(
        "stage-limited", empty, y, empty, labels, weights,
        empty, y, empty, labels, weights, config,
        checkpoint=checkpoint, checkpoint_path=tmp_path / "stage.json",
        max_iterations=10, max_seconds=0.5, candidate_fit_seconds=10.0,
    )
    assert stage_record["status"] == "stage_timeout"
    assert stage_record["timeout_scope"] == "stage_budget"
    assert "stage-limited" not in checkpoint["fit_records"]
    assert budgets[-1] == 0.5


def test_benchmark_order_defaults_focal_first_and_respects_protocol_order():
    benchmark_rows = {"pooled": ([], []), "Kanga": ([], [])}
    assert experiment._benchmark_order(benchmark_rows, "Kanga", {}) == ["Kanga", "pooled"]
    assert experiment._benchmark_order(
        benchmark_rows, "Kanga", {"features": {"focal_benchmarks": ["pooled", "Kanga"]}}
    ) == ["pooled", "Kanga"]


def test_timing_pilot_records_fit_budget_failure(monkeypatch):
    def timed_out(*args, **kwargs):
        assert kwargs["max_seconds"] == 0.1
        raise experiment.FitBudgetExceeded("pilot budget")

    monkeypatch.setattr(experiment, "fit_model", timed_out)
    result = experiment._pilot(
        np.zeros((2, 1)), np.array([0, 1]), np.ones((2, 1)),
        np.array(["g", "g"]), np.ones(2),
        max_iterations=5000, max_seconds=0.1, rank=2, seed=1312, penalty=0.01,
    )
    assert result["status"] == "fit_timeout"
    assert result["converged"] is False
    assert result["reason"] == "pilot budget"


@pytest.mark.parametrize("cohort_status,expected", [
    ("complete", "complete"), ("inconclusive", "inconclusive"),
    ("not_identified", "inconclusive"), ("incomplete", "incomplete"),
])
def test_evaluation_distinguishes_scientific_status_from_pending_work(
    monkeypatch, tmp_path, cohort_status, expected,
):
    context, _, test_rows, protocol = _context()
    protocol["features"]["focal_benchmarks"] = ["Kanga", "pooled"]
    seen = []

    def measured(*args, **kwargs):
        benchmark = args[4]
        seen.append(benchmark)
        return {"status": cohort_status if benchmark == "pooled" else "complete"}

    monkeypatch.setattr(experiment, "_evaluate_benchmark", measured)
    result = experiment.evaluate_experiment(
        context, test_rows, {"status": "complete"}, protocol,
        archetype_id="Kanga", field_override=None,
        output_path=tmp_path / "evaluation.json", max_seconds=10,
    )
    assert result["status"] == expected
    assert seen == ["Kanga", "pooled"]
