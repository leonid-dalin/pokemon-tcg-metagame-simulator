import numpy as np
import pytest
from scipy import sparse

from src.ingestion.best60_research.simulation import run_simulation


def _paired_design(pairs=100):
    rng = np.random.default_rng(2)
    deck_a = rng.integers(0, 2, (pairs, 3)).astype(float)
    deck_b = rng.integers(0, 2, (pairs, 3)).astype(float)
    X = np.vstack([deck_a, deck_b])
    nuisance = np.ones((2 * pairs, 1))
    groups = np.where(np.arange(2 * pairs) % 2 == 0, "A", "B")
    pairing_ids = np.asarray([f"p{i}" for i in range(pairs)] * 2, dtype=object)
    event_ids = np.asarray([f"e{i // 10}" for i in range(pairs)] * 2, dtype=object)
    player_ids = np.asarray([f"a{i}" for i in range(pairs)] + [f"b{i}" for i in range(pairs)], dtype=object)
    return {
        "X": X,
        "nuisance": nuisance,
        "groups": groups,
        "pairing_ids": pairing_ids,
        "event_ids": event_ids,
        "player_ids": player_ids,
        "side": np.asarray([1.0] * pairs + [-1.0] * pairs),
        "sample_weight": np.full(2 * pairs, 0.5),
    }


def test_observed_design_simulation_is_pair_clustered_and_resumable(tmp_path):
    context = _paired_design()
    checkpoint = tmp_path / "simulation.json"
    first = run_simulation(
        context,
        replicates=2,
        effect=0.2,
        mode="package",
        base_seed=1312,
        checkpoint=checkpoint,
        max_seconds=0,
        rank_grid=(2,),
        interaction_penalties=(0.1,),
    )
    assert first["status"] == "incomplete"
    assert first["next_index"] == 0
    second = run_simulation(
        context,
        replicates=2,
        effect=0.2,
        mode="package",
        base_seed=1312,
        checkpoint=checkpoint,
        rank_grid=(2,),
        interaction_penalties=(0.1,),
    )
    assert second["status"] == "complete"
    assert len(second["rows"]) == 2
    assert [item["seed"] for item in second["rows"]] == [1312, 1313]
    for row in second["rows"]:
        assert row["status"] == "complete"
        assert row["train_pairings"] + row["selection_pairings"] + row["measurement_pairings"] == 100
        assert row["split_unit"] == "event_random"
        assert row["train_events"] + row["selection_events"] + row["measurement_events"] == 10


def test_checkpoint_rejects_changed_opponent_block_or_injection_inputs(tmp_path):
    context = _paired_design()
    context["opponent_ids"] = np.asarray([f"o{i}" for i in range(len(context["X"]))], dtype=object)
    context["block_ids"] = context["event_ids"].copy()
    context["feature_pair"] = (0, 1)
    checkpoint = tmp_path / "guard.json"
    run_simulation(context, replicates=1, checkpoint=checkpoint, max_seconds=0, rank_grid=(2,), interaction_penalties=(0.1,))
    context["block_ids"][0] = "changed-block"
    try:
        run_simulation(context, replicates=1, checkpoint=checkpoint, max_seconds=0, rank_grid=(2,), interaction_penalties=(0.1,))
    except ValueError as exc:
        assert "do not match" in str(exc)
    else:
        raise AssertionError("changed checkpoint design was accepted")


def test_mirrored_views_share_one_outcome_draw():
    from src.ingestion.best60_research.simulation import _pair_outcomes

    context = _paired_design(pairs=20)
    arrays = context | {"pairing_ids": context["pairing_ids"], "side": context["side"]}
    rng = np.random.default_rng(44)
    y = _pair_outcomes(arrays, np.zeros(40), rng)
    for first, second in zip(y[:20], y[20:]):
        assert first + second == 1.0


def test_real_context_side_codes_are_mapped_to_opposite_orientations():
    from src.ingestion.best60_research.simulation import _pair_arrays, _pair_outcomes

    key = ("event-1", 1, 0, "match-1", "a", "b")
    context = {
        "slot_names": [("Card A", 1), ("Card B", 1)],
        "eligible_pairings": [{"key": key, "player1": "a", "player2": "b"}],
        "eligible_views": [
            {"pairing_key": key, "event": "event-1", "player": "a", "opponent": "b", "player1": "a", "player2": "b", "side": 1,
             "archetype_id": "A", "opponent_archetype_id": "B", "counts": {"Card A": 1, "Card B": 1}, "won": 1, "weight": 0.5},
            {"pairing_key": key, "event": "event-1", "player": "b", "opponent": "a", "player1": "a", "player2": "b", "side": 2,
             "archetype_id": "B", "opponent_archetype_id": "A", "counts": {"Card A": 1}, "won": 0, "weight": 0.5},
        ],
    }
    arrays = _pair_arrays(context)
    assert arrays["side"].tolist() == [1.0, -1.0]
    assert arrays["sample_weight"].tolist() == [0.5, 0.5]
    y = _pair_outcomes(arrays, np.array([0.0, 0.0]), np.random.default_rng(5))
    assert y[0] + y[1] == 1.0


def test_single_focal_view_weight_is_renormalised_to_one():
    from src.ingestion.best60_research.simulation import _pair_arrays

    key = ("event-1", 1, 0, "match-1", "a", "b")
    context = {
        "slot_names": [("Card A", 1)],
        "eligible_pairings": [{"key": key, "player1": "a", "player2": "b"}],
        "eligible_views": [
            {"pairing_key": key, "event": "event-1", "player": "a", "opponent": "b", "player1": "a", "player2": "b", "side": 1,
             "archetype_id": "A", "opponent_archetype_id": "B", "counts": {"Card A": 1}, "won": 1, "weight": 0.5},
        ],
    }
    assert _pair_arrays(context)["sample_weight"].tolist() == [1.0]


def test_package_pair_selection_requires_training_residual_information():
    from src.ingestion.best60_research.simulation import _select_supported_package

    X = np.tile(np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float), (25, 1))
    pair, info = _select_supported_package(
        X,
        np.column_stack([np.ones(len(X)), X]),
        np.ones(len(X)),
        np.repeat(np.arange(10), 10),
        ["Card A", "Card B"],
    )
    assert pair == (0, 1)
    assert info["status"] == "identified"
    assert info["information"] == pytest.approx(len(X) / 64)

    aliased = np.column_stack([X[:, 0], X[:, 0] * X[:, 1]])
    missing, missing_info = _select_supported_package(
        aliased,
        np.column_stack([np.ones(len(X)), aliased]),
        np.ones(len(X)),
        np.repeat(np.arange(10), 10),
        ["Card A", "Card B"],
    )
    assert missing is None
    assert missing_info["status"] == "not_identified"


def test_nonconverged_grid_fit_makes_simulation_inconclusive():
    result = run_simulation(
        _paired_design(),
        replicates=1,
        effect=0.2,
        mode="package",
        rank_grid=(2,),
        interaction_penalties=(0.1,),
        seed_grid=(1312,),
        max_iter=1,
    )
    assert result["status"] == "inconclusive"
    assert result["failed_replicates"] == 1
    assert result["rows"][0]["status"] == "fit_failure"
    assert result["rows"][0]["failed_fit_details"]


def test_prepared_real_context_keeps_sparse_nuisance_and_runs_simulation(tmp_path):
    from datetime import datetime, timedelta, timezone

    from src.ingestion.best60_research.experiment import prepare_simulation_context

    views = []
    for event_index in range(40):
        event = f"event-{event_index:03d}"
        day = datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(days=7 * event_index)
        for match in range(6):
            card_a, card_b = (match % 2), ((match // 2) % 2)
            key = (event, 1, 1, str(match), f"p-{event}-{match}", f"o-{event}-{match}")
            views.append({
                "pairing_key": key,
                "event": event,
                "date": day.isoformat(),
                "player": f"p-{event}-{match}",
                "opponent": f"o-{event}-{match}",
                "archetype_id": "target",
                "opponent_archetype_id": "other",
                "counts": {"Card A": card_a, "Card B": card_b},
                "won": float((event_index + match) % 2),
                "side": 1,
                "weight": 1.0,
                "game": "PTCG",
                "format": "standard",
                "delivery_mode": "online",
                "match_mode": "best-of-three",
                "is_online": True,
            })
    context = {
        "eligible_views": views,
        "slot_names": [("Card A", 1), ("Card B", 1)],
        "strength": {},
        "strength_at_cutoff": {},
        "event_block_ids": {
            event: f"utc-date:{(datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(days=7 * index)).date()}"
            for index, event in enumerate(sorted({row["event"] for row in views}))
        },
    }
    prepared = prepare_simulation_context(context, archetype_id="target")
    assert sparse.issparse(prepared["nuisance"])
    result = run_simulation(
        prepared,
        replicates=1,
        effect=0.0,
        mode="package",
        checkpoint=tmp_path / "prepared-context.json",
        rank_grid=(1,),
        interaction_penalties=(0.1,),
        seed_grid=(1312,),
        max_iter=100,
    )
    assert result["status"] in {"complete", "inconclusive", "not_identified"}
    assert result["next_index"] == 1
    assert len(result["rows"]) == 1
    assert result["rows"][0]["status"] in {"complete", "fit_failure", "not_identified"}
