from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from src.ingestion.best60_research import experiment
from src.ingestion.best60_research.experiment import _evaluate_benchmark


def _views(archetype="kangaskhan-bouffalant"):
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows = []
    for day in range(20):
        when = start + timedelta(days=day)
        for block in range(2):
            event = f"e{day}-{block}"
            for index in range(2):
                rows.append({
                    "event": event, "player": f"p{day}-{block}-{index}", "opponent": f"o{index}",
                    "side": 1, "won": index % 2, "counts": {"A": 4, "B": 56},
                    "archetype_id": archetype, "pairing_key": (event, index),
                    "date": when.isoformat(),
                    "date_block_id": f"event-block:{when.date().isoformat()}:{event}",
                    "player1": f"p{day}-{block}-{index}", "player2": f"o{index}",
                })
    return rows


class _Model:
    converged = True

    def predict_logit(self, X, *, nuisance, groups=None):
        return np.zeros(X.shape[0], dtype=float)


def test_day_unit_sensitivity_counts_days_while_the_primary_counts_blocks(monkeypatch, tmp_path):
    views = _views()
    context = {"eligible_views": views, "slot_names": [("A", 1)], "strength": {}, "strength_at_cutoff": {}}
    schema = {"categories": list(experiment._CATEGORIES),
              "vocabulary": {field: ["unknown"] for field in experiment._CATEGORIES}}
    monkeypatch.setattr(experiment, "_evaluation_model", lambda *args, **kwargs: (_Model(), schema, False))
    monkeypatch.setattr(experiment, "_model_for_config", lambda *args, **kwargs: (_Model(), schema))
    validation = {"benchmarks": {"pooled": {
        "status": "complete",
        "selected_models": {"primary_additive": {"prior_sd": 0.1, "kind": "additive"},
                            "best_interaction": {"kind": "fm", "rank": 2, "prior_sd": 0.1}},
    }}}
    protocol = {"h1": {"minimum_gain_nats_per_pairing": 0.0001, "minimum_independent_event_clusters": 20},
                "inference": {"seed": 1312, "h1_bootstrap_draws": 50},
                "estimators": {"max_iterations": 5},
                "selection": {"field_source": "observed"}}

    result = _evaluate_benchmark(context, views, validation, protocol, "pooled",
                                 archetype_id="kangaskhan-bouffalant", field_override=None,
                                 max_seconds=None, output_path=tmp_path / "eval.json")

    assert result["h1_predictive"]["cluster_count"] == 40
    assert result["h1_day_unit_sensitivity"]["cluster_count"] == 20
    assert set(result["h1_day_unit_sensitivity"]) == {"status", "estimate", "interval", "cluster_count"}


def test_ranking_interval_labels_event_blocks(monkeypatch):
    monkeypatch.setattr(experiment, "ranking_agreement", lambda *args, **kwargs: {"weighted_agreement": 0.6})
    data = {
        "events": [f"event-block:2026-01-{day:02d}:e{day}" for day in range(1, 21)],
        "model": [0.0] * 20, "baseline": [0.0] * 20, "proxy": [0.0] * 20,
        "matches": [1] * 20, "archetypes": ["a"] * 20,
    }

    result = experiment._ranking_gain_interval(data, min_matches=1, seed=1312, draws=5)

    assert result["resampling_unit"] == "atomic_event_block"
