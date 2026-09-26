from __future__ import annotations

import numpy as np

from src.ingestion.model import PlayerObservation, fit_card_model


def _observations(seed: int, fits: int) -> list[PlayerObservation]:
    rng = np.random.default_rng(seed)
    rows = []
    for fit in range(fits):
        for player in range(120):
            has_tech = player % 2 == 0
            skill = rng.normal(0.0, 0.5)
            logit = -0.2 + (1.1 if has_tech else 0.0) + skill
            result = int(rng.random() < 1.0 / (1.0 + np.exp(-logit)))
            rows.append(PlayerObservation(
                "a", "b", frozenset({"Tech"} if has_tech else set()), frozenset(), result,
                (f"p{player}", f"q{player}"),
            ))
    return rows


def main() -> int:
    estimates = []
    reported = []
    standard_error_kinds = []
    for seed in range(40):
        model = fit_card_model(_observations(seed, 1), ["Tech"])
        estimates.append(model.estimator.coef_[0, 1])
        reported.append(model.standard_errors["Tech"])
        standard_error_kinds.append(model.standard_error_kind)
    true_se = float(np.std(estimates, ddof=1))
    reported_se = float(np.mean(reported))
    ratio = reported_se / true_se
    coverage = float(np.mean([abs(estimate - 1.1) <= 1.96 * error for estimate, error in zip(estimates, reported)]))
    print(f"fits={len(estimates)} true_se={true_se:.4f} reported_se={reported_se:.4f} ratio={ratio:.3f} coverage={coverage:.3f}")
    return int(not (
        all(kind == "player-clustered" for kind in standard_error_kinds)
        and 0.8 <= ratio <= 1.25
        and coverage >= 0.9
    ))


if __name__ == "__main__":
    raise SystemExit(main())
