"""Calibration gate for the card model's player-clustered standard errors.

Simulates Swiss-style events in which each player keeps one decklist and a persistent skill, fits the card
model on each, and compares the reported standard error with the spread of the estimate across events.
Exits 1 when any fit is not player-clustered, when the ratio leaves [0.85, 1.15], or when the 95% interval
covers the true zero effect less than 90% of the time.
"""
import argparse
import sys

import numpy as np

from src.ingestion.model import Z_95, CardModelNotIdentifiable, PlayerObservation, fit_card_model

DECK_EFFECT = {"alpha": 0.2, "beta": 0.0, "gamma": -0.2}


def simulate(rng: np.random.Generator, players: int, rounds: int, skill_sd: float) -> list[PlayerObservation]:
    decks = rng.choice(list(DECK_EFFECT), size=players)
    tech = rng.random(players) < 0.5
    skill = rng.normal(0.0, skill_sd, size=players)
    rows = []
    for _ in range(rounds):
        order = rng.permutation(players)
        for a, b in zip(order[::2], order[1::2]):
            logit = DECK_EFFECT[decks[a]] - DECK_EFFECT[decks[b]] + skill[a] - skill[b]
            rows.append(PlayerObservation(
                str(decks[a]), str(decks[b]),
                frozenset({"Filler", "Tech"} if tech[a] else {"Filler"}),
                frozenset({"Filler", "Tech"} if tech[b] else {"Filler"}),
                int(rng.random() < 1.0 / (1.0 + np.exp(-logit))),
                players=(f"player-{a}", f"player-{b}"),
            ))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--replicates", type=int, default=200)
    parser.add_argument("--skill-sd", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1312)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    estimates, reported, kinds = [], [], set()
    for _ in range(args.replicates):
        try:
            model = fit_card_model(simulate(rng, 400, 8, args.skill_sd), ["Tech"])
        except CardModelNotIdentifiable:
            continue
        coefficients, _ = model.coefficient_report()
        estimates.append(coefficients["Tech"])
        reported.append(model.standard_errors["Tech"])
        kinds.add(model.standard_error_kind)
    truth = float(np.std(estimates, ddof=1))
    ratio = float(np.mean(reported)) / truth
    coverage = float(np.mean([abs(e) <= Z_95 * s for e, s in zip(estimates, reported)]))
    print(f"fits {len(estimates)}  kinds {sorted(kinds)}  true SE {truth:.4f}  reported SE {np.mean(reported):.4f}  ratio {ratio:.2f}  coverage {coverage:.3f}")
    if kinds != {"player-clustered"} or not 0.85 <= ratio <= 1.15 or coverage < 0.9:
        print("FAIL: clustered standard errors disagree with the replicate spread")
        return 1
    print("OK: clustered standard errors match the replicate spread")
    return 0


if __name__ == "__main__":
    sys.exit(main())
