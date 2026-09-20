from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression

ACE_SPEC_CARDS = frozenset({
    "Amulet of Hope", "Awakening Drum", "Brilliant Blender", "Dangerous Laser",
    "Deluxe Bomb", "Energy Search Pro", "Enriching Energy", "Grand Tree",
    "Hero's Cape", "Hyper Aroma", "Legacy Energy", "Master Ball", "Max Rod",
    "Maximum Belt", "Megaton Blower", "Miracle Headset", "Neo Upper Energy",
    "Neutralization Zone", "Poke Vital A", "Precious Trolley", "Prime Catcher",
    "Reboot Pod", "Scoop Up Cyclone", "Scramble Switch", "Secret Box",
    "Sparkling Crystal", "Survival Brace", "Treasure Tracker", "Unfair Stamp",
})


@dataclass
class FittedCardModel:
    decks: list[str]
    cards: list[str]
    estimator: LogisticRegression
    inclusion: Mapping[str, Mapping[str, float]]

    def probability(self, deck_i: str, deck_j: str) -> float:
        row = np.zeros((1, len(self.decks) + len(self.cards)), dtype=float)
        if deck_i in self.decks:
            row[0, self.decks.index(deck_i)] = 1.0
        if deck_j in self.decks:
            row[0, self.decks.index(deck_j)] = -1.0
        offset = len(self.decks)
        for index, card in enumerate(self.cards):
            row[0, offset + index] = self.inclusion.get(deck_i, {}).get(card, 0.0) - self.inclusion.get(deck_j, {}).get(card, 0.0)
        return float(self.estimator.predict_proba(row)[0, 1])


def fit_model(
    observations: list[tuple[str, str, int]],
    inclusion: Mapping[str, Mapping[str, float]],
) -> FittedCardModel:
    decks = sorted({deck for row in observations for deck in row[:2]})
    cards = sorted({card for values in inclusion.values() for card in values})
    rows = []
    labels = []
    for deck_i, deck_j, result in observations:
        row = np.zeros(len(decks) + len(cards), dtype=float)
        row[decks.index(deck_i)] = 1.0
        row[decks.index(deck_j)] = -1.0
        for index, card in enumerate(cards):
            row[len(decks) + index] = inclusion.get(deck_i, {}).get(card, 0.0) - inclusion.get(deck_j, {}).get(card, 0.0)
        rows.append(row)
        labels.append(int(result))
    estimator = LogisticRegression(C=1.0, max_iter=1000, random_state=1312)
    estimator.fit(np.asarray(rows), np.asarray(labels))
    return FittedCardModel(decks, cards, estimator, inclusion)


def model_artifact(model: FittedCardModel) -> dict[str, Any]:
    return {
        "archetypes": model.decks,
        "win_rate_matrix": {
            deck_i: {
                deck_j: {"win_rate": 0.5 if deck_i == deck_j else model.probability(deck_i, deck_j), "match_count": 0}
                for deck_j in model.decks
            }
            for deck_i in model.decks
        },
    }


def validate_recommendation(cards: Sequence[Mapping[str, Any]], banned_cards: set[str] | None = None) -> None:
    banned_cards = banned_cards or set()
    ace_count = 0
    for row in cards:
        card = str(row["card"])
        copies = int(row["copies"])
        if card in banned_cards:
            raise ValueError(f"banned card in recommendation: {card}")
        if copies < 0 or copies > 4:
            raise ValueError(f"card copy limit exceeded: {card}")
        if card in ACE_SPEC_CARDS:
            ace_count += copies
    if ace_count > 1:
        raise ValueError("recommendation may contain at most one ACE SPEC card")


def recommend_best60(
    archetype: str,
    candidates: Sequence[str],
    coefficients: Mapping[str, float],
    coefficient_intervals: Mapping[str, tuple[float, float]],
    inclusion: Mapping[str, Mapping[str, float]],
    meta_weights: Mapping[str, float],
    banned_cards: set[str] | None = None,
) -> dict[str, Any]:
    banned_cards = banned_cards or set()
    scored = []
    for card in candidates:
        if card in banned_cards:
            continue
        delta = sum(
            weight * (inclusion.get(archetype, {}).get(card, 0.0) - inclusion.get(opponent, {}).get(card, 0.0))
            for opponent, weight in meta_weights.items()
        )
        coefficient = coefficients.get(card, 0.0)
        lower, upper = coefficient_intervals.get(card, (coefficient, coefficient))
        score = coefficient * delta
        interval = (min(lower * delta, upper * delta), max(lower * delta, upper * delta))
        scored.append({
            "card": card,
            "score": score,
            "lower": interval[0],
            "upper": interval[1],
            "bucket": "signal" if interval[0] > 0 or interval[1] < 0 else "no signal",
        })
    scored.sort(key=lambda row: row["score"], reverse=True)
    signal = [row for row in scored if row["bucket"] == "signal"]
    ace_signal = [row for row in signal if row["card"] in ACE_SPEC_CARDS]
    selected = [{**max(ace_signal, key=lambda row: row["score"]), "copies": 1}] if ace_signal else []
    for row in signal:
        if row["card"] in ACE_SPEC_CARDS:
            continue
        if sum(item["copies"] for item in selected) >= 60:
            break
        selected.append({**row, "copies": min(4, 60 - sum(item["copies"] for item in selected))})
    validate_recommendation(selected, banned_cards)
    return {
        "archetype": archetype,
        "cards": selected,
        "no_signal": [row for row in scored if row["bucket"] == "no signal"],
        "observational": True,
        "total_copies": sum(item["copies"] for item in selected),
    }


def fit_h1_misty_variant(
    observations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Report the Misty association with and without the Hammer variant control."""
    def fit(include_variant: bool) -> tuple[float, tuple[float, float]]:
        rows = []
        labels = []
        for observation in observations:
            row = [float(observation.get("misty", 0.0))]
            if include_variant:
                row.append(float(observation.get("hammer_variant", 0.0)))
            rows.append(row)
            labels.append(int(observation["result"]))
        estimator = LogisticRegression(C=1.0, max_iter=1000, random_state=1312)
        estimator.fit(np.asarray(rows), np.asarray(labels))
        coefficient = float(estimator.coef_[0, 0])
        information = np.asarray(rows).T @ np.asarray(rows)
        standard_error = float(np.sqrt(1.0 / max(np.linalg.pinv(information)[0, 0], 1e-9)))
        return coefficient, (coefficient - 1.96 * standard_error, coefficient + 1.96 * standard_error)

    without_variant = fit(False)
    with_variant = fit(True)
    return {
        "hypothesis": "H1",
        "card": "Misty Energy",
        "target": "Alakazam Dudunsparce",
        "without_variant": {"beta": without_variant[0], "interval": without_variant[1]},
        "with_variant": {"beta": with_variant[0], "interval": with_variant[1]},
        "status": "supported" if without_variant[0] > 0 and with_variant[0] > 0 else "rejected",
        "interpretation": "observational association, not a causal effect",
    }
