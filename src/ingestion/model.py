from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from sklearn.linear_model import LogisticRegression


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
