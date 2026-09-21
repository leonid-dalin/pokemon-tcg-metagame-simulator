"""Optional offline card-covariate matchup model."""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Iterable, Mapping


CARD_COVARIATES_ENABLED = os.environ.get("BDIF_CARD_COVARIATES_ENABLED", "false").lower() in {
    "1", "true", "yes"
}


@dataclass(frozen=True)
class CardCovariateModel:
    strengths: Mapping[str, float]
    coefficients: Mapping[str, float]
    enabled: bool = CARD_COVARIATES_ENABLED

    def logit_probability(
        self,
        deck_i: str,
        deck_j: str,
        cards_i: Mapping[str, float],
        cards_j: Mapping[str, float],
    ) -> float:
        if not self.enabled:
            raise RuntimeError("card covariate model is disabled")
        logit = self.strengths.get(deck_i, 0.0) - self.strengths.get(deck_j, 0.0)
        for card, coefficient in self.coefficients.items():
            logit += coefficient * (cards_i.get(card, 0.0) - cards_j.get(card, 0.0))
        return 1.0 / (1.0 + math.exp(-logit))


def fit_strengths(
    observations: Iterable[tuple[str, str, float]],
) -> dict[str, float]:
    """Fit a deterministic offline baseline from observed pairwise rates."""
    totals: dict[str, float] = {}
    for winner, loser, result in observations:
        totals[winner] = totals.get(winner, 0.0) + float(result)
        totals[loser] = totals.get(loser, 0.0) - float(result)
    return {deck: value for deck, value in sorted(totals.items())}
