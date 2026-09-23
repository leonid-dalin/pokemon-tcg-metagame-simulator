import numpy as np

from src.ingestion.model import PlayerObservation, fit_card_model


RNG = np.random.default_rng(11)
DECKS = ("A", "B", "C", "D")
STRENGTHS = {"A": 0.3, "B": 0.0, "C": -0.1, "D": -0.2}
TECH_PRESENCE = {"A": 0.6, "B": 0.3, "C": 0.5, "D": 0.1}
FILLER_PRESENCE = {"A": 0.5, "B": 0.5, "C": 0.2, "D": 0.8}


def _presence(deck, rates):
    return rates[deck] > RNG.random()


def _logistic(value):
    return 1.0 / (1.0 + np.exp(-value))


DATA = []
for _ in range(3000):
    deck, opponent = RNG.choice(DECKS, size=2, replace=False)
    deck_cards = frozenset(
        card
        for card, rates in (("Tech", TECH_PRESENCE), ("Filler", FILLER_PRESENCE))
        if _presence(deck, rates)
    )
    opponent_cards = frozenset(
        card
        for card, rates in (("Tech", TECH_PRESENCE), ("Filler", FILLER_PRESENCE))
        if _presence(opponent, rates)
    )
    logit = STRENGTHS[deck] - STRENGTHS[opponent]
    logit += 0.4 * (int("Tech" in deck_cards) - int("Tech" in opponent_cards))
    result = int(RNG.random() < _logistic(logit))
    DATA.append(PlayerObservation(deck, opponent, deck_cards, opponent_cards, result))


def _fit(card):
    def fit(data):
        model = fit_card_model(data, ["Filler", "Tech"])
        coefficients, intervals = model.coefficient_report()
        return coefficients[card], intervals[card]

    return fit


CASES = [
    {"name": "Tech card coefficient", "data": DATA, "fit": _fit("Tech")},
    {"name": "Filler card coefficient", "data": DATA, "fit": _fit("Filler")},
]