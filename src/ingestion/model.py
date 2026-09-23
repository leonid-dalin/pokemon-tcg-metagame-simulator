from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression

from src.core.config import BDIF_PANEL_MAX_DECKS, BDIF_PANEL_SHARE_THRESHOLD

MIST_ENERGY_NAME = "Mist Energy"


def _decklist_card_names(raw: str | None) -> frozenset[str]:
    if not raw or raw == "null":
        return frozenset()
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return frozenset()
    if not isinstance(payload, dict):
        return frozenset()
    return frozenset(
        str(card["name"])
        for group in payload.values()
        if isinstance(group, list)
        for card in group
        if isinstance(card, dict) and card.get("name") is not None and str(card["name"]).strip()
    )

ACE_SPEC_CARDS = frozenset({
    "Amulet of Hope", "Awakening Drum", "Brilliant Blender", "Dangerous Laser",
    "Deluxe Bomb", "Energy Search Pro", "Enriching Energy", "Grand Tree",
    "Hero's Cape", "Hyper Aroma", "Legacy Energy", "Master Ball", "Max Rod",
    "Maximum Belt", "Megaton Blower", "Miracle Headset", "Neo Upper Energy",
    "Neutralization Zone", "Poke Vital A", "Precious Trolley", "Prime Catcher",
    "Reboot Pod", "Scoop Up Cyclone", "Scramble Switch", "Secret Box",
    "Sparkling Crystal", "Survival Brace", "Treasure Tracker", "Unfair Stamp",
})

BASIC_ENERGY_NAMES = frozenset({"Grass Energy", "Fire Energy", "Water Energy", "Lightning Energy", "Psychic Energy", "Fighting Energy", "Darkness Energy", "Metal Energy"})


def select_panel_decks(
    deck_shares: Mapping[str, float],
    threshold: float = BDIF_PANEL_SHARE_THRESHOLD,
    max_decks: int | None = BDIF_PANEL_MAX_DECKS,
) -> list[str]:
    """Return decks at or above the empirical share threshold in deterministic order."""
    selected = [
        deck
        for deck, share in sorted(deck_shares.items(), key=lambda item: (-item[1], item[0]))
        if share >= threshold
    ]
    return selected[:max_decks] if max_decks is not None else selected


def _logistic_standard_errors(estimator: LogisticRegression, design: np.ndarray) -> np.ndarray:
    probabilities = estimator.predict_proba(design)[:, 1]
    weights = probabilities * (1.0 - probabilities)
    information = design.T @ (weights[:, None] * design)
    covariance = np.linalg.pinv(information)
    return np.sqrt(np.maximum(np.diag(covariance), 0.0))


def benjamini_hochberg(p_values: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    adjusted = {}
    running = 1.0
    for rank, (card, p_value) in reversed(list(enumerate(ordered, start=1))):
        running = min(running, p_value * len(ordered) / rank)
        adjusted[card] = min(1.0, running)
    return adjusted


def _card_limit(card: str, card_rules: Mapping[str, Mapping[str, Any]]) -> int | None:
    rule = card_rules.get(card, {})
    if rule.get("basic_energy") or rule.get("type") == "basic_energy" or card in BASIC_ENERGY_NAMES:
        return None
    if rule.get("ace_spec") or card in ACE_SPEC_CARDS:
        return 1
    return int(rule.get("max_copies", 4))


def _is_ace_spec(card: str, card_rules: Mapping[str, Mapping[str, Any]]) -> bool:
    return bool(card_rules.get(card, {}).get("ace_spec") or card in ACE_SPEC_CARDS)


@dataclass
class FittedCardModel:
    decks: list[str]
    cards: list[str]
    estimator: LogisticRegression
    inclusion: Mapping[str, Mapping[str, float]]
    standard_errors: Mapping[str, float]
    match_counts: Mapping[tuple[str, str], int]

    def coefficient_report(self) -> tuple[dict[str, float], dict[str, tuple[float, float]]]:
        offset = len(self.decks)
        coefficients = {
            card: float(self.estimator.coef_[0, offset + index])
            for index, card in enumerate(self.cards)
        }
        intervals = {card: (value - 1.96 * self.standard_errors.get(card, 1.0), value + 1.96 * self.standard_errors.get(card, 1.0)) for card, value in coefficients.items()}
        return coefficients, intervals

    def probability(self, deck_i: str, deck_j: str) -> float:
        row = np.zeros((1, len(self.decks) + len(self.cards)), dtype=float)
        deck_indices = {deck: index for index, deck in enumerate(self.decks)}
        if deck_i in deck_indices:
            row[0, deck_indices[deck_i]] = 1.0
        if deck_j in deck_indices:
            row[0, deck_indices[deck_j]] = -1.0
        offset = len(self.decks)
        for index, card in enumerate(self.cards):
            row[0, offset + index] = self.inclusion.get(deck_i, {}).get(card, 0.0) - self.inclusion.get(deck_j, {}).get(card, 0.0)
        return float(self.estimator.predict_proba(row)[0, 1])


@dataclass(frozen=True)
class Best60Request:
    archetype: str
    candidates: Sequence[str]
    coefficients: Mapping[str, float]
    coefficient_intervals: Mapping[str, tuple[float, float]]
    inclusion: Mapping[str, Mapping[str, float]]
    meta_weights: Mapping[str, float]
    banned_cards: set[str] | None = None
    card_rules: Mapping[str, Mapping[str, Any]] | None = None
    playable_cards: set[str] | None = None
    skeleton: Sequence[Mapping[str, Any]] | None = None


def fit_model(observations: list[tuple[str, str, int]], inclusion: Mapping[str, Mapping[str, float]]) -> FittedCardModel:
    decks = sorted({deck for row in observations for deck in row[:2]})
    cards = sorted({card for values in inclusion.values() for card in values})
    rows = []
    labels = []
    deck_indices = {deck: index for index, deck in enumerate(decks)}
    for deck_i, deck_j, result in observations:
        row = np.zeros(len(decks) + len(cards), dtype=float)
        row[deck_indices[deck_i]] = 1.0
        row[deck_indices[deck_j]] = -1.0
        for index, card in enumerate(cards):
            row[len(decks) + index] = inclusion.get(deck_i, {}).get(card, 0.0) - inclusion.get(deck_j, {}).get(card, 0.0)
        rows.append(row)
        labels.append(int(result))
    estimator = LogisticRegression(C=1.0, max_iter=1000, random_state=1312)
    design = np.asarray(rows)
    target = np.asarray(labels)
    estimator.fit(design, target)
    standard_errors_array = _logistic_standard_errors(estimator, design)
    standard_errors = {card: float(standard_errors_array[len(decks) + index]) for index, card in enumerate(cards)}
    match_counts: dict[tuple[str, str], int] = {}
    for deck_i, deck_j, _ in observations:
        pair = tuple(sorted((deck_i, deck_j)))
        match_counts[pair] = match_counts.get(pair, 0) + 1
    return FittedCardModel(decks, cards, estimator, inclusion, standard_errors, match_counts)


def model_artifact(model: FittedCardModel) -> dict[str, Any]:
    matrix = {}
    for deck_i in model.decks:
        matrix[deck_i] = {}
        for deck_j in model.decks:
            pair = tuple(sorted((deck_i, deck_j)))
            matrix[deck_i][deck_j] = {
                "win_rate": 0.5 if deck_i == deck_j else model.probability(deck_i, deck_j),
                "match_count": 0 if deck_i == deck_j else model.match_counts.get(pair, 0),
            }
    return {"archetypes": model.decks, "win_rate_matrix": matrix}


def validate_recommendation(cards: Sequence[Mapping[str, Any]], banned_cards: set[str] | None = None, card_rules: Mapping[str, Mapping[str, Any]] | None = None) -> None:
    banned_cards = banned_cards or set()
    card_rules = card_rules or {}
    ace_count = 0
    total_copies = 0
    for row in cards:
        card = str(row["card"])
        copies = int(row["copies"])
        total_copies += copies
        if card in banned_cards:
            raise ValueError(f"banned card in recommendation: {card}")
        limit = _card_limit(card, card_rules)
        if copies < 0 or limit is not None and copies > limit:
            raise ValueError(f"card copy limit exceeded: {card}")
        if card_rules.get(card, {}).get("ace_spec") or card in ACE_SPEC_CARDS:
            ace_count += copies
    if ace_count > 1:
        raise ValueError("recommendation may contain at most one ACE SPEC card")
    if total_copies != 60:
        raise ValueError("recommendation must contain exactly 60 cards")


def recommend_best60(request: Best60Request) -> dict[str, Any]:
    archetype = request.archetype
    candidates = request.candidates
    coefficients = request.coefficients
    coefficient_intervals = request.coefficient_intervals
    inclusion = request.inclusion
    meta_weights = request.meta_weights
    banned_cards = request.banned_cards
    card_rules = request.card_rules
    playable_cards = request.playable_cards
    skeleton = request.skeleton
    banned_cards = banned_cards or set()
    card_rules = card_rules or {}
    playable_cards = set(playable_cards) if playable_cards is not None else set(candidates)
    scored = []
    for card in candidates:
        if card in banned_cards or card not in playable_cards:
            continue
        delta = sum(weight * (inclusion.get(archetype, {}).get(card, 0.0) - inclusion.get(opponent, {}).get(card, 0.0)) for opponent, weight in meta_weights.items())
        coefficient = coefficients.get(card, 0.0)
        lower, upper = coefficient_intervals.get(card, (coefficient, coefficient))
        score = coefficient * delta
        interval = (min(lower * delta, upper * delta), max(lower * delta, upper * delta))
        scored.append({"card": card, "score": score, "lower": interval[0], "upper": interval[1], "bucket": "signal" if interval[0] > 0 or interval[1] < 0 else "no signal"})
    p_values = {}
    for row in scored:
        width = max(row["upper"] - row["lower"], 1e-9)
        p_values[row["card"]] = min(1.0, 2.0 * (1.0 - norm.cdf(abs(row["score"]) / (width / (2 * 1.96)))))
    q_values = benjamini_hochberg(p_values)
    for row in scored:
        row["q_value"] = q_values[row["card"]]
        if row["q_value"] > 0.05 or row["lower"] <= 0 <= row["upper"]:
            row["bucket"] = "no signal"
    scored.sort(key=lambda row: row["score"], reverse=True)
    no_signal = [{"card": row["card"]} for row in scored if row["bucket"] == "no signal"]
    signal = [row for row in scored if row["bucket"] == "signal"]
    card_evidence = {
        card: {
            "inclusion_rate": inclusion.get(archetype, {}).get(card, 0.0),
            "field_inclusion_rate": sum(
                weight * inclusion.get(opponent, {}).get(card, 0.0)
                for opponent, weight in meta_weights.items()
            ),
            "inclusion_delta": inclusion.get(archetype, {}).get(card, 0.0)
            - sum(
                weight * inclusion.get(opponent, {}).get(card, 0.0)
                for opponent, weight in meta_weights.items()
            ),
            "coefficient": coefficients.get(card, 0.0),
            "contribution": coefficients.get(card, 0.0)
            * (
                inclusion.get(archetype, {}).get(card, 0.0)
                - sum(
                    weight * inclusion.get(opponent, {}).get(card, 0.0)
                    for opponent, weight in meta_weights.items()
                )
            ),
            "interval": coefficient_intervals.get(card, (coefficients.get(card, 0.0), coefficients.get(card, 0.0))),
        }
        for card in candidates
    }
    if not skeleton:
        return {
            "archetype": archetype,
            "cards": [],
            "no_signal": no_signal,
            "card_evidence": card_evidence,
            "observational": True,
            "total_copies": 0,
            "status": "missing observed skeleton",
        }

    selected: list[dict[str, Any]] = []
    selected_by_card: dict[str, dict[str, Any]] = {}
    ace_selected = False
    for item in skeleton:
        card = str(item["card"])
        copies = int(item["copies"])
        if copies <= 0 or card in banned_cards:
            continue
        if _is_ace_spec(card, card_rules):
            if ace_selected:
                continue
            copies = 1
            ace_selected = True
        else:
            limit = _card_limit(card, card_rules)
            if limit is not None:
                copies = min(copies, limit)
        if copies:
            selected_by_card[card] = {"card": card, "copies": copies}
    selected = list(selected_by_card.values())

    def add_cards(card: str, copies: int) -> int:
        if copies <= 0 or card in banned_cards or card not in playable_cards:
            return 0
        if _is_ace_spec(card, card_rules) and ace_selected:
            return 0
        existing = selected_by_card.get(card, {"card": card, "copies": 0})
        limit = _card_limit(card, card_rules)
        available = copies if limit is None else min(copies, max(0, limit - int(existing["copies"])))
        available = min(available, 60 - sum(int(item["copies"]) for item in selected_by_card.values()))
        if available <= 0:
            return 0
        existing["copies"] = int(existing["copies"]) + available
        selected_by_card[card] = existing
        return available

    for row in signal:
        if _is_ace_spec(row["card"], card_rules):
            continue
        if sum(int(item["copies"]) for item in selected_by_card.values()) >= 60:
            break
        add_cards(row["card"], 60)

    fallback_cards = sorted(
        (card for card in playable_cards if card not in banned_cards),
        key=lambda card: inclusion.get(archetype, {}).get(card, 0.0),
        reverse=True,
    )
    for card in fallback_cards:
        if card in {row["card"] for row in signal} or _is_ace_spec(card, card_rules):
            continue
        if sum(int(item["copies"]) for item in selected_by_card.values()) >= 60:
            break
        add_cards(card, 60)

    total_copies = sum(int(item["copies"]) for item in selected_by_card.values())
    if total_copies < 60:
        basic_energy = next(
            (card for card in fallback_cards if _card_limit(card, card_rules) is None),
            next((item["card"] for item in selected if _card_limit(item["card"], card_rules) is None), None),
        )
        if basic_energy:
            add_cards(basic_energy, 60 - total_copies)

    selected = list(selected_by_card.values())
    total_copies = sum(int(item["copies"]) for item in selected)
    if total_copies != 60:
        return {
            "archetype": archetype,
            "cards": selected,
            "no_signal": no_signal,
            "card_evidence": card_evidence,
            "observational": True,
            "total_copies": total_copies,
            "status": "insufficient legal observed cards to complete 60",
        }
    validate_recommendation(selected, banned_cards, card_rules)
    return {"archetype": archetype, "cards": selected, "no_signal": no_signal, "card_evidence": card_evidence, "observational": True, "total_copies": total_copies}


def fit_h1_misty_variant(observations: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def fit(include_variant: bool) -> tuple[float, tuple[float, float]]:
        rows, labels = [], []
        for observation in observations:
            row = [float(observation.get("misty", 0.0))]
            if include_variant:
                row.append(float(observation.get("hammer_variant", 0.0)))
            rows.append(row)
            labels.append(int(observation["result"]))
        estimator = LogisticRegression(C=1.0, max_iter=1000, random_state=1312)
        estimator.fit(np.asarray(rows), np.asarray(labels))
        coefficient = float(estimator.coef_[0, 0])
        design = np.asarray(rows)
        standard_error = float(_logistic_standard_errors(estimator, design)[0])
        return coefficient, (coefficient - 1.96 * standard_error, coefficient + 1.96 * standard_error)
    without_variant = fit(False)
    with_variant = fit(True)
    return {"hypothesis": "H1", "card": MIST_ENERGY_NAME, "target": "Alakazam Dudunsparce", "without_variant": {"beta": without_variant[0], "interval": without_variant[1]}, "with_variant": {"beta": with_variant[0], "interval": with_variant[1]}, "status": "supported" if without_variant[0] > 0 and with_variant[0] > 0 else "rejected", "interpretation": "observational association, not a causal effect"}


def h1_observations(rows: Iterable[tuple[Any, ...]]) -> list[dict[str, int]]:
    result = []
    for deck1_id, deck2_id, deck1_raw, deck2_raw, winner, player1, player2 in rows:
        deck1 = json.loads(deck1_raw) if deck1_raw else {}
        deck2 = json.loads(deck2_raw) if deck2_raw else {}
        if not isinstance(deck1, dict):
            deck1 = {}
        if not isinstance(deck2, dict):
            deck2 = {}
        target_is_first = "alakazam" in str(deck1_id).lower()
        target_cards = deck1 if target_is_first else deck2
        opponent_cards = deck2 if target_is_first else deck1
        names = {
            str(card["name"]).lower()
            for group in target_cards.values()
            if isinstance(group, list)
            for card in group
            if isinstance(card, dict) and card.get("name")
        }
        opponent_names = {
            str(card["name"]).lower()
            for group in opponent_cards.values()
            if isinstance(group, list)
            for card in group
            if isinstance(card, dict) and card.get("name")
        }
        if not names:
            continue
        target_player = player1 if target_is_first else player2
        result.append({
            "misty": int(MIST_ENERGY_NAME.lower() in opponent_names),
            "hammer_variant": int("dedenne" in names and "enhanced hammer" in names),
            "result": int(str(winner) == str(target_player)),
        })
    return result
