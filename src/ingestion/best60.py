"""Best-60 builder: an archetype's consensus list, improved by the card-count swaps its results support.

A list's outcome is its match record at its event, so every round counts against the field that
event really fielded. Each player's record at their other events is controlled for, because strong
players adopt cards first. Every card-count slot ("the 3rd Boss's Orders") gets a normal prior whose
width is chosen on held-out events; when no width predicts better than player strength alone, the
builder keeps the consensus list and says so.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

from src.core.config import (
    BDIF_BEST60_APPLY_PROBABILITY,
    BDIF_BEST60_HELD_OUT_SE,
    BDIF_BEST60_LEAN_PROBABILITY,
    BDIF_BEST60_LEVEL_SHARE,
    BDIF_BEST60_LIST_MODE,
    BDIF_BEST60_MAX_SWAPS,
    BDIF_BEST60_MIN_MODEL_LISTS,
    BDIF_BEST60_MIN_SLOT_LISTS,
    BDIF_BEST60_PREREQUISITE_SHARE,
    BDIF_BEST60_PRIOR_GRID,
    BDIF_BEST60_STRENGTH_PRIOR_GAMES,
    BDIF_BEST60_SUPPORT_CHANGES,
    BDIF_BEST60_SUPPORT_LISTS,
    BDIF_BEST60_TREND_DAYS,
    BDIF_BEST60_TREND_POINTS,
)
from src.ingestion.model import ACE_SPEC_CARDS, BASIC_ENERGY_NAMES, validate_recommendation

GROUP_ORDER = ("pokemon", "trainer", "energy")
FOLDS = 5
SEED = 1312
# A finish tier counts only at events large enough to earn it (Ciphermaniac's success-tag floors).
TIERS = (("top50", 0.5, 8), ("top25", 0.25, 12), ("winner", None, 2))

Slot = tuple[str, int]


@dataclass(frozen=True)
class ArchetypeList:
    event: str
    player: str
    deck_id: str | None
    date: str
    players: int
    placing: int | None
    wins: int
    losses: int
    counts: Mapping[str, int]
    groups: Mapping[str, str]


@dataclass(frozen=True)
class SlotModel:
    slots: list[Slot]
    beta: np.ndarray
    covariance: np.ndarray
    prior_sd: float
    intercept: float = 0.0
    strength_beta: float = 0.0
    slot_means: np.ndarray | None = None
    strength_mean: float = 0.0


def parse_list(row: Mapping[str, Any]) -> ArchetypeList | None:
    """Return the row's list when it is a legal 60, otherwise None."""
    try:
        payload = json.loads(row["decklist"]) if isinstance(row["decklist"], str) else row["decklist"]
    except (TypeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    counts: Counter[str] = Counter()
    groups: dict[str, str] = {}
    for group, items in payload.items():
        if not isinstance(items, list):
            continue
        for item in items:
            if isinstance(item, dict) and str(item.get("name") or "").strip():
                counts[str(item["name"])] += int(item.get("count") or 0)
                groups[str(item["name"])] = str(group)
    rows = [{"card": card, "copies": copies} for card, copies in counts.items()]
    try:
        validate_recommendation(rows, card_rules=_rules(counts))
    except ValueError:
        return None
    return ArchetypeList(
        event=str(row["event"]),
        player=str(row["player"]),
        deck_id=row.get("deck_id"),
        date=str(row.get("date") or ""),
        players=int(row.get("players") or 0),
        placing=None if row.get("placing") is None else int(row["placing"]),
        wins=int(row.get("wins") or 0),
        losses=int(row.get("losses") or 0),
        counts=dict(counts),
        groups=groups,
    )


def _rules(cards: Iterable[str]) -> dict[str, dict[str, bool]]:
    return {
        card: {"ace_spec": True} if card in ACE_SPEC_CARDS else {"basic_energy": True} if card in BASIC_ENERGY_NAMES else {}
        for card in cards
    }


def slot_support(lists: Sequence[ArchetypeList]) -> Counter[Slot]:
    """How many lists play at least k copies of each card, for every (card, k)."""
    support: Counter[Slot] = Counter()
    for entry in lists:
        for card, copies in entry.counts.items():
            for copy in range(1, copies + 1):
                support[(card, copy)] += 1
    return support


def consensus_sixty(lists: Sequence[ArchetypeList]) -> Counter[str]:
    """The 60 most-played slots: each card at the count most lists reach, one ACE SPEC at most."""
    support = slot_support(lists)
    deck: Counter[str] = Counter()
    for card, copy in sorted(support, key=lambda slot: (-support[slot], slot[0], slot[1])):
        if sum(deck.values()) == 60:
            break
        if deck[card] != copy - 1:
            continue
        if card in ACE_SPEC_CARDS and any(other in ACE_SPEC_CARDS for other in deck):
            continue
        deck[card] = copy
    return deck


def player_strength(records: Mapping[tuple[str, str], tuple[int, int]]) -> dict[tuple[str, str], float]:
    """Log-odds of each player's record at their other events, shrunk toward even."""
    totals: dict[str, list[int]] = {}
    for (_, player), (wins, losses) in records.items():
        total = totals.setdefault(player, [0, 0])
        total[0] += wins
        total[1] += losses
    prior = BDIF_BEST60_STRENGTH_PRIOR_GAMES
    strength = {}
    for (event, player), (wins, losses) in records.items():
        other_wins = totals[player][0] - wins
        other_losses = totals[player][1] - losses
        strength[(event, player)] = math.log((other_wins + prior) / (other_losses + prior))
    return strength


def _modelled_slots(lists: Sequence[ArchetypeList]) -> list[Slot]:
    support = slot_support(lists)
    floor = BDIF_BEST60_MIN_SLOT_LISTS
    return sorted(slot for slot, count in support.items() if count >= floor and len(lists) - count >= floor)


def _slot_matrix(lists: Sequence[ArchetypeList], slots: Sequence[Slot]) -> np.ndarray:
    return np.array([[entry.counts.get(card, 0) >= copy for card, copy in slots] for entry in lists], dtype=float)


def _fit(design: np.ndarray, wins: np.ndarray, games: np.ndarray, penalty: np.ndarray) -> np.ndarray:
    def objective(coef: np.ndarray) -> tuple[float, np.ndarray]:
        eta = design @ coef
        probability = 1.0 / (1.0 + np.exp(-eta))
        loss = -(wins * eta - games * np.logaddexp(0.0, eta)).sum() + 0.5 * (penalty * coef * coef).sum()
        return loss, -(design.T @ (wins - games * probability)) + penalty * coef

    result = minimize(objective, np.zeros(design.shape[1]), jac=True, method="L-BFGS-B", options={"maxiter": 5_000, "gtol": 1e-8})
    return result.x


def _fit_strength_model(
    lists: Sequence[ArchetypeList],
    strength: Mapping[tuple[str, str], float],
) -> SlotModel:
    strength_values = _strength_values(lists, strength)
    strength_mean = float(strength_values.mean()) if len(strength_values) else 0.0
    wins, games, skill = _outcomes(lists, strength, strength_mean)
    design = np.column_stack([np.ones(len(skill)), skill])
    coef = _fit(design, wins, games, np.full(2, 1e-4))
    return SlotModel(
        slots=[],
        beta=np.zeros(0),
        covariance=np.zeros((0, 0)),
        prior_sd=0.0,
        intercept=float(coef[0]),
        strength_beta=float(coef[1]),
        slot_means=np.zeros(0),
        strength_mean=strength_mean,
    )


def _strength_values(lists: Sequence[ArchetypeList], strength: Mapping[tuple[str, str], float]) -> np.ndarray:
    return np.array([strength.get((entry.event, entry.player), 0.0) for entry in lists], dtype=float)


def _outcomes(
    lists: Sequence[ArchetypeList],
    strength: Mapping[tuple[str, str], float],
    skill_mean: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    wins = np.array([entry.wins for entry in lists], dtype=float)
    games = np.array([entry.wins + entry.losses for entry in lists], dtype=float)
    skill = _strength_values(lists, strength)
    if skill_mean is None:
        skill_mean = float(skill.mean()) if len(skill) else 0.0
    return wins, games, skill - skill_mean


def _design(slot_matrix: np.ndarray, skill: np.ndarray, with_slots: bool) -> np.ndarray:
    base = np.column_stack([np.ones(len(skill)), skill])
    return np.hstack([base, slot_matrix - slot_matrix.mean(axis=0)]) if with_slots else base


def _penalty(slot_count: int, prior_sd: float | None) -> np.ndarray:
    slots = np.full(slot_count, 1.0 / prior_sd**2) if prior_sd else np.zeros(0)
    return np.r_[np.full(2, 1e-4), slots]


def event_folds(lists: Sequence[ArchetypeList]) -> np.ndarray:
    """Assign whole events to folds, so a held-out fold never shares an event with its training data."""
    events = sorted({entry.event for entry in lists})
    order = np.random.default_rng(SEED).permutation(len(events)) % FOLDS
    fold_of = dict(zip(events, order))
    return np.array([fold_of[entry.event] for entry in lists])


def _fold_strength(
    records: Mapping[tuple[str, str], tuple[int, int]],
    training_events: set[str],
    validation_events: set[str],
) -> dict[tuple[str, str], float]:
    training_records = {key: value for key, value in records.items() if key[0] in training_events}
    strength = player_strength(training_records)
    totals: dict[str, list[int]] = {}
    for (_, player), (wins, losses) in training_records.items():
        total = totals.setdefault(player, [0, 0])
        total[0] += wins
        total[1] += losses
    prior = BDIF_BEST60_STRENGTH_PRIOR_GAMES
    for (event, player) in records:
        if event in validation_events:
            wins, losses = totals.get(player, (0, 0))
            strength[(event, player)] = math.log((wins + prior) / (losses + prior))
    return strength


def _model_design(
    lists: Sequence[ArchetypeList],
    strength: Mapping[tuple[str, str], float],
    model: SlotModel,
) -> np.ndarray:
    matrix = _slot_matrix(lists, model.slots)
    slot_means = model.slot_means if model.slot_means is not None else np.zeros(len(model.slots))
    skill = _strength_values(lists, strength) - model.strength_mean
    return np.hstack([np.ones((len(lists), 1)), skill[:, None], matrix - slot_means])


def _model_loss(design: np.ndarray, wins: np.ndarray, games: np.ndarray, model: SlotModel) -> float:
    coefficient = np.r_[model.intercept, model.strength_beta, model.beta]
    eta = design @ coefficient
    return float(-(wins * eta - games * np.logaddexp(0.0, eta)).sum())


def held_out_losses(
    lists: Sequence[ArchetypeList],
    strength: Mapping[tuple[str, str], float],
    records: Mapping[tuple[str, str], tuple[int, int]] | None = None,
) -> dict[str, float]:
    """Log loss per game on held-out events: player strength alone, then with slots at each prior width."""
    if len(set(entry.event for entry in lists)) < FOLDS:
        return {"strength only": 0.0}
    folds = event_folds(lists)
    games = np.array([entry.wins + entry.losses for entry in lists], dtype=float)
    losses = {}
    for label, prior_sd in [("strength only", None), *((f"prior sd {sd}", sd) for sd in BDIF_BEST60_PRIOR_GRID)]:
        total = 0.0
        for fold in range(FOLDS):
            train, test = folds != fold, folds == fold
            train_events = {entry.event for entry, selected in zip(lists, train) if selected}
            test_events = {entry.event for entry, selected in zip(lists, test) if selected}
            fold_strength = _fold_strength(records, train_events, test_events) if records is not None else strength
            train_lists = [entry for entry, selected in zip(lists, train) if selected]
            test_lists = [entry for entry, selected in zip(lists, test) if selected]
            model = _fit_strength_model(train_lists, fold_strength) if prior_sd is None else fit_slot_model(train_lists, fold_strength, prior_sd)
            test_wins = np.array([entry.wins for entry in test_lists], dtype=float)
            test_games = games[test]
            total += _model_loss(_model_design(test_lists, fold_strength, model), test_wins, test_games, model)
        losses[label] = total / max(1.0, float(games.sum()))
    return losses


def fit_slot_model(lists: Sequence[ArchetypeList], strength: Mapping[tuple[str, str], float], prior_sd: float) -> SlotModel:
    slots = _modelled_slots(lists)
    matrix = _slot_matrix(lists, slots)
    strength_values = _strength_values(lists, strength)
    strength_mean = float(strength_values.mean()) if len(strength_values) else 0.0
    wins, games, skill = _outcomes(lists, strength, strength_mean)
    design = _design(matrix, skill, True)
    penalty = _penalty(len(slots), prior_sd)
    coef = _fit(design, wins, games, penalty)
    probability = 1.0 / (1.0 + np.exp(-(design @ coef)))
    information = design.T @ (design * (games * probability * (1.0 - probability))[:, None]) + np.diag(penalty)
    covariance = np.linalg.inv(information)
    return SlotModel(
        slots=slots,
        beta=coef[2:],
        covariance=covariance[2:, 2:],
        prior_sd=prior_sd,
        intercept=float(coef[0]),
        strength_beta=float(coef[1]),
        slot_means=matrix.mean(axis=0) if len(matrix) else np.zeros(len(slots)),
        strength_mean=strength_mean,
    )


def prerequisites(lists: Sequence[ArchetypeList]) -> dict[str, set[str]]:
    """Cards that appear alongside a card in nearly every list that plays it (evolution lines, engines)."""
    plays: dict[str, set[int]] = {}
    for index, entry in enumerate(lists):
        for card in entry.counts:
            plays.setdefault(card, set()).add(index)
    needs = {}
    for card, mine in plays.items():
        needs[card] = {
            other for other, theirs in plays.items()
            if other != card and len(mine & theirs) / len(mine) >= BDIF_BEST60_PREREQUISITE_SHARE
        }
    return needs


def count_levels(lists: Sequence[ArchetypeList], consensus: Mapping[str, int]) -> dict[str, set[int]]:
    """Counts of each card that enough lists play exactly (BDIF_BEST60_LEVEL_SHARE of them, and at least
    BDIF_BEST60_MIN_SLOT_LISTS), plus its consensus count.

    A card such as Transformation Tome that is played at 0 or 4 and almost never in between has
    levels {0, 4}: Best-60 may move it from 4 to 0 in one step, but never leaves it at 1 to 3.
    """
    floor = max(BDIF_BEST60_MIN_SLOT_LISTS, BDIF_BEST60_LEVEL_SHARE * len(lists))
    levels = {}
    for card in {card for entry in lists for card in entry.counts}:
        exact = Counter(entry.counts.get(card, 0) for entry in lists)
        levels[card] = {count for count, seen in exact.items() if seen >= floor} | {consensus.get(card, 0)}
    return levels


class ListSupport:
    """How many observed lists sit within a number of card changes of a 60-card list."""

    def __init__(self, lists: Sequence[ArchetypeList]):
        self.cards = sorted({card for entry in lists for card in entry.counts})
        self.position = {card: index for index, card in enumerate(self.cards)}
        self.matrix = np.array([[entry.counts.get(card, 0) for card in self.cards] for entry in lists], dtype=float).reshape(len(lists), len(self.cards))

    def within(self, deck: Mapping[str, int], changes: int) -> int:
        target = np.zeros(len(self.cards))
        unseen = 0
        for card, copies in deck.items():
            if card in self.position:
                target[self.position[card]] = copies
            else:
                unseen += copies
        distance = (np.abs(self.matrix - target).sum(axis=1) + unseen) / 2
        return int((distance <= changes).sum())

    def profile(self, deck: Mapping[str, int]) -> dict[int, int]:
        return {changes: self.within(deck, changes) for changes in (2, 4, 6)}


def _allowed(card: str, count: int, levels: Mapping[str, set[int]] | None) -> bool:
    """Basic Energy is fungible, so any count is allowed; other cards stay at counts lists play."""
    return levels is None or card in BASIC_ENERGY_NAMES or count in levels.get(card, {count})


def _priced_move(removed: list[Slot], added: list[Slot], index: Mapping[Slot, int], model: SlotModel) -> dict[str, Any]:
    weights = np.zeros(len(model.slots))
    for slot in added:
        weights[index[slot]] += 1.0
    for slot in removed:
        weights[index[slot]] -= 1.0
    gain = float(weights @ model.beta)
    variance = float(weights @ model.covariance @ weights)
    probability = float(norm.cdf(gain / math.sqrt(max(variance, 1e-12))))
    return {
        "remove": removed[0][0], "remove_copy": removed[0][1],
        "add": added[0][0], "add_copy": added[0][1],
        "removed": removed, "added": added,
        "gain": gain, "probability": probability,
    }


def _candidate_swaps(
    deck: Counter[str],
    model: SlotModel,
    groups: Mapping[str, str],
    needs: Mapping[str, set[str]],
    levels: Mapping[str, set[int]] | None = None,
) -> list[dict[str, Any]]:
    """Every move from this list: one copy for one copy, or one card to its next played count.

    A level move (for example 4 Transformation Tome to 0) is filled or freed by the best single-copy
    changes, and the whole move is priced together, with the covariance of every slot it touches.
    """
    index = {slot: position for position, slot in enumerate(model.slots)}
    modelled = sorted({slot[0] for slot in model.slots})

    def keeps_last(card: str, state: Mapping[str, int]) -> bool:
        present = {other for other, copies in state.items() if copies > 0}
        return groups.get(card) == "pokemon" or any(card in needs.get(other, set()) for other in present if other != card)

    def removals(state: Mapping[str, int]) -> list[Slot]:
        found = []
        for card, copies in state.items():
            if copies <= 0 or (card, copies) not in index or not _allowed(card, copies - 1, levels):
                continue
            if copies == 1 and keeps_last(card, state):
                continue
            found.append((card, copies))
        return found

    def additions(state: Mapping[str, int]) -> list[Slot]:
        present = {card for card, copies in state.items() if copies > 0}
        found = []
        for card in modelled:
            copy = state.get(card, 0) + 1
            if (card, copy) not in index or not _allowed(card, copy, levels):
                continue
            if card not in BASIC_ENERGY_NAMES and copy > 4:
                continue
            if card in ACE_SPEC_CARDS and any(other in ACE_SPEC_CARDS for other in present if other != card):
                continue
            if copy == 1 and not needs.get(card, set()) <= present:
                continue
            found.append((card, copy))
        return found

    moves = {}

    def keep(move: dict[str, Any]) -> None:
        moves[(tuple(sorted(move["removed"])), tuple(sorted(move["added"])))] = move

    for removed in removals(deck):
        for added in additions(deck):
            if added[0] != removed[0]:
                keep(_priced_move([removed], [added], index, model))
    if levels is None:
        return list(moves.values())
    for card in modelled:
        current = deck.get(card, 0)
        lower = [level for level in levels.get(card, ()) if level < current]
        if lower and current - max(lower) > 1:
            target = max(lower)
            block = [(card, copy) for copy in range(current, target, -1)]
            if all(slot in index for slot in block) and not (target == 0 and keeps_last(card, deck)):
                state = Counter(deck)
                state[card] = target
                fill = []
                for _ in block:
                    options = [slot for slot in additions(state) if slot[0] != card]
                    if not options:
                        break
                    best = max(options, key=lambda slot: (model.beta[index[slot]], slot))
                    fill.append(best)
                    state[best[0]] += 1
                if len(fill) == len(block):
                    keep(_priced_move(block, fill, index, model))
        higher = [level for level in levels.get(card, ()) if level > current]
        if higher and min(higher) - current > 1:
            target = min(higher)
            block = [(card, copy) for copy in range(current + 1, target + 1)]
            present = {other for other, copies in deck.items() if copies > 0}
            legal = (card in BASIC_ENERGY_NAMES or target <= 4) and not (
                card in ACE_SPEC_CARDS) and (current > 0 or needs.get(card, set()) <= present)
            if legal and all(slot in index for slot in block):
                state = Counter(deck)
                state[card] = target
                freed = []
                for _ in block:
                    options = [slot for slot in removals(state) if slot[0] != card]
                    if not options:
                        break
                    worst = min(options, key=lambda slot: (model.beta[index[slot]], slot))
                    freed.append(worst)
                    state[worst[0]] -= 1
                if len(freed) == len(block):
                    keep(_priced_move(freed, block, index, model))
    return list(moves.values())


def _after(deck: Mapping[str, int], move: Mapping[str, Any]) -> Counter[str]:
    state = Counter(deck)
    for card, _ in move["removed"]:
        state[card] -= 1
    for card, _ in move["added"]:
        state[card] += 1
    return Counter({card: copies for card, copies in state.items() if copies > 0})


def improve(
    deck: Counter[str],
    model: SlotModel,
    groups: Mapping[str, str],
    needs: Mapping[str, set[str]],
    levels: Mapping[str, set[int]] | None = None,
    supported: Callable[[Mapping[str, int]], bool] | None = None,
) -> tuple[Counter[str], list[dict[str, Any]], list[dict[str, Any]]]:
    """Apply the largest-gain move that is likely enough to help, until none is; return the leaning ones too.

    With `supported`, a move is applied only if the list it leads to passes that check (observed mode).
    """
    deck = Counter(deck)
    applied: list[dict[str, Any]] = []
    raised: set[str] = set()
    lowered: set[str] = set()

    def keeps_direction(move: Mapping[str, Any]) -> bool:
        """A move never takes back an earlier one: no removing a card a move added, or re-adding one it cut."""
        return not ({card for card, _ in move["removed"]} & raised or {card for card, _ in move["added"]} & lowered)

    while len(applied) < BDIF_BEST60_MAX_SWAPS:
        likely = [swap for swap in _candidate_swaps(deck, model, groups, needs, levels) if swap["probability"] >= BDIF_BEST60_APPLY_PROBABILITY and keeps_direction(swap)]
        likely.sort(key=lambda row: (row["gain"], row["add"], row["remove"]), reverse=True)
        chosen = next((swap for swap in likely if supported is None or supported(_after(deck, swap))), None)
        if chosen is None:
            break
        deck = _after(deck, chosen)
        applied.append(chosen)
        raised |= {card for card, _ in chosen["added"]}
        lowered |= {card for card, _ in chosen["removed"]}
    leaning = sorted(
        (swap for swap in _candidate_swaps(deck, model, groups, needs, levels)
         if BDIF_BEST60_LEAN_PROBABILITY <= swap["probability"] < BDIF_BEST60_APPLY_PROBABILITY and swap["gain"] > 0 and keeps_direction(swap)),
        key=lambda row: (-row["gain"], row["add"], row["remove"]),
    )[:5]
    return deck, applied, leaning


def joint_probability(consensus: Mapping[str, int], deck: Mapping[str, int], model: SlotModel) -> float | None:
    """Chance that the whole recommended list beats the consensus, from every slot it changes at once."""
    index = {slot: position for position, slot in enumerate(model.slots)}
    weights = np.zeros(len(model.slots))
    for card in set(consensus) | set(deck):
        before, after = consensus.get(card, 0), deck.get(card, 0)
        for copy in range(min(before, after) + 1, max(before, after) + 1):
            if (card, copy) not in index:
                return None
            weights[index[(card, copy)]] += 1.0 if after > before else -1.0
    if not weights.any():
        return None
    gain = float(weights @ model.beta)
    return float(norm.cdf(gain / math.sqrt(max(float(weights @ model.covariance @ weights), 1e-12))))


def _support_check(lists: Sequence[ArchetypeList], mode: str) -> Callable[[Mapping[str, int]], bool] | None:
    if mode not in ("observed", "novel"):
        raise ValueError(f"unknown Best-60 list mode: {mode}")
    if mode == "novel":
        return None
    support = ListSupport(lists)
    return lambda deck: support.within(deck, BDIF_BEST60_SUPPORT_CHANGES) >= BDIF_BEST60_SUPPORT_LISTS


def _tier_reached(entry: ArchetypeList, fraction: float | None, floor: int) -> bool | None:
    if entry.players < floor or entry.placing is None:
        return None
    cutoff = 1 if fraction is None else max(1, math.ceil(entry.players * fraction))
    return entry.placing <= cutoff


def card_stats(lists: Sequence[ArchetypeList]) -> dict[str, dict[str, Any]]:
    """Play rate, average copies and finish-tier rates with and without each card."""
    stats = {}
    cards = sorted({card for entry in lists for card in entry.counts})
    for card in cards:
        with_card = [entry for entry in lists if card in entry.counts]
        row: dict[str, Any] = {
            "play_rate": len(with_card) / len(lists),
            "average_copies": sum(entry.counts[card] for entry in with_card) / len(with_card),
        }
        for tier, fraction, floor in TIERS:
            reached_with = [r for r in (_tier_reached(entry, fraction, floor) for entry in with_card) if r is not None]
            reached_all = [r for r in (_tier_reached(entry, fraction, floor) for entry in lists) if r is not None]
            row[tier] = {
                "with_card": sum(reached_with) / len(reached_with) if reached_with else None,
                "archetype": sum(reached_all) / len(reached_all) if reached_all else None,
                "lists": len(reached_with),
            }
        stats[card] = row
    return stats


def _parse_date(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def trends(lists: Sequence[ArchetypeList], stats: Mapping[str, Mapping[str, Any]], model: SlotModel | None = None) -> dict[str, Any]:
    """Play-rate change between the first and last weeks of the snapshot.

    A breakthrough is a rising card that out-finishes its archetype and, when the slot model scores the
    card, whose first copy the model rates positive once player strength is controlled.
    """
    effects = {} if model is None else {card: float(model.beta[index]) for index, (card, copy) in enumerate(model.slots) if copy == 1}
    dated = [(when, entry) for entry in lists if (when := _parse_date(entry.date)) is not None]
    if not dated:
        return {"status": "no dated lists"}
    first, last = min(when for when, _ in dated), max(when for when, _ in dated)
    span = timedelta(days=BDIF_BEST60_TREND_DAYS)
    early = [entry for when, entry in dated if when < first + span]
    late = [entry for when, entry in dated if when > last - span]
    if last - first < 2 * span or len(early) < BDIF_BEST60_MIN_SLOT_LISTS or len(late) < BDIF_BEST60_MIN_SLOT_LISTS:
        return {"status": "window too short or too thin", "early_lists": len(early), "late_lists": len(late)}

    def share(group: Sequence[ArchetypeList], card: str) -> float:
        return sum(card in entry.counts for entry in group) / len(group)

    def copies(group: Sequence[ArchetypeList], card: str) -> float:
        played = [entry.counts[card] for entry in group if card in entry.counts]
        return sum(played) / len(played) if played else 0.0

    rows = []
    for card in sorted({card for entry in lists for card in entry.counts}):
        start, end = share(early, card), share(late, card)
        top25 = stats[card]["top25"]
        out_finishes = top25["with_card"] is not None and top25["archetype"] is not None and top25["with_card"] > top25["archetype"]
        rows.append({
            "card": card, "start_share": start, "end_share": end, "change": end - start,
            "copies_change": copies(late, card) - copies(early, card), "out_finishes_archetype": out_finishes,
            "model_effect": effects.get(card),
        })
    rising = [row for row in rows if row["change"] >= BDIF_BEST60_TREND_POINTS]
    return {
        "status": "complete",
        "window_days": BDIF_BEST60_TREND_DAYS,
        "early_lists": len(early),
        "late_lists": len(late),
        "rising": sorted(rising, key=lambda row: -row["change"]),
        "falling": sorted((row for row in rows if row["change"] <= -BDIF_BEST60_TREND_POINTS), key=lambda row: row["change"]),
        "breakthrough": [row["card"] for row in sorted(rising, key=lambda row: -row["change"]) if row["out_finishes_archetype"] and (row["model_effect"] is None or row["model_effect"] > 0)],
    }


def held_out_gain(
    lists: Sequence[ArchetypeList],
    strength: Mapping[tuple[str, str], float],
    prior_sd: float,
    consensus: Counter[str],
    groups: Mapping[str, str],
    records: Mapping[tuple[str, str], tuple[int, int]] | None = None,
    list_mode: str = "novel",
) -> tuple[float, float]:
    """Cross-fitted value of the swap procedure, in log-odds: mean and standard error over five folds.

    Swaps are chosen by a model fitted on four folds, then valued by slot coefficients fitted only on
    the fifth. The held-out lists never influence which swaps are chosen, and the training lists never
    influence their value, so a swap that only looks good in sample is worth about zero here.
    """
    folds = event_folds(lists)
    if len(set(entry.event for entry in lists)) < FOLDS:
        return 0.0, 0.0
    gains = []
    for fold in range(FOLDS):
        train = [entry for entry, f in zip(lists, folds) if f != fold]
        test = [entry for entry, f in zip(lists, folds) if f == fold]
        train_events = {entry.event for entry in train}
        test_events = {entry.event for entry in test}
        fold_strength = _fold_strength(records, train_events, test_events) if records is not None else strength
        _, applied, _ = improve(
            consensus, fit_slot_model(train, fold_strength, prior_sd), groups, prerequisites(train),
            count_levels(train, consensus), _support_check(train, list_mode),
        )
        priced = fit_slot_model(test, fold_strength, prior_sd)
        beta = dict(zip(priced.slots, priced.beta))
        gains.append(sum(
            sum(beta.get(slot, 0.0) for slot in move["added"]) - sum(beta.get(slot, 0.0) for slot in move["removed"])
            for move in applied
        ))
    return float(np.mean(gains)), float(np.std(gains, ddof=1) / math.sqrt(len(gains)))


def _expected_rate(lists: Sequence[ArchetypeList], gain: float) -> float:
    wins = sum(entry.wins for entry in lists)
    games = sum(entry.wins + entry.losses for entry in lists)
    rate = min(max(wins / games, 1e-6), 1 - 1e-6) if games else 0.5
    logit = math.log(rate / (1.0 - rate)) + gain
    return 1.0 / (1.0 + math.exp(-logit))


def build_best60(
    archetype: str,
    rows: Sequence[Mapping[str, Any]],
    strength: Mapping[tuple[str, str], float],
    records: Mapping[tuple[str, str], tuple[int, int]] | None = None,
    list_mode: str | None = None,
) -> dict[str, Any]:
    """The Best-60 report for one archetype from its stored lists.

    `list_mode` "novel" (the default) may recommend a list nobody has played; "observed" stops before
    the list has fewer than BDIF_BEST60_SUPPORT_LISTS observed lists within BDIF_BEST60_SUPPORT_CHANGES.
    """
    list_mode = list_mode or BDIF_BEST60_LIST_MODE
    lists = [entry for entry in (parse_list(row) for row in rows) if entry is not None]
    _support_check(lists, list_mode)
    records = records or {(entry.event, entry.player): (entry.wins, entry.losses) for entry in lists}
    deck_ids = sorted({str(row.get("deck_id")) for row in rows if row.get("deck_id")})
    report: dict[str, Any] = {
        "archetype": archetype, "deck_ids": deck_ids, "list_count": len(rows),
        "legal_list_count": len(lists), "observational": True,
    }
    if not lists:
        return {**report, "status": "no legal lists", "cards": [], "total_copies": 0}
    consensus = consensus_sixty(lists)
    groups: dict[str, str] = {}
    for entry in lists:
        for card, group in entry.groups.items():
            groups.setdefault(card, group)
    deck, applied, proposed, leaning, model = Counter(consensus), [], [], [], None
    model_report: dict[str, Any] = {"outcome": "match record at the event, player strength controlled", "prior_sd": None}
    if len(lists) < BDIF_BEST60_MIN_MODEL_LISTS:
        status = "consensus only: too few lists to score cards"
    else:
        losses = held_out_losses(lists, strength, records)
        best_label = min(losses, key=lambda label: (losses[label], label))
        model_report["held_out_loss"] = losses
        if best_label == "strength only":
            status = "consensus only: card counts did not predict held-out results"
        else:
            prior_sd = float(best_label.removeprefix("prior sd "))
            model = fit_slot_model(lists, strength, prior_sd)
            improved, proposed, leaning = improve(
                consensus, model, groups, prerequisites(lists), count_levels(lists, consensus), _support_check(lists, list_mode),
            )
            gain_held_out, gain_se = held_out_gain(lists, strength, prior_sd, consensus, groups, records, list_mode)
            model_report.update({
                "prior_sd": prior_sd, "slots": len(model.slots),
                "held_out_gain": gain_held_out, "held_out_gain_se": gain_se,
            })
            if gain_held_out - BDIF_BEST60_HELD_OUT_SE * gain_se > 0:
                deck, applied, status = improved, proposed, "complete"
            else:
                status = "consensus kept: swaps did not hold up on held-out events"
    cards = [
        {"card": card, "copies": copies, "consensus_copies": consensus.get(card, 0), "group": groups.get(card, "trainer"),
         "source": "consensus" if consensus.get(card, 0) == copies else "swap"}
        for card, copies in sorted(deck.items(), key=lambda item: (GROUP_ORDER.index(groups.get(item[0], "trainer")) if groups.get(item[0], "trainer") in GROUP_ORDER else 1, -item[1], item[0]))
    ]
    removed = [
        {"card": card, "copies": 0, "consensus_copies": copies, "group": groups.get(card, "trainer"), "source": "swap"}
        for card, copies in sorted(consensus.items()) if card not in deck
    ]
    validate_recommendation(cards, card_rules=_rules(deck))
    stats = card_stats(lists)
    gain = sum(swap["gain"] for swap in applied)
    support = ListSupport(lists)
    return {
        **report,
        "status": status,
        "cards": cards,
        "removed_cards": removed,
        "total_copies": sum(deck.values()),
        "ace_spec_choice": next((card for card in deck if card in ACE_SPEC_CARDS), None),
        "swaps": applied,
        "proposed_swaps": [] if applied else proposed,
        "leaning_swaps": leaning,
        "model": model_report,
        "list_mode": list_mode,
        "support": {"consensus": support.profile(consensus), "recommended": support.profile(deck)},
        "joint_probability": None if model is None or not applied else joint_probability(consensus, deck, model),
        "match_win_rate": {
            "archetype_average": _expected_rate(lists, 0.0),
            "with_swaps_in_sample": _expected_rate(lists, gain),
            "with_swaps_held_out": _expected_rate(lists, model_report.get("held_out_gain", 0.0)),
        },
        "card_stats": stats,
        "trends": trends(lists, stats, model),
    }
