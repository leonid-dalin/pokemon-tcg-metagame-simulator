"""Game-level card model for Best-60: matchups, both players' strength, recency and opponent-specific effects.

Each game a list played is one row: the chance of winning is the matchup against the opponent's deck,
plus the difference between the two players' strength, plus the list's card-count effects. A card-count
effect has a shared part and, for opponents met often enough, an opponent-specific part shrunk toward
zero. Games can be weighted by recency. A list without stored pairings enters as its match record
against an unknown opponent, which is the per-list model Best-60 used before.

"Weighted across the field" is then exact: a card count's value is its shared effect plus its
opponent-specific effects averaged over a field (the recent observed field, or one the caller gives).
"""
from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from datetime import datetime
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import sparse
from scipy.optimize import minimize

from src.core.config import BDIF_BEST60_MIN_OPPONENT_GAMES

UNKNOWN = "(unknown opponent)"
Slot = tuple[str, int]


@dataclass(frozen=True)
class Game:
    event: str
    player: str
    opponent: str
    opponent_deck: str
    won: int
    date: str = ""


@dataclass(frozen=True)
class ModelSpec:
    """Everything a fit needs besides the lists and player strength."""

    prior_sd: float
    opponent_sd: float = 0.0
    half_life: float | None = None
    games: tuple[Game, ...] | None = None
    field: Mapping[str, float] | None = None
    as_of: str | None = None


def parse_date(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def reference_date(lists: Sequence[Any], as_of: str | None) -> datetime | None:
    pinned = parse_date(as_of)
    if pinned is not None:
        return pinned
    dated = [when for when in (parse_date(entry.date) for entry in lists) if when is not None]
    return max(dated) if dated else None


def recency_weight(date: str, reference: datetime | None, half_life: float | None) -> float:
    if half_life is None or reference is None:
        return 1.0
    when = parse_date(date)
    if when is None:
        return 1.0
    age = max(0.0, (reference - when).total_seconds() / 86_400.0)
    return 0.5 ** (age / half_life)


def games_by_list(games: Sequence[Game] | None) -> dict[tuple[str, str], list[Game]]:
    found: dict[tuple[str, str], list[Game]] = {}
    for game in games or ():
        found.setdefault((game.event, game.player), []).append(game)
    return found


@dataclass
class Rows:
    list_index: np.ndarray
    opponent: list[str]
    wins: np.ndarray
    games: np.ndarray
    strength: np.ndarray
    weight: np.ndarray


def build_rows(lists: Sequence[Any], strength: Mapping[tuple[str, str], float], spec: ModelSpec) -> Rows:
    """One row per game, or one per list without pairings (its record against an unknown opponent)."""
    by_list = games_by_list(spec.games)
    reference = reference_date(lists, spec.as_of)
    index, opponent, wins, games, gap, weight = [], [], [], [], [], []
    for position, entry in enumerate(lists):
        own = strength.get((entry.event, entry.player), 0.0)
        mine = by_list.get((entry.event, entry.player))
        if mine:
            for game in mine:
                index.append(position)
                opponent.append(game.opponent_deck)
                wins.append(game.won)
                games.append(1)
                gap.append(own - strength.get((game.event, game.opponent), 0.0))
                weight.append(recency_weight(game.date or entry.date, reference, spec.half_life))
        elif entry.wins + entry.losses > 0:
            index.append(position)
            opponent.append(UNKNOWN)
            wins.append(entry.wins)
            games.append(entry.wins + entry.losses)
            gap.append(own)
            weight.append(recency_weight(entry.date, reference, spec.half_life))
    return Rows(np.array(index, dtype=int), opponent, np.array(wins, float), np.array(games, float), np.array(gap, float), np.array(weight, float))


def observed_field(lists: Sequence[Any], games: Sequence[Game] | None, as_of: str | None = None, half_life: float | None = None) -> dict[str, float]:
    """Opponent decks this archetype faced, weighted by the same recency as the model."""
    reference = reference_date(lists, as_of)
    keys = {(entry.event, entry.player) for entry in lists}
    totals: dict[str, float] = {}
    for game in games or ():
        if (game.event, game.player) in keys:
            totals[game.opponent_deck] = totals.get(game.opponent_deck, 0.0) + recency_weight(game.date, reference, half_life)
    whole = sum(totals.values())
    return {deck: share / whole for deck, share in sorted(totals.items())} if whole else {}


def normalise_field(field: Mapping[str, float] | None) -> dict[str, float]:
    positive = {str(deck): float(share) for deck, share in (field or {}).items() if float(share) > 0}
    whole = sum(positive.values())
    return {deck: share / whole for deck, share in sorted(positive.items())} if whole else {}


@dataclass
class GameModel:
    slots: list[Slot]
    slot_means: np.ndarray
    opponents: list[str]
    specific: list[str]
    theta: np.ndarray
    covariance: np.ndarray
    spec: ModelSpec
    extras: dict[str, Any] = dataclass_field(default_factory=dict)

    @property
    def shared(self) -> np.ndarray:
        start = len(self.opponents) + 1
        return self.theta[start:start + len(self.slots)]

    def _blocks(self) -> tuple[int, int]:
        start = len(self.opponents) + 1
        return start, start + len(self.slots)

    def view(self, field: Mapping[str, float] | None) -> tuple[np.ndarray, np.ndarray]:
        """Each slot's effect averaged over `field`, with its covariance."""
        start, end = self._blocks()
        weights = normalise_field(field)
        size = len(self.slots)
        transform = np.zeros((size, len(self.theta)))
        transform[:, start:end] = np.eye(size)
        for block, deck in enumerate(self.specific):
            share = weights.get(deck, 0.0)
            if share:
                offset = end + block * size
                transform[:, offset:offset + size] += share * np.eye(size)
        return transform @ self.theta, transform @ self.covariance @ transform.T

    def against(self, deck: str) -> tuple[np.ndarray, np.ndarray]:
        """Each slot's effect against one opponent deck (its shared effect if the deck has no block)."""
        return self.view({deck: 1.0})

    def design(self, lists: Sequence[Any], rows: Rows) -> sparse.csr_matrix:
        index = {deck: position for position, deck in enumerate(self.opponents)}
        specific = {deck: position for position, deck in enumerate(self.specific)}
        size = len(self.slots)
        matrix = np.array([[entry.counts.get(card, 0) >= copy for card, copy in self.slots] for entry in lists], dtype=float).reshape(len(lists), size) - self.slot_means
        per_row = matrix[rows.list_index] if len(rows.list_index) else np.zeros((0, size))
        intercept = sparse.csr_matrix(
            ([1.0 for deck in rows.opponent if deck in index], ([r for r, deck in enumerate(rows.opponent) if deck in index], [index[deck] for deck in rows.opponent if deck in index])),
            shape=(len(rows.opponent), len(self.opponents)),
        )
        shared = sparse.csr_matrix(per_row)
        parts = [intercept, sparse.csr_matrix(rows.strength[:, None]), shared]
        opponents = np.array(rows.opponent, dtype=object)
        for deck in self.specific:
            parts.append(sparse.diags((opponents == deck).astype(float)) @ shared)
        return sparse.hstack(parts).tocsr()

    def loss(self, lists: Sequence[Any], strength: Mapping[tuple[str, str], float]) -> tuple[float, float]:
        """Unweighted log loss of the given lists' games, and their number of games."""
        rows = build_rows(lists, strength, ModelSpec(prior_sd=self.spec.prior_sd, games=self.spec.games))
        if not len(rows.wins):
            return 0.0, 0.0
        eta = self.design(lists, rows) @ self.theta
        return float(-(rows.wins * eta - rows.games * np.logaddexp(0.0, eta)).sum()), float(rows.games.sum())


def _penalty(opponents: Sequence[str], slots: int, specific: int, spec: ModelSpec) -> np.ndarray:
    intercepts = np.array([1e-4 if deck == UNKNOWN or len(opponents) == 1 else 0.25 for deck in opponents])
    parts = [intercepts, np.array([1e-4]), np.full(slots, 1.0 / spec.prior_sd ** 2)]
    if specific:
        parts.append(np.full(slots * specific, 1.0 / spec.opponent_sd ** 2))
    return np.concatenate(parts)


def fit_game_model(
    lists: Sequence[Any],
    strength: Mapping[tuple[str, str], float],
    slots: Sequence[Slot],
    spec: ModelSpec,
    with_covariance: bool = True,
) -> GameModel:
    """Fit by penalised maximum likelihood; `with_covariance=False` skips the covariance for fits that
    only score held-out loss or price swaps."""
    rows = build_rows(lists, strength, spec)
    opponent_games: dict[str, float] = {}
    for deck, count in zip(rows.opponent, rows.games):
        opponent_games[deck] = opponent_games.get(deck, 0.0) + count
    opponents = sorted(opponent_games) or [UNKNOWN]
    specific = sorted(deck for deck, count in opponent_games.items() if deck != UNKNOWN and count >= BDIF_BEST60_MIN_OPPONENT_GAMES) if spec.opponent_sd > 0 else []
    matrix = np.array([[entry.counts.get(card, 0) >= copy for card, copy in slots] for entry in lists], dtype=float).reshape(len(lists), len(slots))
    model = GameModel(
        slots=list(slots), slot_means=matrix.mean(axis=0) if len(lists) else np.zeros(len(slots)),
        opponents=opponents, specific=specific, theta=np.zeros(0), covariance=np.zeros((0, 0)), spec=spec,
    )
    design = model.design(lists, rows)
    penalty = _penalty(opponents, len(slots), len(specific), spec)
    weighted_wins, weighted_games = rows.weight * rows.wins, rows.weight * rows.games

    def objective(coef: np.ndarray) -> tuple[float, np.ndarray]:
        eta = design @ coef
        probability = 1.0 / (1.0 + np.exp(-eta))
        value = -(weighted_wins * eta - weighted_games * np.logaddexp(0.0, eta)).sum() + 0.5 * (penalty * coef * coef).sum()
        return float(value), -(design.T @ (weighted_wins - weighted_games * probability)) + penalty * coef

    tolerance = 1e-8 if with_covariance else 1e-6
    theta = minimize(objective, np.zeros(design.shape[1]), jac=True, method="L-BFGS-B", options={"maxiter": 5_000, "gtol": tolerance}).x
    model.theta = theta
    if with_covariance:
        probability = 1.0 / (1.0 + np.exp(-(design @ theta)))
        scale = np.sqrt(weighted_games * probability * (1.0 - probability))
        scaled = design.multiply(scale[:, None]).tocsr()
        information = (scaled.T @ scaled).toarray() + np.diag(penalty)
        model.covariance = np.linalg.inv(information)
    else:
        model.covariance = np.zeros((len(theta), len(theta)))
    model.extras["games"] = float(rows.games.sum())
    return model


def recent_split(lists: Sequence[Any], as_of: str | None, days: float) -> tuple[list[int], list[int]]:
    """List positions before and within the last `days` days before the reference date."""
    reference = reference_date(lists, as_of)
    if reference is None:
        return list(range(len(lists))), []
    early, late = [], []
    for position, entry in enumerate(lists):
        when = parse_date(entry.date)
        (late if when is not None and (reference - when).total_seconds() / 86_400.0 <= days else early).append(position)
    return early, late
