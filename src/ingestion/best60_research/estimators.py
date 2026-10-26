"""Comparable weighted logistic estimators for the Best-60 study."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import numpy as np
from scipy import sparse
from scipy.optimize import minimize
from scipy.special import expit


_KINDS = {"nuisance", "additive", "pairs", "fm"}
_NUISANCE_PENALTY = 1e-5


class FitBudgetExceeded(TimeoutError):
    """Raised when an optimizer exceeds its per-fit time allowance."""


def _arrays(X: np.ndarray, y: np.ndarray, nuisance: np.ndarray, sample_weight: np.ndarray | None):
    X = sparse.csr_matrix(X, dtype=float) if sparse.issparse(X) else np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    nuisance = sparse.csr_matrix(nuisance, dtype=float) if sparse.issparse(nuisance) else np.asarray(nuisance, dtype=float)
    n_rows = X.shape[0] if sparse.issparse(X) else len(X)
    nuisance_rows = nuisance.shape[0] if sparse.issparse(nuisance) else len(nuisance)
    if X.ndim != 2 or nuisance.ndim != 2 or n_rows != len(y) or n_rows != nuisance_rows:
        raise ValueError("X, y and nuisance must be aligned two-dimensional design arrays")
    x_data = X.data if sparse.issparse(X) else X
    nuisance_data = nuisance.data if sparse.issparse(nuisance) else nuisance
    if not np.isfinite(x_data).all() or not np.isfinite(y).all() or not np.isfinite(nuisance_data).all():
        raise ValueError("model inputs must be finite")
    if not np.isin(y, (0.0, 1.0)).all():
        raise ValueError("y must contain binary outcomes")
    if sample_weight is None:
        weights = np.ones(len(y), dtype=float)
    else:
        weights = np.asarray(sample_weight, dtype=float).reshape(-1)
        if len(weights) != len(y) or not np.isfinite(weights).all() or (weights < 0).any():
            raise ValueError("sample_weight must be finite, nonnegative and aligned with y")
    if weights.sum() <= 0:
        raise ValueError("sample_weight must have positive total weight")
    return X, y, nuisance, weights


def _group_deviation_design(X: np.ndarray, labels: np.ndarray, levels: tuple[Any, ...]):
    columns = []
    for level in levels[1:]:
        mask = (labels == level)[:, None]
        if sparse.issparse(X):
            block = X.multiply(mask).tocsr()
            # Dense row masks leave explicit zeros in SciPy sparse matrices.
            # Retaining them multiplies every objective evaluation by the
            # number of archetypes without adding any information.
            block.eliminate_zeros()
        else:
            block = X * mask
        columns.append(block)
    if sparse.issparse(X):
        return sparse.hstack(columns, format="csr") if columns else sparse.csr_matrix((X.shape[0], 0))
    return np.column_stack(columns) if columns else np.empty((X.shape[0], 0))


def _pair_indices(X: np.ndarray, limit: int = 256) -> tuple[tuple[int, int], ...]:
    p = X.shape[1]
    if p < 2:
        return ()
    if sparse.issparse(X):
        cooccur = (X.T @ X).tocoo()
        valid = [(int(i), int(j), float(value)) for i, j, value in zip(cooccur.row, cooccur.col, cooccur.data) if i < j and value > 0]
        valid.sort(key=lambda item: (-item[2], item[0], item[1]))
        return tuple((i, j) for i, j, _ in valid[:limit])
    cooccur = X.T @ X
    left, right = np.triu_indices(p, k=1)
    counts = np.asarray(cooccur[left, right]).reshape(-1)
    order = np.lexsort((right, left, -counts))[:limit]
    return tuple((int(left[i]), int(right[i])) for i in order if counts[i] > 0)


def _pair_design(X: np.ndarray, pairs: tuple[tuple[int, int], ...] | None = None):
    if pairs is None:
        pairs = tuple(zip(*np.triu_indices(X.shape[1], k=1)))
    if sparse.issparse(X):
        columns = [X[:, i].multiply(X[:, j]) for i, j in pairs]
        return sparse.hstack(columns, format="csr") if columns else sparse.csr_matrix((X.shape[0], 0))
    return np.column_stack([X[:, i] * X[:, j] for i, j in pairs]) if pairs else np.empty((X.shape[0], 0))


def _fm_objective_core(
    theta: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    nuisance: np.ndarray,
    *,
    rank: int,
    penalty: float,
    sample_weight: np.ndarray,
    deviation_design: np.ndarray | None = None,
    main_effect_penalty: float | None = None,
    validated: bool = False,
    normalised_weights: bool = False,
    squared_design: np.ndarray | None = None,
) -> tuple[float, np.ndarray]:
    if validated:
        weights = sample_weight
    else:
        X, y, nuisance, weights = _arrays(X, y, nuisance, sample_weight)
    if rank < 1 or penalty < 0 or not np.isfinite(penalty):
        raise ValueError("rank must be positive and penalty finite and nonnegative")
    n, p = X.shape
    q = nuisance.shape[1]
    deviations = None if deviation_design is None else (
        sparse.csr_matrix(deviation_design, dtype=float)
        if sparse.issparse(deviation_design) else np.asarray(deviation_design, dtype=float)
    )
    if deviations is not None and (deviations.ndim != 2 or deviations.shape[0] != n):
        raise ValueError("deviation_design must be a two-dimensional array aligned with X")
    d = 0 if deviations is None else deviations.shape[1]
    main_penalty = penalty if main_effect_penalty is None else main_effect_penalty
    if main_penalty < 0 or not np.isfinite(main_penalty):
        raise ValueError("main_effect_penalty must be finite and nonnegative")
    expected = q + p + d + p * rank
    theta = np.asarray(theta, dtype=float).reshape(-1)
    if theta.size != expected or not np.isfinite(theta).all():
        raise ValueError(f"theta must contain {expected} finite parameters")
    beta_n = theta[:q]
    beta_x = theta[q : q + p]
    beta_dev = theta[q + p : q + p + d]
    factors = theta[q + p + d :].reshape(p, rank)
    latent = X @ factors
    squared = squared_design if squared_design is not None else (
        X.multiply(X) if sparse.issparse(X) else X * X
    )
    interactions = 0.5 * np.sum(latent * latent - squared @ (factors * factors), axis=1)
    logits = nuisance @ beta_n + X @ beta_x + interactions
    if d:
        logits = logits + deviations @ beta_dev
    normalised = weights if normalised_weights else weights / weights.sum()
    residual = (expit(logits) - y) * normalised
    loss = float(np.dot(normalised, np.logaddexp(0.0, logits) - y * logits))
    if q > 1:
        loss += 0.5 * _NUISANCE_PENALTY * float(np.dot(beta_n[1:], beta_n[1:]))
    loss += 0.5 * main_penalty * float(np.dot(beta_x, beta_x) + np.dot(beta_dev, beta_dev))
    loss += 0.5 * penalty * float(np.sum(factors * factors))
    gradient = np.empty_like(theta)
    gradient[:q] = nuisance.T @ residual
    if q > 1:
        gradient[1:q] += _NUISANCE_PENALTY * beta_n[1:]
    gradient[q : q + p] = np.asarray(X.T @ residual).reshape(-1) + main_penalty * beta_x
    if d:
        gradient[q + p : q + p + d] = np.asarray(deviations.T @ residual).reshape(-1) + main_penalty * beta_dev
    factor_gradient = X.T @ (residual[:, None] * latent) - squared.T @ residual[:, None] * factors
    gradient[q + p + d :] = (factor_gradient + penalty * factors).reshape(-1)
    return loss, gradient


def fm_objective(
    theta: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    nuisance: np.ndarray,
    *,
    rank: int,
    penalty: float,
    sample_weight: np.ndarray | None = None,
) -> tuple[float, np.ndarray]:
    """Return the weighted FM loss and gradient in the public parameter layout."""
    return _fm_objective_core(
        theta, X, y, nuisance, rank=rank, penalty=penalty,
        sample_weight=sample_weight, validated=False,
    )


@dataclass
class FittedModel:
    kind: str
    nuisance_coef: np.ndarray
    main_coef: np.ndarray
    interaction_coef: np.ndarray
    pair_indices: tuple[tuple[int, int], ...]
    factors: np.ndarray
    feature_count: int
    group_levels: tuple[Any, ...]
    reference_group: Any
    deviation_coef: np.ndarray
    rank: int
    penalty: float
    main_effect_penalty: float
    seed: int
    converged: bool
    status: int
    message: str
    objective: float
    iterations: int
    gradient_norm: float
    parameter_count: int

    def _group_design(self, X: np.ndarray, groups: np.ndarray | None) -> np.ndarray:
        n_rows = X.shape[0] if sparse.issparse(X) else len(X)
        if self.deviation_coef.size == 0:
            return sparse.csr_matrix((n_rows, 0)) if sparse.issparse(X) else np.empty((n_rows, 0))
        labels = np.asarray(groups).reshape(-1) if groups is not None else np.full(n_rows, None, dtype=object)
        if len(labels) != n_rows:
            raise ValueError("groups must be aligned with X")
        # New groups receive the pooled main effect and no group deviation.
        return _group_deviation_design(X, labels, self.group_levels)

    def predict_logit(self, X: np.ndarray, *, nuisance: np.ndarray, groups: np.ndarray | None = None) -> np.ndarray:
        X = sparse.csr_matrix(X, dtype=float) if sparse.issparse(X) else np.asarray(X, dtype=float)
        nuisance = sparse.csr_matrix(nuisance, dtype=float) if sparse.issparse(nuisance) else np.asarray(nuisance, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.feature_count:
            raise ValueError("X has an incompatible feature shape")
        if nuisance.ndim != 2 or nuisance.shape[0] != X.shape[0] or nuisance.shape[1] != len(self.nuisance_coef):
            raise ValueError("nuisance has an incompatible shape")
        logits = nuisance @ self.nuisance_coef
        if self.kind != "nuisance":
            logits = logits + X @ self.main_coef
            if self.deviation_coef.size:
                logits = logits + self._group_design(X, groups) @ self.deviation_coef
        if self.kind == "pairs":
            logits = logits + _pair_design(X, self.pair_indices) @ self.interaction_coef
        elif self.kind == "fm":
            latent = X @ self.factors
            squared = X.multiply(X) if sparse.issparse(X) else X * X
            logits = logits + 0.5 * np.sum(latent * latent - squared @ (self.factors * self.factors), axis=1)
        if not np.isfinite(logits).all():
            raise ValueError("prediction produced nonfinite logits")
        return logits


def fit_model(
    X: np.ndarray,
    y: np.ndarray,
    *,
    nuisance: np.ndarray,
    groups: np.ndarray | None,
    kind: str,
    rank: int = 2,
    seed: int = 1312,
    penalty: float = 1.0,
    max_iter: int = 5000,
    sample_weight: np.ndarray | None = None,
    max_seconds: float | None = None,
    main_effect_penalty: float | None = None,
) -> FittedModel:
    """Fit a nuisance, additive, pairwise or factorisation-machine model.

    ``max_seconds`` bounds time spent inside the optimizer objective as well as
    the optimizer call. It raises :class:`FitBudgetExceeded` when exhausted.
    """
    started = time.monotonic()
    if max_seconds is not None and (max_seconds < 0 or not np.isfinite(max_seconds)):
        raise ValueError("max_seconds must be finite and nonnegative")

    def check_budget() -> None:
        if max_seconds is not None and time.monotonic() - started >= max_seconds:
            raise FitBudgetExceeded(f"fit exceeded its {max_seconds:.3f}-second budget")

    if kind not in _KINDS:
        raise ValueError(f"kind must be one of {sorted(_KINDS)}")
    X, y, nuisance, weights = _arrays(X, y, nuisance, sample_weight)
    check_budget()
    if main_effect_penalty is None:
        main_effect_penalty = penalty
    if (penalty < 0 or not np.isfinite(penalty) or main_effect_penalty < 0
            or not np.isfinite(main_effect_penalty) or rank < 1 or max_iter < 1):
        raise ValueError("penalty, rank or max_iter is invalid")
    labels = np.asarray(groups).reshape(-1) if groups is not None else np.full(len(y), None, dtype=object)
    if len(labels) != len(y):
        raise ValueError("groups must be aligned with X")
    levels = tuple(sorted(set(labels.tolist()), key=repr))
    if not levels:
        levels = (None,)
    deviations = (
        _group_deviation_design(X, labels, levels)
        if kind != "nuisance" else np.empty((len(y), 0))
    )
    n, p = X.shape
    q = nuisance.shape[1]
    candidate_pairs = _pair_indices(X) if kind == "pairs" else ()
    pair_x = _pair_design(X, candidate_pairs) if kind == "pairs" else np.empty((n, 0))
    r = rank if kind == "fm" else 0
    # Parameters: nuisance, pooled main effects, group deviations, then pair
    # coefficients or pooled embeddings. Group deviations receive the same
    # declared ridge penalty as other non-nuisance terms.
    total = q + (p if kind != "nuisance" else 0) + deviations.shape[1]
    if kind == "pairs":
        total += pair_x.shape[1]
    if kind == "fm":
        total += p * r
    rng = np.random.default_rng(seed)
    initial = np.zeros(total, dtype=float)
    if kind == "fm" and p:
        initial[-p * r :] = rng.normal(0.0, 0.08, p * r)

    n_main = p if kind != "nuisance" else 0
    d = deviations.shape[1]
    pair_count = pair_x.shape[1]
    normalised_weights = weights / weights.sum()
    squared_X = (
        X.multiply(X) if sparse.issparse(X) else X * X
    ) if kind == "fm" else None

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        check_budget()
        if kind == "fm":
            if d == 0 and main_effect_penalty == penalty:
                value_gradient = fm_objective(
                    theta, X, y, nuisance, rank=r, penalty=penalty, sample_weight=weights,
                )
            else:
                value_gradient = _fm_objective_core(
                    theta, X, y, nuisance, rank=r, penalty=penalty, sample_weight=normalised_weights,
                    deviation_design=deviations, main_effect_penalty=main_effect_penalty,
                    validated=True, normalised_weights=True, squared_design=squared_X,
                )
            check_budget()
            return value_gradient
        beta_n = theta[:q]
        cursor = q
        beta_x = theta[cursor : cursor + n_main]
        cursor += n_main
        beta_dev = theta[cursor : cursor + d]
        cursor += d
        beta_pair = theta[cursor : cursor + pair_count]
        cursor += pair_count
        factors = theta[cursor:].reshape(p, r) if r else np.empty((p, 0))
        logits = nuisance @ beta_n
        if n_main:
            logits += X @ beta_x
        if d:
            logits += deviations @ beta_dev
        if pair_count:
            logits += pair_x @ beta_pair
        latent = X @ factors if r else np.empty((n, 0))
        if r:
            logits += 0.5 * np.sum(latent * latent - squared_X @ (factors * factors), axis=1)
        residual = (expit(logits) - y) * normalised_weights
        value = float(np.dot(normalised_weights, np.logaddexp(0.0, logits) - y * logits))
        if q > 1:
            value += 0.5 * _NUISANCE_PENALTY * float(np.dot(beta_n[1:], beta_n[1:]))
        value += 0.5 * main_effect_penalty * float(np.dot(beta_x, beta_x) + np.dot(beta_dev, beta_dev))
        value += 0.5 * penalty * float(np.dot(beta_pair, beta_pair) + np.sum(factors * factors))
        grad = np.empty_like(theta)
        grad[:q] = nuisance.T @ residual
        if q > 1:
            grad[1:q] += _NUISANCE_PENALTY * beta_n[1:]
        cursor = q
        if n_main:
            grad[cursor : cursor + p] = np.asarray(X.T @ residual).reshape(-1) + main_effect_penalty * beta_x
            cursor += p
        if d:
            grad[cursor : cursor + d] = np.asarray(deviations.T @ residual).reshape(-1) + main_effect_penalty * beta_dev
            cursor += d
        if pair_count:
            grad[cursor : cursor + pair_count] = np.asarray(pair_x.T @ residual).reshape(-1) + penalty * beta_pair
            cursor += pair_count
        if r:
            factor_grad = X.T @ (residual[:, None] * latent) - squared_X.T @ residual[:, None] * factors
            grad[cursor:] = (factor_grad + penalty * factors).reshape(-1)
        check_budget()
        return value, grad

    check_budget()
    result = minimize(
        objective,
        initial,
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": int(max_iter), "ftol": 1e-9, "gtol": 1e-5},
    )
    check_budget()
    theta = np.asarray(result.x)
    cursor = q
    main = theta[cursor : cursor + n_main] if n_main else np.zeros(p)
    cursor += n_main
    dev = theta[cursor : cursor + d]
    cursor += d
    pair_coef = theta[cursor : cursor + pair_count]
    cursor += pair_count
    factors = theta[cursor:].reshape(p, r) if r else np.empty((p, 0))
    return FittedModel(
        kind=kind,
        nuisance_coef=theta[:q].copy(),
        main_coef=main.copy(),
        interaction_coef=pair_coef.copy(),
        pair_indices=candidate_pairs,
        factors=factors.copy(),
        feature_count=p,
        group_levels=levels,
        reference_group=levels[0],
        deviation_coef=dev.copy(),
        rank=r,
        penalty=float(penalty),
        main_effect_penalty=float(main_effect_penalty),
        seed=int(seed),
        converged=bool(result.success),
        status=int(result.status),
        message=str(result.message),
        objective=float(result.fun),
        iterations=int(result.nit),
        gradient_norm=float(np.linalg.norm(result.jac)),
        parameter_count=int(theta.size),
    )


def predict_logit(model: FittedModel, X: np.ndarray, *, nuisance: np.ndarray, groups: np.ndarray | None = None) -> np.ndarray:
    """Convenience wrapper matching the fitted model prediction contract."""
    return model.predict_logit(X, nuisance=nuisance, groups=groups)
