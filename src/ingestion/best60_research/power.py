"""Data information for candidate Best-60 interactions."""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.stats import norm


def interaction_information(
    nuisance: np.ndarray,
    exposure: np.ndarray,
    *,
    probabilities: np.ndarray,
    sample_weight: np.ndarray | None = None,
    cluster_ids: np.ndarray | None = None,
    confidence: float = 0.95,
    power: float = 0.80,
) -> dict[str, object]:
    """Estimate local logistic information after projecting out nuisance terms.

    The projection uses outcome precision `p * (1 - p)` and optional sampling
    weights. Priors do not enter this calculation. ``mde80`` is a local Wald
    approximation on the log-odds scale, not an exact power guarantee.
    """
    z = sparse.csr_matrix(nuisance, dtype=float) if sparse.issparse(nuisance) else np.asarray(nuisance, dtype=float)
    x = (
        np.asarray(exposure.toarray(), dtype=float).reshape(-1)
        if sparse.issparse(exposure) else np.asarray(exposure, dtype=float).reshape(-1)
    )
    p = np.asarray(probabilities, dtype=float).reshape(-1)
    if z.ndim != 2 or z.shape[0] != len(x) or len(x) != len(p):
        raise ValueError("nuisance, exposure and probabilities must align")
    z_data = z.data if sparse.issparse(z) else z
    if not np.isfinite(z_data).all() or not np.isfinite(x).all() or not np.isfinite(p).all():
        raise ValueError("design and probabilities must be finite")
    if ((p < 0) | (p > 1)).any():
        raise ValueError("probabilities must be in [0, 1]")
    if not 0 < confidence < 1 or not 0 < power < 1:
        raise ValueError("confidence and power must be between zero and one")
    if sample_weight is None:
        sampling = np.ones(len(x), dtype=float)
    else:
        sampling = np.asarray(sample_weight, dtype=float).reshape(-1)
        if len(sampling) != len(x) or not np.isfinite(sampling).all() or (sampling < 0).any():
            raise ValueError("sample_weight must be finite, nonnegative and aligned")
    precision = sampling * p * (1.0 - p)
    active = precision > 0
    rank = 0
    residual = np.zeros_like(x)
    if active.any() and z.shape[1]:
        za = z[active]
        precision_active = precision[active]
        xa = x[active]
        if sparse.issparse(za):
            gram = (za.T @ za.multiply(precision_active[:, None])).toarray()
            rhs = np.asarray(za.T @ (precision_active * xa)).reshape(-1)
        else:
            gram = za.T @ (precision_active[:, None] * za)
            rhs = za.T @ (precision_active * xa)
        eigenvalues, eigenvectors = np.linalg.eigh(gram)
        largest = max(0.0, float(eigenvalues[-1])) if eigenvalues.size else 0.0
        tolerance = largest * max(za.shape) * np.finfo(float).eps
        keep = eigenvalues > tolerance
        rank = int(np.sum(keep))
        if rank:
            beta = eigenvectors[:, keep] @ ((eigenvectors[:, keep].T @ rhs) / eigenvalues[keep])
            fitted = np.asarray(za @ beta).reshape(-1)
        else:
            fitted = np.zeros_like(xa)
        residual[active] = xa - fitted
    elif active.any():
        residual[active] = x[active]
    info = float(np.dot(precision, residual * residual))
    exposure_count = int(np.sum((x != 0) & active))
    if cluster_ids is None:
        cluster_count = int(len(x))
    else:
        clusters = np.asarray(cluster_ids).reshape(-1)
        if len(clusters) != len(x):
            raise ValueError("cluster_ids must align with exposure")
        cluster_count = int(len(set(clusters[active].tolist())))
    weighted_mean = float(np.average(residual[active], weights=precision[active])) if active.any() else 0.0
    residual_variance = float(np.average((residual[active] - weighted_mean) ** 2, weights=precision[active])) if active.any() else 0.0
    aliased = info <= np.finfo(float).eps * max(1.0, float(np.dot(precision, x * x))) * 100
    zcrit = norm.ppf(0.5 + confidence / 2) + norm.ppf(power)
    mde = None if aliased else float(zcrit / np.sqrt(info))
    return {
        "status": "not_identified" if aliased else "identified",
        "information": info,
        "rank": rank,
        "residual_variance": residual_variance,
        "exposure_count": exposure_count,
        "cluster_count": cluster_count,
        "mde80": mde,
        "mde_is_local_approximation": True,
        "large_effect_approximation": bool(mde is not None and mde > 1.0),
        "aliases": ["exposure is in the weighted nuisance column space"] if aliased else [],
    }
