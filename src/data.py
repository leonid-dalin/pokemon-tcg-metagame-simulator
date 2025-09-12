#!/usr/bin/env python3
# data.py — Load, clean, validate, and preprocess metagame matchup data.

from __future__ import annotations
import json
import logging
from pydantic import BaseModel, Field, validator, ValidationError
from typing import Dict, Union, List, Tuple, Dict, Any, Optional, Literal
from collections import defaultdict
import numpy as np

class MatchupDetail(BaseModel):
    win_rate: float = Field(..., ge=0.0, le=1.0)
    match_count: int = Field(..., ge=1)

class ArchetypeData(BaseModel):
    archetypes: List[str]
    win_rate_matrix: Dict[str, Dict[str, MatchupDetail]]
    
    @validator('win_rate_matrix')
    def validate_matrix(cls, v, values):
        archetypes = values.get('archetypes', [])
        # Ensure every archetype has a row and every row has entries for all archetypes
        for archetype in archetypes:
            if archetype not in v:
                raise ValueError(f"Missing row for archetype: {archetype}")
            row = v[archetype]
            for opp in archetypes:
                if opp not in row:
                    raise ValueError(f"Missing matchup for {archetype} vs {opp}")
        return v

# ----------------------------
# Core Utilities
# ----------------------------
def safe_normalize(vec: np.ndarray) -> np.ndarray:
    """Normalise a vector to sum to 1.0. If sum is zero, returns uniform distribution.

    Args:
        vec (np.ndarray): Input vector of probabilities or frequencies.

    Returns:
        np.ndarray: Normalised vector summing to 1.0.
    """
    s = vec.sum()
    if s <= 0:
        n = len(vec)
        return np.ones(n, dtype=float) / n
    return vec / s

# ----------------------------
# Matchup Data Loader
# ----------------------------
def load_matchup_data(
    file_path: str,
    min_matches_required: int = 700,
    symmetry_tolerance: float = 1e-5
) -> Tuple[List[str], np.ndarray, Dict[Tuple[str, str], Dict[str, Any]]]:
    """Load and preprocess archetype matchup data from JSON.
    Performs:
        - Deck filtering by total match volume.
        - Validation of diagonal and probabilistic bounds.
        - (Removed) Win-rate matrix symmetrisation.
    Args:
        file_path (str): Path to the JSON matchup data file.
        min_matches_required (int): Minimum total matches for a deck to be included.
        symmetry_tolerance (float): (No longer used for symmetrisation, kept for potential future warnings).
    Returns:
        Tuple containing:
            - List[str]: Names of reliable archetypes.
            - np.ndarray: Win-rate matrix (n x n). **NOT symmetric.**
            - Dict: Raw matchup details for Bayesian sampling.
    Raises:
        FileNotFoundError: If the input file is missing.
        ValueError: If matrix properties are violated.
    """
    logging.info(f"📂 Loading matchup data from: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    archetypes: List[str] = list(data.get("archetypes", []))
    raw_win = data.get("win_rate_matrix", {})
    if not archetypes:
        raise ValueError("No archetypes found in data file.")
    matchup_details: Dict[Tuple[str, str], Dict[str, Any]] = {}
    # Populate matchup details
    for a in archetypes:
        row = raw_win.get(a, {})
        for b in archetypes:
            wr, match_count = 0.5, 100
            raw_val = row.get(b, {}) if isinstance(row, dict) else row
            if isinstance(raw_val, dict):
                wr = float(raw_val.get("win_rate", 0.5))
                match_count = int(raw_val.get("match_count", 100))
            else:
                try:
                    wr = float(raw_val)
                except (ValueError, TypeError):
                    wr = 0.5
            match_count = max(1, match_count)
            matchup_details[(a, b)] = {"win_rate": wr, "match_count": match_count}
    # Compute total matches per deck
    deck_total_matches = defaultdict(int)
    for (d1, d2), rec in matchup_details.items():
        deck_total_matches[d1] += rec["match_count"]
        if d1 != d2:
            deck_total_matches[d2] += rec["match_count"]
    # Filter decks
    reliable_decks = [d for d in archetypes if deck_total_matches[d] >= min_matches_required]
    excluded = sorted([d for d in archetypes if d not in reliable_decks])
    logging.info(f"📊 Loaded {len(archetypes)} archetypes.")
    logging.info(f"✅ {len(reliable_decks)} meet minimum match threshold ({min_matches_required}).")
    if excluded:
        logging.info(f"❌ Excluded {len(excluded)}: {excluded[:5]}{'...' if len(excluded) > 5 else ''}")
    n = len(reliable_decks)
    if n == 0:
        return [], np.zeros((0, 0)), {}
    # Build win matrix
    win_matrix = np.zeros((n, n), dtype=float)
    deck_to_idx = {deck: i for i, deck in enumerate(reliable_decks)}
    for i, d1 in enumerate(reliable_decks):
        for j, d2 in enumerate(reliable_decks):
            win_matrix[i, j] = matchup_details.get((d1, d2), {"win_rate": 0.5})["win_rate"]
    # --- FIX: Only enforce diagonal = 0.5. DO NOT SYMMETRISE. ---
    np.fill_diagonal(win_matrix, 0.5)
    # --- END OF FIX ---
    # Final validation
    if not np.allclose(np.diag(win_matrix), 0.5, atol=1e-5):
        raise ValueError("Diagonal of win matrix must be exactly 0.5.")
    if not np.all((win_matrix >= 0.0) & (win_matrix <= 1.0)):
        raise ValueError("Win rates must be between 0.0 and 1.0.")
    # --- Optional: Log a warning if asymmetry is very high, but don't fix it. ---
    asymmetry = np.abs(win_matrix + win_matrix.T - 1.0)
    max_asymmetry = np.max(asymmetry)
    if max_asymmetry > 0.2:  # Arbitrary threshold for "very high"
        logging.warning(f"⚠️  High asymmetry detected in data (max: {max_asymmetry:.2f}). This is normal for real-world data.")
    logging.info(f"✅ Win matrix built: {n}x{n}. Diagonal enforced to 0.5.")
    return reliable_decks, win_matrix, matchup_details


# ----------------------------
# Deck Clustering (Analysis Prep)
# ----------------------------
def cluster_decks_by_matchup_profile(
    win_matrix: np.ndarray,
    deck_names: List[str],
    n_clusters: int = 5,
    method: Literal["kmeans", "hierarchical"] = "kmeans"
) -> Dict[str, Any]:
    """Group decks into clusters based on similarity of their matchup vectors.

    Useful for identifying “archetype families” — e.g., all control decks, all aggro decks.

    Args:
        win_matrix (np.ndarray): n x n win-rate matrix.
        deck_names (List[str]): List of deck names.
        n_clusters (int): Number of clusters to form.
        method (str): Clustering algorithm — "kmeans" or "hierarchical".

    Returns:
        Dict containing:
            - "labels": Cluster assignment per deck.
            - "centroids": (for kmeans) Representative win profiles.
            - "distances": Distance matrix used for clustering.
    """
    try:
        from sklearn.cluster import KMeans, AgglomerativeClustering
        from sklearn.metrics import pairwise_distances
    except ImportError:
        logging.warning("⚠️  scikit-learn not installed. Clustering unavailable.")
        return {"labels": [0] * len(deck_names), "centroids": None, "distances": None}

    # Use 1D win profiles as features (each deck = vector of win rates vs others)
    X = win_matrix.copy()

    # Compute distance matrix (cosine or Euclidean)
    distances = pairwise_distances(X, metric="euclidean")

    if method == "kmeans":
        # ✅ FIX: Explicitly specify n_init as int — Pyright was confused by stubs
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)  # type: ignore
        labels = kmeans.fit_predict(X)
        centroids = kmeans.cluster_centers_
    elif method == "hierarchical":
        hc = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
        labels = hc.fit_predict(X)
        centroids = None
    else:
        raise ValueError(f"Unsupported clustering method: {method}")

    cluster_map = {
        "labels": labels.tolist(),
        "centroids": centroids.tolist() if centroids is not None else None,
        "distances": distances.tolist(),
        "method": method,
        "n_clusters": n_clusters
    }

    # Log cluster summaries
    for i in range(n_clusters):
        members = [deck_names[j] for j in range(len(labels)) if labels[j] == i]
        logging.info(f"🧩 Cluster {i}: {len(members)} decks — {members}")


    return cluster_map


# ----------------------------
# Meta Dominance Diagnostic
# ----------------------------
def compute_deck_dominance(win_matrix: np.ndarray, deck_names: List[str]) -> np.ndarray:
    """Compute and log the deck with the highest meta-weighted win rate against the initial uniform field.
    This predicts which deck has the highest growth potential at the start of the simulation.
    Args:
        win_matrix (np.ndarray): Symmetric win-rate matrix.
        deck_names (List[str]): Deck names (for logging).
    Returns:
        np.ndarray: Meta-weighted win rates per deck against the initial uniform field.
    """
    n = len(deck_names)
    if n == 0:
        return np.array([])

    # Calculate meta-weighted win rate against the initial uniform field
    # This is the expected payoff for each deck at t=0.
    initial_uniform_field = np.full(n, 1.0 / n)
    meta_weighted_win_rates = win_matrix.dot(initial_uniform_field)

    # Find the deck with the highest meta-weighted win rate
    top_idx = np.argmax(meta_weighted_win_rates)
    top_deck = deck_names[top_idx]
    top_mwr = meta_weighted_win_rates[top_idx]

    logging.info(f"👑 Deck with highest initial growth potential: {top_deck} (Meta-Weighted WR: {top_mwr:.2%})")

    return meta_weighted_win_rates