#!/usr/bin/env python3
# data.py | Matrix loading, validation, and clustering
from __future__ import annotations

import json
import logging
from collections import defaultdict
from typing import Dict, List, Tuple, Any

import numpy as np

# ----------------------------
# Core Utilities
# ----------------------------
def safe_normalize(vec: np.ndarray) -> np.ndarray:
    """Normalise a vector to sum to 1.0. If sum is zero, returns uniform distribution."""
    s = vec.sum()
    if s <= 0:
        n = len(vec)
        return np.ones(n, dtype=float) / n
    return vec / s

# ----------------------------
# Matchup Data Loader
# ----------------------------
def load_matchup_data(
    file_path: str, min_matches_required: int = 700
) -> Tuple[List[str], np.ndarray, Dict[Tuple[str, str], Dict[str, Any]]]:
    """Load and preprocess archetype matchup data from JSON."""
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
            wr, match_count = 0.5, 0 
            raw_val = row.get(b, {}) if isinstance(row, dict) else row
            
            if isinstance(raw_val, dict):
                wr = float(raw_val.get("win_rate", 0.5))
                match_count = int(raw_val.get("match_count", 0))
            else:
                try:
                    wr = float(raw_val)
                except (ValueError, TypeError):
                    wr = 0.5
            
            match_count = max(0, match_count)
            matchup_details[(a, b)] = {"win_rate": wr, "match_count": match_count}

    # Compute total matches per deck
    deck_total_matches = defaultdict(int)
    for (d1, d2), rec in matchup_details.items():
        deck_total_matches[d1] += rec["match_count"]

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
    for i, d1 in enumerate(reliable_decks):
        for j, d2 in enumerate(reliable_decks):
            win_matrix[i, j] = matchup_details.get((d1, d2), {"win_rate": 0.5})["win_rate"]

    np.fill_diagonal(win_matrix, 0.5)

    # Final validation
    if not np.allclose(np.diag(win_matrix), 0.5, atol=1e-5):
        raise ValueError("Diagonal of win matrix must be exactly 0.5.")
    if not np.all((win_matrix >= 0.0) & (win_matrix <= 1.0)):
        raise ValueError("Win rates must be between 0.0 and 1.0.")

    asymmetry = np.abs(win_matrix + win_matrix.T - 1.0)
    max_asymmetry = np.max(asymmetry)
    if max_asymmetry >= 0.2 - 1e-5:
        logging.warning(
            f"⚠️  High asymmetry detected in data (max: {max_asymmetry:.2f}). This is normal for real-world data."
        )
    logging.info(f"✅ Win matrix built: {n}x{n}. Diagonal enforced to 0.5.")
    return reliable_decks, win_matrix, matchup_details

# ----------------------------
# Deck Clustering (Analysis Prep)
# ----------------------------
def cluster_decks_by_matchup_profile(
    win_matrix: np.ndarray,
    deck_names: List[str],
    n_clusters: int | str = "auto",
    method: str = "kmeans",
) -> Dict[str, Any]:
    """Group decks into clusters based on similarity of their matchup vectors."""
    n_samples = len(deck_names)
    
    if n_samples < 2:
        return {
            "labels": [0] * n_samples,
            "centroids": None,
            "distances": None,
            "method": method,
            "n_clusters": 1 if n_samples == 1 else 0,
        }

    try:
        from sklearn.cluster import KMeans, AgglomerativeClustering
        from sklearn.metrics import pairwise_distances, silhouette_score
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        logging.warning("⚠️  scikit-learn not installed. Clustering unavailable.")
        return {"labels": [0] * len(deck_names), "centroids": None, "distances": None}

    scaler = StandardScaler()
    WM_scaled = scaler.fit_transform(win_matrix)
    distances = pairwise_distances(WM_scaled, metric="euclidean")


    max_possible_k = min(6, n_samples - 1) if n_samples > 2 else 2

    if method == "kmeans":
        best_k = min(5, n_samples) if isinstance(n_clusters, str) else n_clusters
        best_labels, best_centroids = None, None

        if n_clusters == "auto" and n_samples > 2:
            best_score = -1.0
            for k in range(2, max_possible_k + 1):
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(WM_scaled)
                score = silhouette_score(WM_scaled, labels)
                
                if score > best_score:
                    best_score = score
                    best_k = k
                    best_labels = labels
                    best_centroids = kmeans.cluster_centers_
                    
            logging.info(f"🔍 Silhouette Optimization selected k={best_k} (Score: {best_score:.3f})")
        else:
            kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
            best_labels = kmeans.fit_predict(WM_scaled)
            best_centroids = kmeans.cluster_centers_

        labels, centroids, final_k = best_labels, best_centroids, best_k

    elif method == "hierarchical":
        final_k = 5 if n_clusters == "auto" else int(n_clusters)
        hc = AgglomerativeClustering(n_clusters=final_k, linkage="ward")
        labels = hc.fit_predict(WM_scaled)
        centroids = None
    else:
        raise ValueError(f"Unsupported clustering method: {method}")

    cluster_map = {
        "labels": labels.tolist(),
        "centroids": centroids.tolist() if centroids is not None else None,
        "distances": distances.tolist(),
        "method": method,
        "n_clusters": final_k,
    }
    
    for i in range(final_k):
        members = [deck_names[j] for j in range(len(labels)) if labels[j] == i]
        logging.info(f"🧩 Cluster {i}: {len(members)} decks — {members}")
        
    return cluster_map

# ----------------------------
# Meta Dominance Diagnostic
# ----------------------------
def compute_deck_dominance(win_matrix: np.ndarray, deck_names: List[str]) -> np.ndarray:
    """Compute and log the deck with the highest meta-weighted win rate against the initial uniform field."""
    n = len(deck_names)
    if n == 0:
        return np.array([])

    initial_uniform_field = np.full(n, 1.0 / n)
    meta_weighted_win_rates = win_matrix.dot(initial_uniform_field)

    top_idx = np.argmax(meta_weighted_win_rates)
    top_deck = deck_names[top_idx]
    top_mwr = meta_weighted_win_rates[top_idx]

    logging.info(f"👑 Deck with highest initial growth potential: {top_deck} (Meta-Weighted WR: {top_mwr:.2%})")
    return meta_weighted_win_rates