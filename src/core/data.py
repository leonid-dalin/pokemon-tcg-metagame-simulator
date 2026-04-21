#!/usr/bin/env python3
# data.py | Matrix loading, validation, and clustering
from __future__ import annotations

import json
import structlog
from collections import defaultdict
from typing import Dict, List, Tuple, Any

import numpy as np
from opentelemetry import trace
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import pairwise_distances, silhouette_score
from sklearn.preprocessing import StandardScaler
from src.core.config import RNG_SEED, MIN_GAMES

# Initialise tracer and structured logger
tracer = trace.get_tracer(__name__)
logger = structlog.get_logger()


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
        file_path: str, min_matches_required: int = MIN_GAMES
) -> Tuple[List[str], np.ndarray, Dict[Tuple[str, str], Dict[str, Any]]]:
    """Load and preprocess archetype matchup data from JSON with tracing."""
    with tracer.start_as_current_span("load_matchup_data") as span:
        span.set_attribute("file.path", file_path)
        logger.info("loading_matchup_data", path=file_path)

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        archetypes: List[str] = list(data.get("archetypes", []))
        raw_win = data.get("win_rate_matrix", {})
        if not archetypes:
            raise ValueError("No archetypes found in data file.")

        matchup_details: Dict[Tuple[str, str], Dict[str, Any]] = {}

        # Populate matchup details
        with tracer.start_as_current_span("parse_and_filter_matchups"):
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
                            val_any: Any = raw_val
                            wr = float(val_any)
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

        logger.info("archetypes_loaded", count=len(archetypes))
        logger.info("threshold_met", count=len(reliable_decks), threshold=min_matches_required)
        if excluded:
            logger.info("archetypes_excluded", count=len(excluded), decks=excluded[:5])

        n = len(reliable_decks)
        if n == 0:
            return [], np.zeros((0, 0)), {}

        # Build win matrix
        with tracer.start_as_current_span("build_win_matrix"):
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
            max_asymmetry = float(np.max(asymmetry))
            if max_asymmetry >= 0.2 - 1e-5:
                logger.warning(
                    "high_asymmetry_detected",
                    max_asymmetry=max_asymmetry,
                    detail="This is normal for real-world data."
                )

        logger.info("win_matrix_built", size=n, detail="Diagonal enforced to 0.5.")
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
    with tracer.start_as_current_span("cluster_decks_by_matchup_profile") as span:
        span.set_attribute("method", method)
        n_samples = len(deck_names)

        if n_samples < 2:
            return {
                "labels": [0] * n_samples,
                "centroids": None,
                "distances": None,
                "method": method,
                "n_clusters": 1 if n_samples == 1 else 0,
            }

        scaler = StandardScaler()
        wm_scaled = scaler.fit_transform(win_matrix)
        distances = pairwise_distances(wm_scaled, metric="euclidean")

        max_possible_k = min(6, n_samples - 1) if n_samples > 2 else 2

        if method == "kmeans":
            best_k = min(5, n_samples) if isinstance(n_clusters, str) else n_clusters
            best_labels, best_centroids = None, None

            if n_clusters == "auto" and n_samples > 2:
                best_score = -1.0
                for k in range(2, max_possible_k + 1):
                    kmeans = KMeans(n_clusters=k, random_state=RNG_SEED, n_init=10)
                    labels = kmeans.fit_predict(wm_scaled)
                    score = silhouette_score(wm_scaled, labels)

                    if score > best_score:
                        best_score = score
                        best_k = k
                        best_labels = labels
                        best_centroids = kmeans.cluster_centers_

                logger.info("silhouette_optimization", best_k=best_k, score=float(best_score))
            else:
                kmeans = KMeans(n_clusters=best_k, random_state=RNG_SEED, n_init=10)
                best_labels = kmeans.fit_predict(wm_scaled)
                best_centroids = kmeans.cluster_centers_

            labels, centroids, final_k = best_labels, best_centroids, best_k

        elif method == "hierarchical":
            final_k = 5 if n_clusters == "auto" else int(n_clusters)
            hc = AgglomerativeClustering(n_clusters=final_k, linkage="ward")
            labels = hc.fit_predict(wm_scaled)
            centroids = None
        else:
            raise ValueError(f"Unsupported clustering method: {method}")

        if labels is None:
            labels = np.zeros(n_samples, dtype=int)

        labels_arr: np.ndarray = np.asarray(labels)
        centroids_arr: np.ndarray | None = np.asarray(centroids) if centroids is not None else None
        distances_arr: np.ndarray = np.asarray(distances)

        cluster_map = {
            "labels": labels_arr.tolist(),
            "centroids": centroids_arr.tolist() if centroids_arr is not None else None,
            "distances": distances_arr.tolist(),
            "method": method,
            "n_clusters": final_k,
        }
        for i in range(final_k):
            members = [deck_names[j] for j in range(len(labels_arr)) if labels_arr.item(j) == i]
            logger.info("cluster_generated", cluster_id=i, member_count=len(members), members=members)

        return cluster_map


# ----------------------------
# Meta Dominance Diagnostic
# ----------------------------
def compute_deck_dominance(win_matrix: np.ndarray, deck_names: List[str]) -> np.ndarray:
    """Compute and log the deck with the highest meta-weighted win rate against the initial uniform field."""
    with tracer.start_as_current_span("compute_deck_dominance"):
        n = len(deck_names)
        if n == 0:
            return np.array([])

        initial_uniform_field = np.full(n, 1.0 / n)

        with tracer.start_as_current_span("matrix_vector_multiplication"):
            meta_weighted_win_rates = win_matrix.dot(initial_uniform_field)

        top_idx = int(np.argmax(meta_weighted_win_rates))
        top_deck = deck_names[top_idx]
        top_mwr = float(meta_weighted_win_rates[top_idx])

        logger.info("dominance_calculated", top_deck=top_deck, win_rate=top_mwr)
        return meta_weighted_win_rates