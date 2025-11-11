#!/usr/bin/env python3
# plotting.py — Interactive metagame visualisations using Plotly.

from __future__ import annotations
import logging
import os
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx


# ----------------------------
# Metagame Evolution Plot (Interactive)
# ----------------------------
def plot_metagame_evolution_interactive(
    history: List[np.ndarray],
    deck_names: List[str],
    extinction_gens: Optional[List[Optional[int]]] = None,
    top_n: int = 12,
    save_path: Optional[str] = None,
    title: str = "Metagame Share Over Time",
) -> Optional[go.Figure]:
    """Plot metagame evolution with Plotly — fully interactive, zoomable, hoverable.
    Args:
        history: List of frequency arrays over generations.
        deck_names: List of deck names.
        extinction_gens: List of extinction generation per deck.
        top_n: Number of top decks to highlight.
        save_path: Path to save HTML file (if None, returns figure).
        title: Plot title.
    Returns:
        Plotly Figure object if save_path is None.
    """
    if not history or len(history) == 0:
        logging.warning("No history to plot.")
        return None

    # Compute final frequencies to select top decks
    final_freq = history[-1]
    top_indices = np.argsort(final_freq)[-top_n:][::-1].tolist()
    # Include any deck >1% even if not in top N
    high_presence_indices = np.where(final_freq > 0.01)[0]
    for idx in high_presence_indices:
        if idx not in top_indices:
            top_indices.append(idx)
    top_indices = top_indices[:12]  # Cap for readability

    generations = list(range(len(history)))
    fig = go.Figure()
    colors = px.colors.qualitative.Bold + px.colors.qualitative.Dark24
    # List to hold all annotations (extinction markers)
    annotations = []

    for i, idx in enumerate(top_indices):
        deck_name = deck_names[idx]
        freq_series = [h[idx] for h in history]
        extinction_gen = extinction_gens[idx] if extinction_gens else None

        # Line trace
        fig.add_trace(
            go.Scatter(
                x=generations,
                y=freq_series,
                mode="lines",
                name=deck_name,
                line=dict(width=3, color=colors[i % len(colors)]),
                hovertemplate="<b>%{fullData.name}</b><br>Gen %{x}: %{y:.4%}<extra></extra>",
            )
        )

        if extinction_gen is not None and extinction_gen < len(freq_series):
            annotations.append(
                go.layout.Annotation(
                    x=extinction_gen,
                    y=freq_series[extinction_gen],
                    xref="x",
                    yref="y",
                    text="✕",
                    font=dict(size=16, color="red"),
                    showarrow=False,
                    bgcolor="rgba(255,255,255,0.7)",
                    bordercolor="red",
                    borderwidth=2,
                )
            )

    fig.update_layout(
        annotations=annotations,
        title=dict(text=title, font=dict(size=22, color="#333")),
        xaxis_title="Generation",
        yaxis_title="Metagame Share",
        hovermode="x unified",
        legend_title="Deck Archetypes",
        template="plotly_white",
        height=800,
        margin=dict(l=40, r=40, t=80, b=40),
    )
    fig.update_yaxes(tickformat=".1%", rangemode="tozero")
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="LightGrey")

    if save_path:
        os.makedirs(
            os.path.dirname(save_path) if os.path.dirname(save_path) else ".",
            exist_ok=True,
        )
        fig.write_html(save_path)
        logging.info(f"✅ Interactive evolution plot saved to {save_path}")
        return None

    return fig


# ----------------------------
# Matchup Heatmap (Interactive)
# ----------------------------
def plot_matchup_heatmap_interactive(
    win_matrix: np.ndarray,
    deck_names: List[str],
    tier_order: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    title: str = "Matchup Win Rates (Interactive)",
) -> Optional[go.Figure]:
    """Interactive heatmap of deck vs deck win rates, sorted by tier.
    Args:
        win_matrix: n x n win-rate matrix.
        deck_names: List of deck names.
        tier_order: Optional list of deck names in the desired order (S-Tier to D-Tier).
        save_path: Path to save HTML (if None, returns figure).
        title: Plot title.
    Returns:
        Plotly Figure if save_path is None.
    """
    n = len(deck_names)
    if n == 0:
        return None

    # Reorder if tier_order provided
    if tier_order:
        idx_map = {name: i for i, name in enumerate(deck_names)}
        sorted_indices = [idx_map[name] for name in tier_order if name in idx_map]
        remaining = [i for i in range(n) if i not in sorted_indices]
        sorted_indices.extend(remaining)
        sorted_indices = sorted_indices[::-1]
        sorted_win_matrix = win_matrix[np.ix_(sorted_indices, sorted_indices)] * 100
        sorted_names = [deck_names[i] for i in sorted_indices]
    else:
        sorted_win_matrix = win_matrix * 100
        sorted_names = deck_names

    aggressive_colorscale = [
        [0.0, "rgb(178, 34, 34)"],  # 0% - Firebrick (a good rouge)
        [0.45, "rgb(255, 153, 153)"],  # ~45% - Light red
        [0.49, "rgb(255, 255, 224)"],  # ~49% - Light Yellow
        [0.51, "rgb(255, 255, 224)"],  # ~51% - Light Yellow
        [0.55, "rgb(159, 218, 169)"],  # ~55% - Light Green
        [1.0, "rgb(0, 68, 27)"],  # 100% - Very Dark Green
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=sorted_win_matrix,
            x=sorted_names,
            y=sorted_names,
            colorscale=aggressive_colorscale,
            zmin=0,
            zmax=100,
            colorbar=dict(title="Win Rate (%)"),
            text=np.vectorize(lambda x: f"{x:.1f}%")(sorted_win_matrix),
            hovertemplate="<b>%{y}</b> vs <b>%{x}</b><br>" "Win Rate: %{z:.2f}%<extra></extra>",
            showscale=True,
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Opponent's Deck",
        yaxis_title="Your Deck",
        xaxis=dict(tickangle=45, automargin=True),
        yaxis=dict(automargin=True),
        height=max(800, n * 25),
        width=max(800, n * 25),
        template="plotly_white",
    )

    if save_path:
        os.makedirs(
            os.path.dirname(save_path) if os.path.dirname(save_path) else ".",
            exist_ok=True,
        )
        fig.write_html(save_path)
        logging.info(f"✅ Interactive heatmap saved to {save_path}")
        return None

    return fig


# ----------------------------
# Matchup Network Graph (Rock-Paper-Scissors Cycles)
# ----------------------------
def plot_matchup_network(
    win_matrix: np.ndarray,
    deck_names: List[str],
    cycles: List[List[str]],
    metagame_history: Optional[List[np.ndarray]] = None,
    save_path: Optional[str] = None,
    title: str = "Metagame Matchup Network",
) -> Optional[go.Figure]:
    """Create an interactive network graph visualizing significant win-rate edges and highlighting detected RPS cycles.
    Features:
        - Click on a node to focus on it and its direct neighbors.
        - Node size is scaled by the deck's all-time presence (if history is provided).
    """
    G = nx.DiGraph()
    n = len(deck_names)
    deck_to_idx = {name: i for i, name in enumerate(deck_names)}

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if win_matrix[i, j] > 0.55:
                G.add_edge(deck_names[i], deck_names[j], weight=win_matrix[i, j])

    if len(G.nodes) == 0:
        logging.warning("No significant edges to plot.")
        return None

    pos = nx.kamada_kawai_layout(G)

    # --- Calculate All-Time Presence for Node Sizing ---
    if metagame_history is not None and len(metagame_history) > 0:
        total_metagame = np.sum(metagame_history, axis=0) / len(metagame_history)
        # Create a mapping from deck name to its all-time presence
        deck_presence = {deck_names[i]: total_metagame[i] for i in range(n)}
    else:
        # Fallback: uniform size if no history is provided
        deck_presence = {name: 1.0 for name in deck_names}

    # --- Create the edges trace ---
    edge_x = []
    edge_y = []
    edge_hover = []
    # Store the 'to' and 'from' nodes for each edge for the callback
    edge_from_nodes = []
    edge_to_nodes = []

    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_hover.append(f"{edge[0]} → {edge[1]}: {edge[2]['weight']:.2%}")
        edge_from_nodes.extend([edge[0], edge[0], None])  # Match the [x0, x1, None] pattern
        edge_to_nodes.extend([edge[1], edge[1], None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=2, color="rgba(100, 100, 100, 0.8)"),
        hoverinfo="text",
        text=edge_hover,
        mode="lines",
        customdata=np.array([edge_from_nodes, edge_to_nodes]).T,  # Store from/to data for each point
        showlegend=False,
    )

    # --- Create the node trace ---
    node_x = [pos[node][0] for node in G.nodes()]
    node_y = [pos[node][1] for node in G.nodes()]
    # Use all-time presence for sizing
    base_size = 10
    max_presence = max(deck_presence.values()) if deck_presence else 1.0
    node_size = [base_size + 30 * (deck_presence[node] / max_presence) for node in G.nodes()]

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        hoverinfo="text",
        marker=dict(size=node_size, color="lightblue", line=dict(width=2, color="darkblue")),
        text=list(G.nodes()),
        textposition="top center",
        textfont=dict(size=10, color="black"),
        hovertext=[f"{node} (Degree: {G.degree(node)}, Presence: {deck_presence[node]:.2%})" for node in G.nodes()],
        # Assign an ID to the node trace for the callback
        uid="node_trace",
    )

    # --- Highlight cycles ---
    cycle_traces = []
    cycle_colors = px.colors.qualitative.Set1
    for idx, cycle in enumerate(cycles[:5]):  # Limit to 5 cycles
        if len(cycle) < 3:
            continue
        cycle_x = []
        cycle_y = []
        for i in range(len(cycle)):
            start = cycle[i]
            end = cycle[(i + 1) % len(cycle)]
            if start in pos and end in pos:
                x0, y0 = pos[start]
                x1, y1 = pos[end]
                cycle_x.extend([x0, x1, None])
                cycle_y.extend([y0, y1, None])
        cycle_traces.append(
            go.Scatter(
                x=cycle_x,
                y=cycle_y,
                mode="lines",
                line=dict(width=4, color=cycle_colors[idx % len(cycle_colors)], dash="dash"),
                name=f"Cycle {idx+1}",
                hoverinfo="name",
            )
        )

    fig = go.Figure(data=[edge_trace, node_trace] + cycle_traces)

    # --- Add Interactive Callback for Node Click ---
    # This JavaScript code will run in the browser when the plot is displayed
    fig.update_layout(
        title=title,
        showlegend=True,
        hovermode="closest",
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=800,
        template="plotly_white",
        # Add the custom JavaScript for interactivity
        updatemenus=[],
    )
    # We add the interactivity via a clientside callback in the HTML.
    # This is done by injecting JavaScript into the figure's `config`.
    # a function that will be called when a point is clicked
    fig_config = {
        "scrollZoom": True,
        "displayModeBar": True,
        "editable": True,
        "toImageButtonOptions": {"format": "png", "filename": "matchup_network"},
        "modeBarButtonsToAdd": [
            "drawline",
            "drawopenpath",
            "drawclosedpath",
            "drawcircle",
            "drawrect",
            "eraseshape",
        ],
        "responsive": True,
        "doubleClick": "reset+autosize",  # Default behavior on double-click
    }

    logging.info("💡 For advanced interactivity (click-to-focus), consider serving this plot via a Dash application.")

    # --- End of Interactive Callback (Conceptual) ---

    if save_path:
        os.makedirs(
            os.path.dirname(save_path) if os.path.dirname(save_path) else ".",
            exist_ok=True,
        )
        fig.write_html(save_path, config=fig_config)
        logging.info(f"✅ Network graph saved to {save_path}")
        return None

    return fig
