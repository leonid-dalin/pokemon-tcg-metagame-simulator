#!/usr/bin/env python3
# plotting.py | Interactive metagame visualisations using Plotly

from __future__ import annotations
import logging
import os
from typing import List, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
from src.core.config import aggressive_colorscale

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

    final_freq = history[-1]
    top_indices = np.argsort(final_freq)[-top_n:][::-1].tolist()
    high_presence_indices = np.where(final_freq > 0.01)[0]
    for idx in high_presence_indices:
        if idx not in top_indices:
            top_indices.append(idx)
    top_indices = top_indices[:12] # for readability

    generations = list(range(len(history)))
    fig = go.Figure()
    colors = px.colors.qualitative.Bold + px.colors.qualitative.Dark24

    annotations_list = []

    for i, idx in enumerate(top_indices):
        deck_name = deck_names[idx]
        freq_series = [h[idx] for h in history]
        extinction_gen = extinction_gens[idx] if extinction_gens else None

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

        if isinstance(extinction_gen, int) and extinction_gen < len(freq_series):
            annotations_list.append(
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
        annotations=annotations_list,
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
        title=dict(text=title),
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
    graph = nx.DiGraph()
    n = len(deck_names)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if win_matrix[i, j] > 0.55:
                graph.add_edge(deck_names[i], deck_names[j], weight=win_matrix[i, j])

    if len(graph.nodes) == 0:
        logging.warning("No significant edges to plot.")
        return None

    pos = nx.kamada_kawai_layout(graph)

    # --- Calculate All-Time Presence for Node Sizing ---
    if metagame_history is not None and len(metagame_history) > 0:
        total_metagame = np.sum(metagame_history, axis=0) / len(metagame_history)
        deck_presence = {deck_names[i]: total_metagame[i] for i in range(n)}
    else:
        # Fallback: uniform size if no history is provided
        deck_presence = {name: 1.0 for name in deck_names}

    edge_x = []
    edge_y = []
    edge_hover = []

    edge_from_nodes = []
    edge_to_nodes = []

    for edge in graph.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_hover.append(f"{edge[0]} → {edge[1]}: {edge[2]['weight']:.2%}")
        edge_from_nodes.extend([edge[0], edge[0], None])
        edge_to_nodes.extend([edge[1], edge[1], None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=2, color="rgba(100, 100, 100, 0.8)"),
        hoverinfo="text",
        text=edge_hover,
        mode="lines",
        customdata=np.array([edge_from_nodes, edge_to_nodes]).T,
        showlegend=False,
    )

    node_x = [pos[node][0] for node in graph.nodes()]
    node_y = [pos[node][1] for node in graph.nodes()]

    base_size = 10
    max_presence = max(deck_presence.values()) if deck_presence else 1.0
    node_size = [base_size + 30 * (deck_presence[node] / max_presence) for node in graph.nodes()]

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        hoverinfo="text",
        marker=dict(size=node_size, color="lightblue", line=dict(width=2, color="darkblue")),
        text=[str(node) for node in graph.nodes()],
        textposition="top center",
        textfont=dict(size=10, color="black"),
        hovertext=[f"{node} (Degree: {graph.degree(node)}, Presence: {deck_presence[str(node)]:.2%})" for node in
                   graph.nodes()],
        uid="node_trace",
    )

    cycle_traces = []
    cycle_colors = px.colors.qualitative.Set1
    for idx, cycle in enumerate(cycles[:5]):
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

    # --- Interactive Callback for Node Click ---
    fig.update_layout(
        title=dict(text=title),
        showlegend=True,
        hovermode="closest",
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=800,
        template="plotly_white",
        updatemenus=[],
    )

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
        "doubleClick": "reset+autosize",
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

# ----------------------------
# Metagame Scatter Plot
# ----------------------------
def plot_metagame_scatter(df: pd.DataFrame, allow_negative_power: bool = False) -> go.Figure:
    """Create an interactive scatter plot of the metagame based on the dataframe."""
    scatter_df = df.copy()
    
    if not allow_negative_power:
        scatter_df = scatter_df[scatter_df["Power Score"] >= 0.0]
    
    scatter_df["Bubble Size"] = scatter_df["Meta Score"].clip(lower=1.0)
    scatter_df["Label"] = scatter_df["Deck"] + " (" + scatter_df["Meta Score"].astype(str) + ")"

    fig = px.scatter(
        scatter_df,
        x="Power Score",
        y="Freq Score",
        size="Bubble Size",
        color="Deck",
        hover_name="Deck",
        hover_data={"Bubble Size": False, "Meta Score": True, "Power Score": True, "Freq Score": True, "Deck": False},
        text="Label"
    )
    
    fig.update_traces(textposition='bottom center')
    min_power = float(scatter_df["Power Score"].min())
    x_min = min(-5.0, min_power - 5.0) if allow_negative_power else -2.0
    fig.update_layout(
        xaxis=dict(range=[x_min, 105], title=dict(text="Power Score (Expected Win Rate)")),
        yaxis=dict(range=[-5, 105], title=dict(text="Frequency Score (Popularity)")),
        showlegend=False,
        height=600,
        margin=dict(t=30, b=30, l=30, r=30),
        template="plotly_white"
    )
    return fig

# ----------------------------
# Radar Chart
# ----------------------------
def plot_head_to_head_radar(deck_a: str, deck_b: str, categories: List[str], da_vals: List[float], db_vals: List[float], da_texts: List[str], db_texts: List[str]) -> go.Figure:
    """Create a dynamic radar chart comparing two decks using normalized coordinates but real hover text."""
    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=da_vals,
        theta=categories,
        fill='toself',
        name=deck_a,
        hoverinfo="text",
        text=da_texts,
        line=dict(color='rgba(46, 204, 113, 1)')  # Emerald Green
    ))
    fig.add_trace(go.Scatterpolar(
        r=db_vals,
        theta=categories,
        fill='toself',
        name=deck_b,
        hoverinfo="text",
        text=db_texts,
        line=dict(color='rgba(231, 76, 60, 1)')  # Alizarin Red
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                showline=False,
                range=[0, 100],
                gridcolor='rgba(255, 255, 255, 0.2)'
            )
        ),
        showlegend=True,
        height=450,
        margin=dict(t=40, b=40, l=40, r=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig