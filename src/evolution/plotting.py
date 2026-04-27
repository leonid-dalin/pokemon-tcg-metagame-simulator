# plotting.py | Interactive metagame visualisations using Plotly
from __future__ import annotations
from typing import List, Optional, Union, Dict, Any
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx

import structlog
from opentelemetry import trace

logger = structlog.get_logger()
tracer = trace.get_tracer(__name__)

# ----------------------------
# Metagame Evolution Plot (Interactive)
# ----------------------------
def plot_metagame_evolution_interactive(
    history: List[np.ndarray],
    deck_names: List[str],
    extinction_gens: Optional[Union[Dict[Any, int], List[Optional[int]]]] = None,
    top_n: int = 12,
    save_path: Optional[str] = None,
    title: str = "Metagame Share Over Time",
) -> Optional[go.Figure]:
    """Plot metagame evolution with Plotly — fully interactive, zoomable, hoverable.
    Args:
        history: List of frequency arrays over generations.
        deck_names: List of deck names.
        extinction_gens: Mapping or list of extinction generation per deck.
        top_n: Number of top decks to highlight.
        save_path: Path to save HTML file (if None, returns figure).
        title: Plot title.
    Returns:
        Plotly Figure object if save_path is None.
    """
    history_arr = np.array(history)
    n_gens = int(history_arr.shape[0])
    final_freqs = history_arr[-1, :]
    top_indices = np.argsort(-final_freqs)[:top_n]

    top_deck_names = [deck_names[i] for i in top_indices]
    history_top = history_arr[:, top_indices]

    df = pd.DataFrame(history_top, columns=top_deck_names)
    df.index.name = "Generation"

    fig = px.line(
        df,
        title=title,
        labels={"value": "Metagame Share", "variable": "Deck"},
        color_discrete_sequence=px.colors.qualitative.Bold
    )

    for idx_raw in top_indices:
        idx = int(idx_raw)
        deck_name = str(deck_names[idx])
        extinction_gen: Optional[int] = None

        if extinction_gens is not None:
            if isinstance(extinction_gens, dict):
                val = extinction_gens.get(deck_name) or extinction_gens.get(idx) or extinction_gens.get(
                    str(idx))
                if isinstance(val, (int, float)): extinction_gen = int(val)
            elif isinstance(extinction_gens, list) and idx < len(extinction_gens):
                val_list = extinction_gens[idx]
                if isinstance(val_list, (int, float)): extinction_gen = int(val_list)

        if extinction_gen is not None and extinction_gen < n_gens:
            fig.add_vline(
                x=extinction_gen,
                line_dash="dot",
                line_color="red",
                annotation_text=f"{deck_name} Extinct",
                annotation_position="top left",
                opacity=0.5
            )

    fig.update_layout(
        hovermode="x unified",
        yaxis_tickformat=".1%",
        template="plotly_white",
        legend_orientation="h",
        legend_yanchor="bottom",
        legend_y=1.02,
        legend_xanchor="right",
        legend_x=1
    )

    if save_path:
        fig.write_html(save_path, include_plotlyjs="cdn")
        logger.info("plot_saved", type="evolution_plot", path=save_path)

    return fig


# ----------------------------
# Matchup Heatmap (Interactive)
# ----------------------------
def plot_matchup_heatmap_interactive(
    win_matrix: np.ndarray,
    deck_names: List[str],
    tier_ordered_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    title: str = "Matchup Matrix (WR%)",
) -> Optional[go.Figure]:
    """Interactive heatmap of deck vs deck win rates, sorted by tier.
    Args:
        win_matrix: n x n win-rate matrix.
        deck_names: List of deck names.
        tier_ordered_names: Optional list of deck names in the desired order (S-Tier to D-Tier).
        save_path: Path to save HTML (if None, returns figure).
        title: Plot title.
    Returns:
        Plotly Figure if save_path is None.
    """
    with tracer.start_as_current_span("plot_matchup_heatmap_interactive"):
        n = len(deck_names)
        if n == 0:
            return None

        display_names: List[str] = deck_names
        display_matrix: np.ndarray = win_matrix

        if tier_ordered_names:
            valid_names = [name for name in tier_ordered_names if name in deck_names]
            missing_names = [name for name in deck_names if name not in valid_names]
            display_names = valid_names + missing_names

            order_indices = [deck_names.index(name) for name in display_names]
            display_matrix = win_matrix[np.ix_(order_indices, order_indices)]

        display_matrix_pct = display_matrix * 100.0

        fig = px.imshow(
            display_matrix_pct,
            x=display_names,
            y=display_names,
            color_continuous_scale="RdBu",
            zmin=20.0,
            zmax=80.0,
            title=title,
            labels=dict(x="Opponent", y="Deck", color="Win Rate")
        )

        fig.update_traces(
            hovertemplate="%{y} vs %{x}<br>Win Rate: %{z:.1f}%<extra></extra>"
        )

        fig.update_layout(
            xaxis_tickangle=-45,
            width=max(800, 30 * n),
            height=max(800, 30 * n),
            template="plotly_white",
            margin_l=150,
            margin_b=150,
            coloraxis_colorbar_ticksuffix="%"
        )

        if save_path:
            fig.write_html(save_path, include_plotlyjs="cdn")
            logger.info("plot_saved", type="matchup_heatmap", path=save_path)

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
    """Create an interactive network graph visualising significant win-rate edges and highlighting detected RPS cycles.
    Features:
        - Click on a node to focus on it and its direct neighbours.
        - Node size is scaled by the deck's all-time presence (if history is provided).
    """
    with tracer.start_as_current_span("plot_matchup_network"):
        graph = nx.DiGraph()
        n = len(deck_names)

        for i in range(n):
            for j in range(n):
                if i != j and win_matrix[i, j] > 0.55:
                    graph.add_edge(str(deck_names[i]), str(deck_names[j]), weight=float(win_matrix[i, j]))

        if not graph.nodes:
            logger.warning("network_plot_aborted", detail="No significant nodes.")
            return None

        pos = nx.kamada_kawai_layout(graph)

        if metagame_history is not None and len(metagame_history) > 0:
            total_metagame = np.mean(metagame_history, axis=0)
            deck_presence = {str(deck_names[i]): float(total_metagame[i]) for i in range(n)}
        else:
            deck_presence = {str(name): 1.0 for name in deck_names}

        edge_x, edge_y, edge_hover = [], [], []
        for u, v, d in graph.edges(data=True):
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_hover.append(f"{u} → {v}: {d['weight']:.2%}")

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=2, color="rgba(100, 100, 100, 0.8)"),
            hoverinfo="text",
            text=edge_hover,
            mode="lines",
            showlegend=False
        )

        node_x = [pos[node][0] for node in graph.nodes()]
        node_y = [pos[node][1] for node in graph.nodes()]
        max_pres = max(deck_presence.values()) or 1.0
        node_size = [10 + 30 * (deck_presence[str(node)] / max_pres) for node in graph.nodes()]

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode="markers+text",
            hoverinfo="text",
            marker=dict(
                size=node_size,
                color="lightblue",
                line=dict(width=2, color="darkblue")
            ),
            text=[str(node) for node in graph.nodes()],
            textposition="top center",
            textfont=dict(size=10),
            hovertext=[f"{n} (Presence: {deck_presence[str(n)]:.2%})" for n in graph.nodes()]
        )

        fig = go.Figure(data=[edge_trace, node_trace])

        colors = px.colors.qualitative.Set1
        for idx, cycle in enumerate(cycles[:5]):
            cx, cy = [], []
            for i in range(len(cycle)):
                u, v = str(cycle[i]), str(cycle[(i + 1) % len(cycle)])
                if u in pos and v in pos:
                    cx.extend([pos[u][0], pos[v][0], None])
                    cy.extend([pos[u][1], pos[v][1], None])
            fig.add_trace(go.Scatter(
                x=cx, y=cy, mode="lines",
                line=dict(width=4, color=colors[idx % len(colors)], dash="dash"),
                name=f"Cycle {idx + 1}"
            ))

        fig.update_layout(
            title=dict(text=title),
            showlegend=True,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=800,
            template="plotly_white"
        )

        if save_path:
            fig.write_html(save_path, config={"scrollZoom": True, "responsive": True})
        return fig

# ----------------------------
# Metagame Scatter Plot
# ----------------------------
def plot_metagame_scatter(df: pd.DataFrame, allow_negative_power: bool = False) -> go.Figure:
    """Create an interactive scatter plot of the metagame based on the dataframe."""
    with tracer.start_as_current_span("plot_metagame_scatter"):
        scatter_df = df.copy()
        if not allow_negative_power:
            scatter_df = scatter_df[scatter_df["Power Score"] >= 0.0]

        scatter_df["Bubble Size"] = scatter_df["Meta Score"].clip(lower=1.0)

        fig = px.scatter(
            scatter_df,
            x="Power Score",
            y="Freq Score",
            size="Bubble Size",
            color="Deck",
            hover_name="Deck",
            hover_data={"Meta Score": True, "Power Score": True, "Freq Score": True, "Bubble Size": False},
            text=scatter_df["Deck"] + " (" + scatter_df["Meta Score"].astype(str) + ")"
        )

        fig.update_traces(textposition='bottom center')

        x_min = -2.0 if not allow_negative_power else min(-5.0, float(scatter_df["Power Score"].min()) - 5.0)

        # Enforce dict assignment for layout constraints
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
def plot_head_to_head_radar(
        deck_a: str, deck_b: str,
        categories: List[str],
        da_vals: List[float], db_vals: List[float],
        da_texts: List[str], db_texts: List[str]
) -> go.Figure:
    """Renders an interactive radar chart with accessible, non-overlapping hover data."""
    with tracer.start_as_current_span("plot_head_to_head_radar"):
        fig = go.Figure()

        marker_cfg = dict(size=12, line=dict(width=2, color='white'))

        fig.add_trace(go.Scatterpolar(
            name=deck_a,
            r=da_vals,
            theta=categories,
            fill='toself',
            fillcolor='rgba(46, 204, 113, 0.3)',
            mode='lines+markers',
            marker=marker_cfg,
            line=dict(color='rgba(46, 204, 113, 1)', width=3),
            hoveron='points',
            customdata=da_texts,
            hovertemplate="<b>%{fullData.name}</b><br>%{customdata}<extra></extra>"
        ))

        fig.add_trace(go.Scatterpolar(
            name=deck_b,
            r=db_vals,
            theta=categories,
            fill='toself',
            fillcolor='rgba(231, 76, 60, 0.3)',
            mode='lines+markers',
            marker=marker_cfg,
            line=dict(color='rgba(231, 76, 60, 1)', width=3),
            hoveron='points',
            customdata=db_texts,
            hovertemplate="<b>%{fullData.name}</b><br>%{customdata}<extra></extra>"
        ))

        fig.update_layout(
            hovermode="closest",
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100],
                    tickvals=[20, 40, 60, 80],
                    gridcolor='rgba(255, 255, 255, 0.2)',
                    tickfont=dict(size=14, color='#131313', family='Arial Black'),
                    angle=45,
                    tickangle=45,
                    layer='above traces'
                ),
                angularaxis=dict(
                    tickfont=dict(size=20, color='#F5F5F5', family='Arial Black'),
                    rotation=90,
                    direction="clockwise"
                )
            ),
            showlegend=True,
            legend=dict(
                font=dict(size=16, color='#F5F5F5'),
                bgcolor='rgba(0, 0, 0, 0)',
                x=1.1,
                y=0.5,
                xanchor='left',
                yanchor='middle'
            ),
            height=650,
            margin=dict(t=80, b=80, l=80, r=80),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        return fig