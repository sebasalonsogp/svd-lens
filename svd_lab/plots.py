"""Framework-independent Plotly figure construction for SVD results."""

import numpy as np
import plotly.graph_objects as go

from svd_lab.svd import SVDResult, metrics_for


def singular_value_figure(result: SVDResult, rank: int) -> go.Figure:
    """Plot pattern strength and make the selected rank visually explicit."""
    metrics_for(result, rank)
    components = np.arange(1, result.max_rank + 1)
    largest = float(result.singular_values[0])
    relative_values = (
        result.singular_values / largest if largest > 0.0 else np.zeros(result.max_rank)
    )

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=components,
            y=relative_values,
            mode="lines",
            line={"color": "#8DA39B", "width": 2},
            name="All available patterns",
            hovertemplate=("Pattern %{x}<br>Strength vs. strongest %{y:.3f}<extra></extra>"),
        )
    )
    figure.add_trace(
        go.Scatter(
            x=components[:rank],
            y=relative_values[:rank],
            mode="lines+markers",
            line={"color": "#2EC4A6", "width": 3},
            marker={"color": "#2EC4A6", "size": 6},
            name=f"Kept patterns — rank {rank}",
            hovertemplate=("Kept pattern %{x}<br>Strength vs. strongest %{y:.3f}<extra></extra>"),
        )
    )
    cutoff = rank + 0.5
    figure.add_vrect(
        x0=0.5,
        x1=cutoff,
        fillcolor="rgba(46, 196, 166, 0.08)",
        line_width=0,
        layer="below",
    )
    figure.add_vline(
        x=cutoff,
        line={"color": "#E6C77B", "dash": "dot", "width": 2},
        annotation_text=f"Rank {rank} cutoff",
        annotation_position="top right",
        annotation_font={"color": "#E6C77B", "size": 12},
    )
    figure.update_layout(
        title={"text": "Pattern strength, strongest to weakest"},
        xaxis={
            "title": {"text": "Pattern number — strongest to weakest"},
            "range": [0.5, result.max_rank + 0.5],
        },
        yaxis={
            "title": {"text": "Strength vs. strongest pattern"},
            "range": [0.0, 1.05],
        },
        height=340,
        margin={"l": 20, "r": 20, "t": 55, "b": 20},
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.03, "x": 1, "xanchor": "right"},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return figure


__all__ = ["singular_value_figure"]
