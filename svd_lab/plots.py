"""Framework-independent Plotly figure construction for SVD results."""

import numpy as np
import plotly.graph_objects as go

from svd_lab.svd import SVDResult, metrics_for


def singular_value_figure(result: SVDResult, rank: int) -> go.Figure:
    """Plot relative singular-value magnitude and highlight the retained rank."""
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
            line={"color": "#8B949E", "width": 2},
            name="All components",
            hovertemplate="Component %{x}<br>Relative magnitude %{y:.3f}<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=components[:rank],
            y=relative_values[:rank],
            mode="lines+markers",
            line={"color": "#58A6FF", "width": 3},
            marker={"color": "#58A6FF", "size": 6},
            name=f"Retained at rank {rank}",
            hovertemplate="Retained component %{x}<br>Relative magnitude %{y:.3f}<extra></extra>",
        )
    )
    figure.update_layout(
        title={"text": "Singular-value spectrum"},
        xaxis={"title": {"text": "Component"}, "rangemode": "tozero"},
        yaxis={"title": {"text": "Relative magnitude"}, "rangemode": "tozero"},
        height=340,
        margin={"l": 20, "r": 20, "t": 55, "b": 20},
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.02, "x": 1, "xanchor": "right"},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return figure


__all__ = ["singular_value_figure"]
