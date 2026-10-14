"""Shared Plotly styling for the dashboard."""

from __future__ import annotations

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Categorical slots in fixed order (validated light/dark palette).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
MUTED = "#898781"

SECTION_COLORS = {"12.25 in": SERIES[2], "12.00 in": SERIES[0], "8.75 in": SERIES[1]}


def style(fig: go.Figure, height: int = 380, legend: bool | None = None) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=60, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        font=dict(family="Google Sans, sans-serif"),
    )
    if legend is not None:
        fig.update_layout(showlegend=legend)
    fig.update_xaxes(showline=True, ticks="outside", zeroline=False)
    fig.update_yaxes(showline=True, ticks="outside", zeroline=False)
    return fig


def depth_tracks(df, tracks: list[tuple[str, str]], color_by: str | None = None,
                 height: int = 650, log_tracks: tuple[str, ...] = ("mse",)) -> go.Figure:
    """Log-style depth tracks: one panel per channel, shared inverted depth axis."""
    fig = make_subplots(rows=1, cols=len(tracks), shared_yaxes=True,
                        subplot_titles=[t for _, t in tracks], horizontal_spacing=0.02)
    groups = [(None, df)] if color_by is None else list(df.groupby(color_by, sort=False))
    for gi, (name, g) in enumerate(groups):
        color = SECTION_COLORS.get(name, SERIES[gi % len(SERIES)]) if name else SERIES[0]
        for i, (col, title) in enumerate(tracks, start=1):
            if col not in g:
                continue
            fig.add_trace(go.Scatter(
                x=g[col], y=g["depth"], mode="lines", line=dict(width=1.2, color=color),
                name=str(name) if name else title, legendgroup=str(name),
                showlegend=(i == 1 and name is not None),
                hovertemplate=f"{title}: %{{x:.1f}}<br>Depth: %{{y:,.0f}} ft<extra>{name or ''}</extra>",
            ), row=1, col=i)
    for i, (col, _) in enumerate(tracks, start=1):
        if col in log_tracks:
            fig.update_xaxes(type="log", row=1, col=i)
    fig.update_yaxes(autorange="reversed", title_text="Measured depth (ft)", row=1, col=1)
    return style(fig, height=height, legend=color_by is not None)
