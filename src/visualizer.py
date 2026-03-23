"""
TemporalNet — Plotly Visualization Engine
==========================================
Dark-themed, publication-quality time series charts.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.stattools import acf, pacf
from typing import Dict, Optional

# Theme constants 
BG_COLOR = "#0a0a0a"
PAPER_COLOR = "#0a0a0a"
GRID_COLOR = "#1a1a2e"
TEXT_COLOR = "#e0e0e0"
FONT_FAMILY = "JetBrains Mono, Fira Code, Consolas, monospace"

ACCENT_COLORS = ["#00ff88", "#00d4ff", "#a855f7", "#f59e0b", "#ef4444", "#ec4899"]


def _base_layout(title: str = "", height: int = 500) -> dict:
    """Shared dark-theme layout settings."""
    return dict(
        template="plotly_dark",
        plot_bgcolor=BG_COLOR,
        paper_bgcolor=PAPER_COLOR,
        font=dict(family=FONT_FAMILY, color=TEXT_COLOR, size=12),
        title=dict(text=title, font=dict(size=18, color="#ffffff"), x=0.02),
        xaxis=dict(gridcolor=GRID_COLOR, zeroline=False),
        yaxis=dict(gridcolor=GRID_COLOR, zeroline=False),
        height=height,
        margin=dict(l=60, r=30, t=60, b=40),
        legend=dict(
            bgcolor="rgba(10,10,10,0.8)",
            bordercolor="#333",
            borderwidth=1,
            font=dict(size=11),
        ),
        hovermode="x unified",
    )


# 
# 1. Forecast Plot
# 
def forecast_plot(
    actual: pd.Series,
    predictions_dict: Dict[str, dict],
    title: str = "Time Series Forecast",
) -> go.Figure:
    """
    Line chart with actual data + multiple model predictions and confidence bands.

    Parameters
    ----------
    actual : pd.Series
        The observed time series.
    predictions_dict : dict
        Keys = model names, Values = dict with 'predictions', 'lower', 'upper'.
    """
    fig = go.Figure()

    # Actual series
    x_actual = list(range(len(actual)))
    fig.add_trace(
        go.Scatter(
            x=x_actual,
            y=actual.values,
            mode="lines",
            name="Actual",
            line=dict(color="#ffffff", width=2),
        )
    )

    # Forecast start
    forecast_start = len(actual)

    for idx, (model_name, data) in enumerate(predictions_dict.items()):
        color = ACCENT_COLORS[idx % len(ACCENT_COLORS)]
        preds = data["predictions"]
        lower = data.get("lower", preds * 0.95)
        upper = data.get("upper", preds * 1.05)

        x_pred = list(range(forecast_start, forecast_start + len(preds)))

        # Confidence band
        fig.add_trace(
            go.Scatter(
                x=x_pred + x_pred[::-1],
                y=list(upper) + list(lower[::-1]),
                fill="toself",
                fillcolor=color.replace(")", ", 0.1)").replace("rgb", "rgba")
                if "rgb" in color
                else f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.1)",
                line=dict(color="rgba(0,0,0,0)"),
                name=f"{model_name} CI",
                showlegend=False,
                hoverinfo="skip",
            )
        )

        # Prediction line
        fig.add_trace(
            go.Scatter(
                x=x_pred,
                y=list(preds),
                mode="lines",
                name=data.get("model_name", model_name),
                line=dict(color=color, width=2, dash="dot"),
            )
        )

    fig.update_layout(**_base_layout(title, height=500))
    return fig


# 
# 2. Model Comparison Bar Chart
# 
def model_comparison_bar(metrics_dict: Dict[str, Dict[str, float]]) -> go.Figure:
    """
    Grouped bar chart comparing MAE / RMSE / MAPE across models.

    Parameters
    ----------
    metrics_dict : dict
        Keys = model names, Values = dict with 'MAE', 'RMSE', 'MAPE'.
    """
    models = list(metrics_dict.keys())
    metric_names = ["MAE", "RMSE", "MAPE"]
    colors = {"MAE": "#00ff88", "RMSE": "#00d4ff", "MAPE": "#a855f7"}

    fig = go.Figure()

    for metric in metric_names:
        values = [metrics_dict[m].get(metric, 0) for m in models]
        fig.add_trace(
            go.Bar(
                x=models,
                y=values,
                name=metric,
                marker_color=colors[metric],
                text=[f"{v:.2f}" for v in values],
                textposition="auto",
                textfont=dict(size=11),
            )
        )

    layout = _base_layout("Model Performance Comparison", height=450)
    layout["barmode"] = "group"
    layout["xaxis"]["title"] = "Model"
    layout["yaxis"]["title"] = "Metric Value"
    fig.update_layout(**layout)
    return fig


# 
# 3. Decomposition Plot
# 
def decomposition_plot(
    trend: pd.Series,
    seasonal: pd.Series,
    residual: pd.Series,
    title: str = "Time Series Decomposition",
) -> go.Figure:
    """3-subplot figure: trend, seasonal pattern, and residuals."""
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        subplot_titles=["Trend", "Seasonal", "Residual"],
        vertical_spacing=0.08,
    )

    fig.add_trace(
        go.Scatter(
            y=trend.values, mode="lines", name="Trend",
            line=dict(color="#00ff88", width=2),
        ),
        row=1, col=1,
    )

    fig.add_trace(
        go.Scatter(
            y=seasonal.values, mode="lines", name="Seasonal",
            line=dict(color="#00d4ff", width=1.5),
        ),
        row=2, col=1,
    )

    fig.add_trace(
        go.Scatter(
            y=residual.values, mode="lines", name="Residual",
            line=dict(color="#a855f7", width=1),
        ),
        row=3, col=1,
    )

    layout = _base_layout(title, height=700)
    fig.update_layout(**layout)

    for i in range(1, 4):
        fig.update_xaxes(gridcolor=GRID_COLOR, row=i, col=1)
        fig.update_yaxes(gridcolor=GRID_COLOR, row=i, col=1)

    return fig


# 
# 4. ACF / PACF Plot
# 
def acf_plot(series: pd.Series, lags: int = 40) -> go.Figure:
    """ACF and PACF side-by-side bar charts."""
    clean = series.dropna().values.astype(float)
    n = len(clean)
    lags = min(lags, n // 2 - 1)
    if lags < 1:
        lags = 1

    acf_vals = acf(clean, nlags=lags, fft=True)
    pacf_vals = pacf(clean, nlags=lags)

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["Autocorrelation (ACF)", "Partial Autocorrelation (PACF)"],
    )

    # Significance band
    conf = 1.96 / np.sqrt(n)

    # ACF
    fig.add_trace(
        go.Bar(
            x=list(range(lags + 1)),
            y=acf_vals,
            marker_color="#00ff88",
            name="ACF",
        ),
        row=1, col=1,
    )
    fig.add_hline(y=conf, line_dash="dash", line_color="#f59e0b", row=1, col=1)
    fig.add_hline(y=-conf, line_dash="dash", line_color="#f59e0b", row=1, col=1)

    # PACF
    fig.add_trace(
        go.Bar(
            x=list(range(lags + 1)),
            y=pacf_vals,
            marker_color="#00d4ff",
            name="PACF",
        ),
        row=1, col=2,
    )
    fig.add_hline(y=conf, line_dash="dash", line_color="#f59e0b", row=1, col=2)
    fig.add_hline(y=-conf, line_dash="dash", line_color="#f59e0b", row=1, col=2)

    layout = _base_layout("ACF & PACF Analysis", height=400)
    fig.update_layout(**layout)

    for i in range(1, 3):
        fig.update_xaxes(gridcolor=GRID_COLOR, title="Lag", row=1, col=i)
        fig.update_yaxes(gridcolor=GRID_COLOR, row=1, col=i)

    return fig
