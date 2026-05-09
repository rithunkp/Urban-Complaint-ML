"""
src/plots.py

Diagnostic plot functions for the Urban Complaint ML demo.
All functions return plotly Figure objects — drop straight into gr.Plot().

Available functions
-------------------
allocation_benchmark_chart   — grouped bar: 3 strategies × 7 sectors
transfer_confusion_matrix    — heatmap: expected vs predicted NYC label
confidence_distribution      — overlapping histogram: correct vs incorrect
per_district_accuracy        — horizontal bar: transfer accuracy per district
sector_load_vs_recommended   — scatter: load share vs recommended allocation
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ── shared style ──────────────────────────────────────────────────────────────

_TEMPLATE = "plotly_white"
_BLUE = "#3b82f6"
_GREEN = "#22c55e"
_RED = "#ef4444"
_GREY = "#94a3b8"
_SLATE = "#64748b"

STRATEGY_COLORS = {
    "Equal Split": _GREY,
    "Count-Only": _SLATE,
    "Model (4-Factor)": _BLUE,
}


# ── 1. Benchmark grouped bar ──────────────────────────────────────────────────

def allocation_benchmark_chart(benchmark_df: pd.DataFrame) -> go.Figure:
    """
    Grouped bar chart comparing three allocation strategies across all sectors.

    Parameters
    ----------
    benchmark_df
        Output of src.benchmark.compute_benchmark().
        Columns: sector | Equal Split | Count-Only | Model (4-Factor)
    """
    strategies = ["Equal Split", "Count-Only", "Model (4-Factor)"]
    fig = go.Figure()

    for col in strategies:
        fig.add_trace(go.Bar(
            name=col,
            x=benchmark_df["sector"],
            y=benchmark_df[col],
            marker_color=STRATEGY_COLORS[col],
            text=benchmark_df[col].map(lambda v: f"{v:.1f}%"),
            textposition="outside",
        ))

    fig.update_layout(
        barmode="group",
        title="Resource Allocation: Baseline Strategies vs Model",
        xaxis_title="Sector",
        yaxis_title="Allocation (%)",
        yaxis=dict(range=[0, benchmark_df[strategies].max().max() * 1.18]),
        legend_title="Strategy",
        template=_TEMPLATE,
        height=430,
        margin=dict(t=60, b=80),
    )
    return fig


# ── 2. Transfer confusion matrix ─────────────────────────────────────────────

def transfer_confusion_matrix(kerala_df: pd.DataFrame) -> go.Figure:
    """
    Row-normalised heatmap of expected_nyc_label vs predicted_nyc_label.
    Shows which complaint families the model confuses across domains.
    """
    sub = kerala_df.dropna(subset=["expected_nyc_label", "predicted_nyc_label"])
    labels = sorted(sub["expected_nyc_label"].unique())

    cm = (
        pd.crosstab(
            sub["expected_nyc_label"],
            sub["predicted_nyc_label"],
            normalize="index",
        )
        .reindex(index=labels, columns=labels, fill_value=0.0)
    )

    fig = px.imshow(
        cm,
        text_auto=".2f",
        color_continuous_scale="Blues",
        title="Transfer Confusion Matrix<br><sup>Kerala → NYC labels · row-normalised</sup>",
        labels={"x": "Predicted label", "y": "Expected label", "color": "Fraction"},
        aspect="auto",
        height=560,
    )
    fig.update_xaxes(tickangle=35)
    fig.update_layout(template=_TEMPLATE, margin=dict(t=80))
    return fig


# ── 3. Confidence distribution ───────────────────────────────────────────────

def confidence_distribution(kerala_df: pd.DataFrame) -> go.Figure:
    """
    Overlapping histogram of prediction confidence split by correct / incorrect.
    A well-behaved model shows high confidence on correct rows and
    low confidence on incorrect rows.
    """
    sub = kerala_df.dropna(subset=["prediction_confidence", "correct_reference"])
    correct = sub.loc[sub["correct_reference"] == True, "prediction_confidence"]
    incorrect = sub.loc[sub["correct_reference"] == False, "prediction_confidence"]

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=correct, name="Correct",
        nbinsx=30, opacity=0.70,
        marker_color=_GREEN,
    ))
    fig.add_trace(go.Histogram(
        x=incorrect, name="Incorrect",
        nbinsx=30, opacity=0.70,
        marker_color=_RED,
    ))

    # mark the 0.8 medium/high threshold
    fig.add_vline(x=0.8, line_dash="dash", line_color=_SLATE,
                  annotation_text="0.80 threshold", annotation_position="top right")

    fig.update_layout(
        barmode="overlay",
        title="Prediction Confidence Distribution<br><sup>Correct vs Incorrect on Kerala transfer set</sup>",
        xaxis_title="Confidence",
        yaxis_title="Count",
        legend_title="Prediction",
        template=_TEMPLATE,
        height=400,
        margin=dict(t=80),
    )
    return fig


# ── 4. Per-district accuracy ─────────────────────────────────────────────────

def per_district_accuracy(kerala_df: pd.DataFrame) -> go.Figure:
    """
    Horizontal bar chart: transfer accuracy per Kerala district.
    Sorted ascending so worst-performing districts are at the top
    (natural reading direction for action priority).
    """
    sub = kerala_df.dropna(subset=["correct_reference"])
    grp = (
        sub.groupby("city_or_district")["correct_reference"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "accuracy", "count": "n", "city_or_district": "district"})
        .sort_values("accuracy", ascending=True)
    )

    fig = px.bar(
        grp,
        x="accuracy",
        y="district",
        orientation="h",
        title="Transfer Accuracy by District<br><sup>Fraction of Kerala complaints correctly mapped to NYC labels</sup>",
        labels={"accuracy": "Accuracy", "district": "District", "n": "Complaints"},
        color="accuracy",
        color_continuous_scale="RdYlGn",
        range_color=[0.0, 1.0],
        hover_data={"n": True, "accuracy": ":.3f"},
        height=520,
    )
    fig.update_coloraxes(showscale=False)
    fig.add_vline(
        x=grp["accuracy"].mean(),
        line_dash="dash",
        line_color=_SLATE,
        annotation_text=f"mean {grp['accuracy'].mean():.2f}",
        annotation_position="top right",
    )
    fig.update_layout(template=_TEMPLATE, margin=dict(t=80, l=130))
    return fig


# ── 5. Sector load vs recommended scatter ────────────────────────────────────

def sector_load_vs_recommended(
    sector_load: dict[str, float],
    model_scores: dict[str, float],
) -> go.Figure:
    """
    Scatter: current complaint-load share (x) vs model recommended allocation (y).

    Points above the diagonal are sectors the model upweights beyond raw load —
    driven by trend, anomaly, or closure-delay pressure.
    Points below are sectors the model dampens despite high complaint volume.

    Parameters
    ----------
    sector_load
        Dict {sector: fraction} — raw complaint share from the dataset.
    model_scores
        Dict {sector: fraction} — recommended allocation from resource_allocation.py.
    """
    sectors = list(sector_load.keys())
    x_vals = [sector_load[s] * 100 for s in sectors]
    y_vals = [model_scores.get(s, 0) * 100 for s in sectors]

    lim = max(max(x_vals), max(y_vals)) * 1.15

    fig = go.Figure()

    # diagonal reference line
    fig.add_trace(go.Scatter(
        x=[0, lim], y=[0, lim],
        mode="lines",
        line=dict(dash="dash", color=_GREY, width=1.5),
        name="Equal weighting",
        hoverinfo="skip",
    ))

    # annotation bands (optional context)
    fig.add_annotation(
        x=lim * 0.82, y=lim * 0.72,
        text="↓ dampened by model",
        showarrow=False,
        font=dict(size=11, color=_SLATE),
        xanchor="left",
    )
    fig.add_annotation(
        x=lim * 0.12, y=lim * 0.30,
        text="↑ upweighted by model",
        showarrow=False,
        font=dict(size=11, color=_SLATE),
        xanchor="left",
    )

    # sector points
    fig.add_trace(go.Scatter(
        x=x_vals, y=y_vals,
        mode="markers+text",
        text=sectors,
        textposition="top center",
        marker=dict(size=13, color=_BLUE, line=dict(color="white", width=1.5)),
        name="Sectors",
        customdata=np.stack([
            [f"{v:.1f}%" for v in x_vals],
            [f"{v:.1f}%" for v in y_vals],
        ], axis=-1),
        hovertemplate="<b>%{text}</b><br>Load: %{customdata[0]}<br>Recommended: %{customdata[1]}<extra></extra>",
    ))

    fig.update_layout(
        title="Sector Load vs Recommended Allocation<br><sup>Points above diagonal are upweighted by trend/anomaly/delay signals</sup>",
        xaxis_title="Complaint Load Share (%)",
        yaxis_title="Recommended Allocation (%)",
        xaxis=dict(range=[0, lim]),
        yaxis=dict(range=[0, lim]),
        template=_TEMPLATE,
        height=460,
        margin=dict(t=90),
        showlegend=True,
    )
    return fig
