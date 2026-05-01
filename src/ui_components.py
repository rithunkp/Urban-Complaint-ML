from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.domain import SECTOR_DISPLAY_NAMES

PALETTE = ["#0b4f6c", "#ff7f50", "#f2c14e", "#2a9d8f", "#3d405b", "#bc4749", "#7f5539"]

PLOT_LAYOUT = dict(
    paper_bgcolor="#f8f4ec",
    plot_bgcolor="#f8f4ec",
    font=dict(family="Arial", color="#173042"),
    margin=dict(l=20, r=20, t=50, b=20),
)

APP_CSS = """
#hero-card {
    background: linear-gradient(135deg, #fffdf8 0%, #f7efe1 100%);
    border: 1px solid #e0d4c3;
    border-radius: 28px;
    padding: 24px 28px;
    margin-bottom: 16px;
    box-shadow: 0 14px 30px rgba(18, 43, 58, 0.08);
}
#hero-card h1 {
    margin: 0;
    font-size: 2.2rem;
    color: #0b4f6c;
}
#hero-card p {
    margin-top: 10px;
    color: #4c5d69;
    font-size: 1rem;
}
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 12px;
}
.kpi-card {
    background: #fffdf8;
    border: 1px solid #e0d4c3;
    border-radius: 20px;
    padding: 14px 16px;
}
.kpi-label {
    font-size: 0.78rem;
    text-transform: uppercase;
    color: #667785;
    letter-spacing: 0.08em;
}
.kpi-value {
    font-size: 1.75rem;
    font-weight: 700;
    color: #0b4f6c;
    margin-top: 6px;
}
.section-note {
    background: #fffdf8;
    border: 1px solid #e0d4c3;
    border-radius: 18px;
    padding: 14px 16px;
    color: #41515d;
}
.section-note,
.section-note * {
    color: #173042 !important;
    opacity: 1 !important;
}
.section-note p,
.section-note li,
.section-note span,
.section-note strong,
.section-note code {
    color: #173042 !important;
}
.section-note ul,
.section-note ol {
    margin: 0.25rem 0 0.25rem 1.2rem !important;
}
.section-note li::marker {
    color: #7a5a2f !important;
}
"""


def hero_html() -> str:
    """Return the top hero section HTML."""
    return """
    <div id="hero-card">
        <div style="text-transform: uppercase; letter-spacing: 0.14em; color: #c65d3a; font-size: 0.8rem; font-weight: 700;">
            Kerala Urban Complaint Analysis
        </div>
        <h1>Kerala Urban Complaint Analysis for Smart City Resource Optimization.</h1>
        <p>
            A decision-support product for municipal teams: monitor complaint pressure, detect operational spikes,
            understand district-level service demand, and allocate resources across roads, water, waste, lighting,
            drainage, and traffic without neglecting any sector.
        </p>
    </div>
    """


def kpi_html(metrics: dict[str, str]) -> str:
    """Render KPI cards as a lightweight HTML grid."""
    cards = [
        ("Complaints in view", metrics["complaints"]),
        ("Complaint types", metrics["complaint_types"]),
        ("Region coverage", metrics["regions"]),
        ("Median closure", metrics["median_closure"]),
    ]
    fragments = [
        f'<div class="kpi-card"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div></div>'
        for label, value in cards
    ]
    return f'<div class="kpi-grid">{"".join(fragments)}</div>'


def build_distribution_figure(data: pd.DataFrame) -> go.Figure:
    """Create the complaint distribution donut chart."""
    if data.empty:
        return empty_figure("Complaint Distribution", "No complaint records are available for the current filter set.")
    fig = px.pie(
        data,
        names="complaint_type",
        values="count",
        hole=0.56,
        color_discrete_sequence=PALETTE,
        title="Complaint Distribution",
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(height=420, **PLOT_LAYOUT)
    return fig


def build_top_categories_figure(data: pd.DataFrame) -> go.Figure:
    """Create the top complaint type bar chart."""
    if data.empty:
        return empty_figure("Top Complaint Types", "No complaint types are available for the current filter set.")
    ordered = data.sort_values("count", ascending=True)
    fig = px.bar(
        ordered,
        x="count",
        y="complaint_type",
        orientation="h",
        color="share",
        color_continuous_scale=["#f6d6b8", "#ff7f50", "#bc4749"],
        title="Top Complaint Types",
    )
    fig.update_layout(height=420, coloraxis_showscale=False, xaxis_title="Requests", yaxis_title="", **PLOT_LAYOUT)
    return fig


def build_timeline_figure(timeline: pd.DataFrame, threshold: float) -> go.Figure:
    """Create a timeline chart with anomaly markers."""
    if timeline.empty:
        return empty_figure("Complaint Volume Over Time", "No timeline data is available for the current filter set.")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=timeline["period"],
            y=timeline["count"],
            mode="lines+markers",
            name="Observed volume",
            line=dict(color="#0b4f6c", width=2.8),
            marker=dict(size=5, color="#0b4f6c"),
            fill="tozeroy",
            fillcolor="rgba(11, 79, 108, 0.08)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=timeline["period"],
            y=timeline["rolling_mean"],
            mode="lines",
            name="Rolling mean",
            line=dict(color="#f2c14e", width=2, dash="dot"),
        )
    )
    flagged = timeline[timeline["is_anomaly"]]
    if not flagged.empty:
        fig.add_trace(
            go.Scatter(
                x=flagged["period"],
                y=flagged["count"],
                mode="markers",
                name=f"Anomaly > {threshold:.2f}σ",
                marker=dict(color="#bc4749", size=11, symbol="diamond"),
            )
        )
    fig.update_layout(height=380, title="Complaint Volume Over Time", yaxis_title="Requests", xaxis_title="", **PLOT_LAYOUT)
    return fig


def build_sector_figure(data: pd.DataFrame, column: str, title: str) -> go.Figure:
    """Create a sector comparison bar chart."""
    if data.empty:
        return empty_figure(title, "No sector data is available.")
    ordered = data.sort_values(column, ascending=True)
    fig = px.bar(
        ordered,
        x=column,
        y="sector_display",
        orientation="h",
        color=column,
        color_continuous_scale=["#f6d6b8", "#f2c14e", "#0b4f6c"],
        title=title,
    )
    fig.update_layout(height=360, coloraxis_showscale=False, xaxis_title="Percent", yaxis_title="", **PLOT_LAYOUT)
    return fig


def build_resource_pie(allocation: pd.DataFrame) -> go.Figure:
    """Create the municipal resource splitup chart."""
    if allocation.empty:
        return empty_figure("Recommended Resource Splitup", "No allocation data is available.")
    fig = px.pie(
        allocation,
        names="sector_display",
        values="recommended_pct",
        hole=0.5,
        color_discrete_sequence=PALETTE,
        title="Recommended Resource Splitup",
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(height=400, **PLOT_LAYOUT)
    return fig


def build_region_sector_heatmap(matrix: pd.DataFrame, domain: str) -> go.Figure:
    """Create the region-by-sector heatmap."""
    if matrix.empty:
        return empty_figure("Region vs Sector Pressure", "No region and sector intersections are available.")
    pivot = matrix.pivot(index="region", columns="sector_display", values="count").fillna(0)
    fig = px.imshow(
        pivot,
        aspect="auto",
        color_continuous_scale=["#fff4e6", "#f2c14e", "#0b4f6c"],
        title=f"{'District' if domain == 'Kerala' else 'Borough'} vs Sector Pressure",
    )
    fig.update_layout(height=420, xaxis_title="", yaxis_title="", **PLOT_LAYOUT)
    return fig


def build_geo_figure(frame: pd.DataFrame, domain: str) -> go.Figure:
    """Create a simple geospatial scatter plot when coordinate coverage is available."""
    geo = frame.dropna(subset=["latitude", "longitude"])
    if geo.empty:
        return empty_figure("Geospatial Lens", "Coordinate coverage is not sufficient for the current filter set.")

    fig = px.scatter_mapbox(
        geo.sample(min(2500, len(geo)), random_state=42),
        lat="latitude",
        lon="longitude",
        color="sector",
        hover_name="complaint_type",
        hover_data={"region": True, "sector": True},
        zoom=10 if domain == "Kerala" else 9,
        center=dict(
            lat=float(geo["latitude"].median()),
            lon=float(geo["longitude"].median()),
        ),
        mapbox_style="carto-positron",
        title="Geospatial Lens",
        color_discrete_sequence=PALETTE,
    )
    fig.update_layout(height=460, margin=dict(l=0, r=0, t=50, b=0))
    return fig


def build_prediction_figure(predictions: pd.DataFrame) -> go.Figure:
    """Create the top-N prediction confidence chart for the ML demo."""
    if predictions.empty:
        return empty_figure("Top Prediction Confidence", "Prediction probabilities will appear here.")
    fig = px.bar(
        predictions,
        x="confidence",
        y="localized_label",
        orientation="h",
        color="confidence",
        color_continuous_scale=["#dceaf0", "#4c8da3", "#0b4f6c"],
        title="Top Prediction Confidence",
    )
    fig.update_layout(height=320, xaxis_tickformat=".0%", coloraxis_showscale=False, yaxis_title="", **PLOT_LAYOUT)
    return fig


def empty_figure(title: str, message: str) -> go.Figure:
    """Return a placeholder figure with a centered message."""
    fig = go.Figure()
    fig.add_annotation(text=message, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False, font=dict(size=15))
    fig.update_layout(title=title, xaxis_visible=False, yaxis_visible=False, height=340, **PLOT_LAYOUT)
    return fig
