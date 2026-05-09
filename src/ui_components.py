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
:root {
    --panel-bg: #fffdf8;
    --panel-border: #e0d4c3;
    --ink-strong: #173042;
    --ink-accent: #0b4f6c;
    --ink-muted: #4c5d69;
    --ink-soft: #667785;
}

#hero-card {
    background: linear-gradient(135deg, #fffdf8 0%, #f7efe1 100%);
    border: 1px solid var(--panel-border);
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
    background: var(--panel-bg);
    border: 1px solid var(--panel-border);
    border-radius: 20px;
    padding: 14px 16px;
}
.kpi-label {
    font-size: 0.78rem;
    text-transform: uppercase;
    color: var(--ink-soft) !important;
    letter-spacing: 0.08em;
    opacity: 1 !important;
}
.kpi-value {
    font-size: 1.75rem;
    font-weight: 700;
    color: var(--ink-accent) !important;
    margin-top: 6px;
    opacity: 1 !important;
}
.section-note {
    background: var(--panel-bg);
    border: 1px solid var(--panel-border);
    border-radius: 18px;
    padding: 14px 16px;
    color: var(--ink-muted);
}
.section-note,
.section-note * {
    color: var(--ink-strong) !important;
    opacity: 1 !important;
}
.section-note p,
.section-note li,
.section-note span,
.section-note strong,
.section-note code {
    color: var(--ink-strong) !important;
}
.section-note ul,
.section-note ol {
    margin: 0.25rem 0 0.25rem 1.2rem !important;
}
.section-note li::marker {
    color: #7a5a2f !important;
}

.kpi-card,
.kpi-card * {
    color: var(--ink-strong) !important;
    opacity: 1 !important;
}

.gradio-container .table-wrap th {
    color: var(--ink-accent) !important;
    opacity: 1 !important;
}

.gradio-container .table-wrap td,
.gradio-container .dataframe tbody td {
    color: #f7f3ec !important;
    opacity: 1 !important;
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


def build_simulation_figure(simulated: pd.DataFrame) -> go.Figure:
    """Create the simulated municipal resource splitup chart."""
    if simulated.empty or "recommended_pct" not in simulated.columns:
        return empty_figure("Simulated Resource Splitup", "No simulated allocation data is available.")
    ordered = simulated.sort_values("recommended_pct", ascending=True)
    fig = go.Figure(
        go.Bar(
            x=ordered["recommended_pct"],
            y=ordered["sector_display"],
            orientation="h",
            marker_color="#0b4f6c",
            text=ordered["recommended_pct"].map(lambda value: f"{value:.1f}%"),
            textposition="auto",
            hovertemplate="%{y}<br>Simulated split: %{x:.2f}%<extra></extra>",
        )
    )
    fig.update_layout(height=320, title="Simulated Resource Splitup", xaxis_title="Percent", yaxis_title="", **PLOT_LAYOUT)
    return fig


def build_simulation_table(default_allocation: pd.DataFrame, simulated: pd.DataFrame) -> pd.DataFrame:
    """Build a side-by-side comparison of default and simulated allocation."""
    columns = ["Sector", "Current Load %", "Current Recommended %", "Simulated Recommended %", "Change in Allocation"]
    required = {"sector", "sector_display", "current_load_pct", "recommended_pct"}
    if default_allocation.empty or simulated.empty or not required.issubset(default_allocation.columns) or not required.issubset(simulated.columns):
        return pd.DataFrame(columns=columns)

    default = default_allocation[["sector", "sector_display", "current_load_pct", "recommended_pct"]].rename(
        columns={"recommended_pct": "default_recommended_pct"}
    )
    sim = simulated[["sector", "recommended_pct"]].rename(columns={"recommended_pct": "simulated_pct"})
    comparison = default.merge(sim, on="sector", how="left")
    comparison["simulated_pct"] = comparison["simulated_pct"].fillna(0.0)
    comparison["delta_pp"] = (comparison["simulated_pct"] - comparison["default_recommended_pct"]).round(2)
    readable = comparison.rename(
        columns={
            "sector_display": "Sector",
            "current_load_pct": "Current Load %",
            "default_recommended_pct": "Current Recommended %",
            "simulated_pct": "Simulated Recommended %",
            "delta_pp": "Change in Allocation",
        }
    )
    return readable[columns].round(
        {
            "Current Load %": 2,
            "Current Recommended %": 2,
            "Simulated Recommended %": 2,
            "Change in Allocation": 2,
        }
    )


def build_simulation_explanation(comparison: pd.DataFrame, demand_changes: dict[str, float], floor_pct: float) -> str:
    """Explain the largest movement caused by simple demand-change settings."""
    if comparison.empty:
        return "No simulation rows are available for the active filters."

    increased = comparison.sort_values("Change in Allocation", ascending=False).iloc[0]
    decreased = comparison.sort_values("Change in Allocation", ascending=True).iloc[0]
    readable_changes = {
        SECTOR_DISPLAY_NAMES.get(sector, sector): float(change)
        for sector, change in demand_changes.items()
    }
    demand_up = max(readable_changes.items(), key=lambda item: item[1])
    demand_down = min(readable_changes.items(), key=lambda item: item[1])
    if all(abs(value) < 1e-9 for value in readable_changes.values()):
        demand_sentence = "No complaint demand changes were applied, so the simulated split matches the current recommendation."
    else:
        demand_sentence = (
            f"The biggest complaint increase is {demand_up[0]} at {demand_up[1]:+.0f}%, "
            f"and the biggest decrease is {demand_down[0]} at {demand_down[1]:+.0f}%."
        )
    return (
        f"{demand_sentence} The minimum guaranteed allocation is {floor_pct:.1f}% per sector. "
        f"The largest allocation gain is {increased['Sector']} at {increased['Change in Allocation']:+.2f} percentage points; "
        f"the largest allocation drop is {decreased['Sector']} at {decreased['Change in Allocation']:+.2f} percentage points."
    )


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


def build_benchmark_figure(benchmark: pd.DataFrame) -> go.Figure:
    """Create a grouped bar chart comparing allocation strategies."""
    required = {"sector_display", "Equal Split", "Count-Only", "Model (4-Factor)"}
    if benchmark.empty or not required.issubset(benchmark.columns):
        return empty_figure("Allocation Benchmark", "No benchmark data is available for the current filter set.")

    strategies = ["Equal Split", "Count-Only", "Model (4-Factor)"]
    colors = {
        "Equal Split": "#667785",
        "Count-Only": "#f2c14e",
        "Model (4-Factor)": "#0b4f6c",
    }
    fig = go.Figure()
    for strategy in strategies:
        fig.add_trace(
            go.Bar(
                name=strategy,
                x=benchmark["sector_display"],
                y=benchmark[strategy],
                marker_color=colors[strategy],
                text=benchmark[strategy].map(lambda value: f"{value:.1f}%"),
                textposition="outside",
            )
        )

    max_value = float(benchmark[strategies].max().max())
    fig.update_layout(
        barmode="group",
        height=460,
        title="Allocation Benchmark",
        xaxis_title="",
        yaxis_title="Allocation percent",
        yaxis=dict(range=[0, max(20.0, max_value * 1.2)]),
        legend_title="Strategy",
        **PLOT_LAYOUT,
    )
    return fig


def build_transfer_confusion_matrix(frame: pd.DataFrame) -> go.Figure:
    """Create a row-normalized expected-vs-predicted transfer heatmap."""
    required = {"expected_nyc_label", "predicted_nyc_label"}
    if frame.empty or not required.issubset(frame.columns):
        return empty_figure("Kerala Transfer Confusion Matrix", "Kerala prediction columns are not available.")

    subset = frame.dropna(subset=["expected_nyc_label", "predicted_nyc_label"])
    if subset.empty:
        return empty_figure("Kerala Transfer Confusion Matrix", "No Kerala transfer rows are available.")

    labels = sorted(set(subset["expected_nyc_label"].astype(str)) | set(subset["predicted_nyc_label"].astype(str)))
    matrix = pd.crosstab(
        subset["expected_nyc_label"].astype(str),
        subset["predicted_nyc_label"].astype(str),
        normalize="index",
    ).reindex(index=labels, columns=labels, fill_value=0.0)

    fig = go.Figure(
        data=go.Heatmap(
            z=matrix.to_numpy(),
            x=matrix.columns,
            y=matrix.index,
            colorscale=[[0.0, "#fff4e6"], [0.5, "#f2c14e"], [1.0, "#0b4f6c"]],
            colorbar=dict(title="Share"),
            hovertemplate="Expected: %{y}<br>Predicted: %{x}<br>Share: %{z:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        height=560,
        title="Kerala Transfer Confusion Matrix",
        xaxis_title="Predicted NYC label",
        yaxis_title="Expected NYC label",
        xaxis_tickangle=35,
        **PLOT_LAYOUT,
    )
    return fig


def build_confidence_distribution(frame: pd.DataFrame) -> go.Figure:
    """Create confidence histograms for correct and incorrect transfer predictions."""
    required = {"prediction_confidence", "correct_reference"}
    if frame.empty or not required.issubset(frame.columns):
        return empty_figure("Prediction Confidence Distribution", "Kerala confidence columns are not available.")

    subset = frame.dropna(subset=["prediction_confidence", "correct_reference"])
    if subset.empty:
        return empty_figure("Prediction Confidence Distribution", "No confidence rows are available.")

    correct = subset[subset["correct_reference"] == True]["prediction_confidence"]
    incorrect = subset[subset["correct_reference"] == False]["prediction_confidence"]
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=correct, name="Correct", nbinsx=30, opacity=0.72, marker_color="#2a9d8f"))
    fig.add_trace(go.Histogram(x=incorrect, name="Incorrect", nbinsx=30, opacity=0.72, marker_color="#bc4749"))
    fig.add_vline(x=0.8, line_dash="dash", line_color="#667785")
    fig.update_layout(
        barmode="overlay",
        height=420,
        title="Prediction Confidence Distribution",
        xaxis_title="Top prediction confidence",
        yaxis_title="Rows",
        **PLOT_LAYOUT,
    )
    return fig


def build_district_accuracy_figure(frame: pd.DataFrame) -> go.Figure:
    """Create a district-level Kerala transfer accuracy chart."""
    required = {"city_or_district", "correct_reference"}
    if frame.empty or not required.issubset(frame.columns):
        return empty_figure("District Transfer Accuracy", "Kerala district accuracy columns are not available.")

    subset = frame.dropna(subset=["city_or_district", "correct_reference"])
    if subset.empty:
        return empty_figure("District Transfer Accuracy", "No district-level transfer rows are available.")

    accuracy = (
        subset.groupby("city_or_district")["correct_reference"]
        .agg(accuracy="mean", rows="count")
        .reset_index()
        .rename(columns={"city_or_district": "district"})
        .sort_values("accuracy", ascending=True)
    )
    fig = px.bar(
        accuracy,
        x="accuracy",
        y="district",
        orientation="h",
        color="accuracy",
        color_continuous_scale=["#bc4749", "#f2c14e", "#2a9d8f"],
        range_color=[0.0, 1.0],
        hover_data={"rows": True, "accuracy": ":.3f"},
        title="District Transfer Accuracy",
    )
    fig.update_layout(
        height=520,
        xaxis_title="Accuracy",
        xaxis_tickformat=".0%",
        yaxis_title="",
        coloraxis_showscale=False,
        **PLOT_LAYOUT,
    )
    return fig


def empty_figure(title: str, message: str) -> go.Figure:
    """Return a placeholder figure with a centered message."""
    fig = go.Figure()
    fig.add_annotation(text=message, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False, font=dict(size=15))
    fig.update_layout(title=title, xaxis_visible=False, yaxis_visible=False, height=340, **PLOT_LAYOUT)
    return fig
