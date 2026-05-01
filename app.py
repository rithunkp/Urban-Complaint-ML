from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import gradio as gr
import pandas as pd

from src.analytics import (
    aggregate_time_series,
    build_anomaly_events,
    complaint_distribution,
    compute_summary_metrics,
    detect_anomalies,
    domain_region_label,
    filter_dataset,
    generate_insights,
    region_sector_matrix,
    sector_distribution,
    write_filtered_download,
)
from src.data_pipeline import load_processed_dataframe
from src.domain import DOMAIN_KERALA, DOMAIN_NYC, localized_label_for_prediction
from src.modeling import load_model_bundle, predict_request
from src.paths import ARTIFACTS_DIR, PROCESSED_DIR, REPORTS_DIR
from src.resource_allocation import build_resource_explanation, compute_resource_split
from src.ui_components import (
    APP_CSS,
    build_distribution_figure,
    build_geo_figure,
    build_prediction_figure,
    build_region_sector_heatmap,
    build_resource_pie,
    build_sector_figure,
    build_timeline_figure,
    build_top_categories_figure,
    hero_html,
    kpi_html,
)


@lru_cache(maxsize=1)
def load_runtime_assets() -> dict[str, Any]:
    """Load processed datasets, reports, and model artifacts for the Gradio app."""
    required_paths = [
        PROCESSED_DIR / "nyc_runtime.csv.gz",
        PROCESSED_DIR / "kerala_transfer_evaluated.csv.gz",
        ARTIFACTS_DIR / "complaint_model_bundle.joblib",
        REPORTS_DIR / "nyc_metrics.json",
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Runtime assets are missing. Run `python scripts/train_nyc_model.py` and "
            "`python scripts/generate_kerala_transfer_set.py` first.\nMissing:\n- "
            + "\n- ".join(missing)
        )

    nyc_frame = load_processed_dataframe("nyc_runtime.csv.gz", parse_dates=["created_at", "closed_at"])
    kerala_frame = load_processed_dataframe("kerala_transfer_evaluated.csv.gz", parse_dates=["created_at", "closed_at"])
    bundle = load_model_bundle()
    nyc_metrics = json.loads((REPORTS_DIR / "nyc_metrics.json").read_text(encoding="utf-8"))
    kerala_metrics_path = REPORTS_DIR / "kerala_transfer_metrics.json"
    kerala_metrics = json.loads(kerala_metrics_path.read_text(encoding="utf-8")) if kerala_metrics_path.exists() else {}
    return {
        "nyc": nyc_frame,
        "kerala": kerala_frame,
        "bundle": bundle,
        "nyc_metrics": nyc_metrics,
        "kerala_metrics": kerala_metrics,
    }


def dataset_for_domain(domain: str) -> pd.DataFrame:
    """Return the active domain dataframe."""
    assets = load_runtime_assets()
    return assets["kerala"].copy() if domain == DOMAIN_KERALA else assets["nyc"].copy()


def default_date_range(domain: str) -> tuple[str, str]:
    """Return default date strings for a domain."""
    frame = dataset_for_domain(domain)
    start = pd.Timestamp(frame["created_at"].min()).strftime("%Y-%m-%d")
    end = pd.Timestamp(frame["created_at"].max()).strftime("%Y-%m-%d")
    return start, end


def update_filter_controls(domain: str) -> tuple[gr.Dropdown, gr.Dropdown, gr.Dropdown, str, str]:
    """Update region, sector, and complaint-type choices when the domain changes."""
    frame = dataset_for_domain(domain)
    regions = sorted(frame["region"].dropna().astype(str).unique().tolist())
    sectors = sorted(frame["sector"].dropna().astype(str).unique().tolist())
    complaint_types = sorted(frame["complaint_type"].dropna().astype(str).unique().tolist())
    start, end = default_date_range(domain)
    return (
        gr.update(choices=regions, value=regions, label=domain_region_label(domain)),
        gr.update(choices=sectors, value=sectors, label="Municipal sector"),
        gr.update(choices=complaint_types, value=complaint_types[: min(10, len(complaint_types))], label="Complaint type"),
        start,
        end,
    )


def transfer_summary_markdown(domain: str) -> str:
    """Build the Kerala transfer summary panel."""
    assets = load_runtime_assets()
    if domain != DOMAIN_KERALA:
        metrics = assets["nyc_metrics"]
        return (
            "### Model backbone summary\n"
            "- This product is branded for Kerala operations, but its complaint understanding layer was trained on NYC 311 as the reference dataset.\n"
            f"- Accuracy: **{metrics['accuracy']:.3f}**\n"
            f"- Macro F1: **{metrics['macro_f1']:.3f}**\n"
            f"- Top-3 Accuracy: **{metrics['top_3_accuracy']:.3f}**\n"
            "- Switch to **Kerala Operations View** to inspect district-level behavior and transfer performance."
        )

    metrics = assets["kerala_metrics"]
    if not metrics:
        return "### Kerala operations readiness\nNo saved Kerala evaluation report was found."
    return (
        "### Kerala operations readiness\n"
        f"- Rows evaluated: **{metrics.get('rows', 0):,}**\n"
        f"- Reference-label coverage: **{metrics.get('reference_label_coverage', 0.0):.1%}**\n"
        f"- Accuracy on covered reference labels: **{metrics.get('accuracy_against_reference', 0.0):.3f}**\n"
        f"- Macro F1 on covered reference labels: **{metrics.get('macro_f1_against_reference', 0.0):.3f}**\n"
        f"- Top-3 Accuracy on covered reference labels: **{metrics.get('top_3_accuracy_against_reference', 0.0):.3f}**"
    )


def update_dashboard(
    domain: str,
    start_date: str,
    end_date: str,
    regions: list[str],
    sectors: list[str],
    complaint_types: list[str],
    anomaly_threshold: float,
    granularity: str,
) -> tuple[Any, ...]:
    """Compute all dashboard outputs from the active filters."""
    frame = dataset_for_domain(domain)
    filtered = filter_dataset(frame, start_date, end_date, regions, sectors, complaint_types)
    summary = compute_summary_metrics(filtered)
    mix = complaint_distribution(filtered)
    timeline = aggregate_time_series(filtered, granularity)
    anomalies = detect_anomalies(timeline, granularity, anomaly_threshold)
    anomaly_events = build_anomaly_events(filtered, anomalies)
    sector_load = sector_distribution(filtered)
    allocation = compute_resource_split(filtered, anomalies)
    matrix = region_sector_matrix(filtered)
    insights = generate_insights(filtered, anomalies, domain)
    download_path = write_filtered_download(filtered, domain)

    explorer_columns = [
        "created_at",
        "region",
        "complaint_type",
        "localized_complaint_type",
        "sector",
        "status",
        "closure_hours",
    ]
    explorer = filtered[explorer_columns].head(300).copy() if not filtered.empty else pd.DataFrame(columns=explorer_columns)
    return (
        kpi_html(summary),
        insights,
        transfer_summary_markdown(domain),
        build_distribution_figure(mix),
        build_top_categories_figure(mix),
        build_timeline_figure(anomalies, anomaly_threshold),
        anomaly_events,
        build_sector_figure(sector_load, "share", "Current Sector Load"),
        build_resource_pie(allocation),
        build_region_sector_heatmap(matrix, domain),
        build_geo_figure(filtered, domain),
        allocation[["sector_display", "current_load_pct", "recommended_pct", "resource_score"]],
        build_resource_explanation(allocation),
        explorer,
        download_path,
    )


def run_demo_prediction(domain: str, descriptor: str, location_type: str) -> tuple[str, str, str, Any]:
    """Run the ML demo and format its outputs for the UI."""
    bundle = load_runtime_assets()["bundle"]
    result = predict_request(bundle, descriptor=descriptor, location_type=location_type)
    display_label = result["localized_label"] if domain == DOMAIN_KERALA else result["predicted_nyc_label"]
    sector_display = result["sector"].replace("_", " ").title()
    confidence = f"{result['confidence']:.1%}"
    return display_label, sector_display, confidence, build_prediction_figure(result["top_predictions"])


def build_app() -> gr.Blocks:
    """Construct the full Gradio Blocks app."""
    with gr.Blocks(title="Kerala Urban Complaint Analysis for Smart City Resource Optimization") as demo:
        gr.HTML(hero_html())
        gr.Markdown(
            "Kerala-first civic operations dashboard with district pressure tracking, anomaly alerts, complaint intelligence, "
            "and explainable municipal resource splitup recommendations."
        )

        with gr.Row():
            with gr.Column(scale=1, min_width=280):
                domain = gr.Radio(
                    choices=[
                        ("Kerala Operations View", DOMAIN_KERALA),
                        ("NYC Training Data", DOMAIN_NYC),
                    ],
                    value=DOMAIN_KERALA,
                    label="Operational view",
                )
                default_frame = dataset_for_domain(DOMAIN_KERALA)
                default_regions = sorted(default_frame["region"].dropna().astype(str).unique().tolist())
                default_sectors = sorted(default_frame["sector"].dropna().astype(str).unique().tolist())
                default_types = sorted(default_frame["complaint_type"].dropna().astype(str).unique().tolist())
                start_value, end_value = default_date_range(DOMAIN_KERALA)
                region_filter = gr.Dropdown(default_regions, value=default_regions, multiselect=True, label="District")
                sector_filter = gr.Dropdown(default_sectors, value=default_sectors, multiselect=True, label="Municipal sector")
                type_filter = gr.Dropdown(
                    default_types,
                    value=default_types[: min(10, len(default_types))],
                    multiselect=True,
                    label="Complaint type",
                )
                start_date = gr.Textbox(value=start_value, label="Start date", info="Format: YYYY-MM-DD")
                end_date = gr.Textbox(value=end_value, label="End date", info="Format: YYYY-MM-DD")
                anomaly_threshold = gr.Slider(1.5, 4.0, value=2.6, step=0.1, label="Anomaly sensitivity")
                granularity = gr.Radio(["Daily", "Monthly", "Yearly"], value="Daily", label="Trend granularity")
                refresh_button = gr.Button("Apply filters", variant="primary")
                download_file = gr.File(label="Download filtered CSV")
            with gr.Column(scale=3):
                kpi_cards = gr.HTML()
                insights_box = gr.Markdown(elem_classes=["section-note"])

                with gr.Tabs():
                    with gr.Tab("Overview"):
                        transfer_summary = gr.Markdown()
                        with gr.Row():
                            complaint_mix_plot = gr.Plot()
                            top_types_plot = gr.Plot()
                        trend_plot = gr.Plot()
                        geo_plot = gr.Plot()

                    with gr.Tab("Resource Splitup"):
                        with gr.Row():
                            current_sector_plot = gr.Plot()
                            resource_split_plot = gr.Plot()
                        region_sector_plot = gr.Plot()
                        allocation_table = gr.Dataframe(interactive=False)
                        allocation_explanation = gr.Markdown(elem_classes=["section-note"])

                    with gr.Tab("Prediction Demo"):
                        demo_descriptor = gr.Textbox(
                            label="Complaint description",
                            lines=4,
                            placeholder="Example: Large potholes on the main road after heavy rain near the bus stand.",
                        )
                        demo_location = gr.Textbox(
                            label="Location type",
                            value="Main Road",
                            placeholder="Example: Main Road / Junction / Residential Lane",
                        )
                        demo_button = gr.Button("Predict complaint", variant="primary")
                        with gr.Row():
                            predicted_label = gr.Textbox(label="Predicted complaint type")
                            predicted_sector = gr.Textbox(label="Predicted sector")
                            predicted_confidence = gr.Textbox(label="Top confidence")
                        prediction_plot = gr.Plot()

                    with gr.Tab("Data Explorer"):
                        anomaly_table = gr.Dataframe(interactive=False)
                        explorer_table = gr.Dataframe(interactive=False)

        domain.change(
            fn=update_filter_controls,
            inputs=domain,
            outputs=[region_filter, sector_filter, type_filter, start_date, end_date],
        )

        refresh_button.click(
            fn=update_dashboard,
            inputs=[domain, start_date, end_date, region_filter, sector_filter, type_filter, anomaly_threshold, granularity],
            outputs=[
                kpi_cards,
                insights_box,
                transfer_summary,
                complaint_mix_plot,
                top_types_plot,
                trend_plot,
                anomaly_table,
                current_sector_plot,
                resource_split_plot,
                region_sector_plot,
                geo_plot,
                allocation_table,
                allocation_explanation,
                explorer_table,
                download_file,
            ],
        )

        demo_button.click(
            fn=run_demo_prediction,
            inputs=[domain, demo_descriptor, demo_location],
            outputs=[predicted_label, predicted_sector, predicted_confidence, prediction_plot],
        )

        demo.load(
            fn=update_dashboard,
            inputs=[domain, start_date, end_date, region_filter, sector_filter, type_filter, anomaly_threshold, granularity],
            outputs=[
                kpi_cards,
                insights_box,
                transfer_summary,
                complaint_mix_plot,
                top_types_plot,
                trend_plot,
                anomaly_table,
                current_sector_plot,
                resource_split_plot,
                region_sector_plot,
                geo_plot,
                allocation_table,
                allocation_explanation,
                explorer_table,
                download_file,
            ],
        )

    return demo


if __name__ == "__main__":
    try:
        app = build_app()
        app.queue().launch(css=APP_CSS)
    except FileNotFoundError as error:
        fallback = gr.Blocks(title="Kerala Urban Complaint Analysis for Smart City Resource Optimization")
        with fallback:
            gr.Markdown("## Runtime assets are missing")
            gr.Markdown(f"```text\n{error}\n```")
            gr.Markdown(
                "Run these commands locally before launching the Gradio app:\n"
                "1. `python scripts/train_nyc_model.py`\n"
                "2. `python scripts/generate_kerala_transfer_set.py`\n"
                "3. `python app.py`"
            )
        fallback.launch(css=APP_CSS)
