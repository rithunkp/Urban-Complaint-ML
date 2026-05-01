from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.domain import DOMAIN_KERALA, DOMAIN_NYC, SECTOR_DISPLAY_NAMES


def parse_date_input(value: str | pd.Timestamp) -> pd.Timestamp:
    """Parse a text or timestamp value into a pandas timestamp."""
    if isinstance(value, pd.Timestamp):
        return value
    return pd.to_datetime(value, errors="coerce")


def domain_region_label(domain: str) -> str:
    """Return the human-friendly label for region filters."""
    return "District" if domain == DOMAIN_KERALA else "Borough"


def filter_dataset(
    frame: pd.DataFrame,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    regions: list[str] | None,
    sectors: list[str] | None,
    complaint_types: list[str] | None,
) -> pd.DataFrame:
    """Apply the dashboard filters in a deterministic order."""
    start_ts = parse_date_input(start_date)
    end_ts = parse_date_input(end_date)
    if pd.isna(start_ts) or pd.isna(end_ts):
        raise ValueError("Dates must be valid and in YYYY-MM-DD format.")

    filtered = frame[(frame["created_at"] >= start_ts) & (frame["created_at"] <= end_ts + pd.Timedelta(days=1))].copy()
    if regions is not None:
        filtered = filtered[filtered["region"].isin(regions)].copy()
    if sectors is not None:
        filtered = filtered[filtered["sector"].isin(sectors)].copy()
    if complaint_types is not None:
        filtered = filtered[filtered["complaint_type"].isin(complaint_types)].copy()
    return filtered


def compute_summary_metrics(filtered: pd.DataFrame) -> dict[str, str]:
    """Compute high-level KPI card values for the active slice."""
    if filtered.empty:
        return {
            "complaints": "0",
            "complaint_types": "0",
            "regions": "0",
            "median_closure": "Unavailable",
        }

    median_closure = filtered["closure_hours"].dropna().median()
    return {
        "complaints": f"{len(filtered):,}",
        "complaint_types": f"{filtered['complaint_type'].nunique()}",
        "regions": f"{filtered['region'].nunique()}",
        "median_closure": f"{median_closure:.1f} hrs" if pd.notna(median_closure) else "Unavailable",
    }


def complaint_distribution(filtered: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Return the top complaint types and their shares."""
    if filtered.empty:
        return pd.DataFrame(columns=["complaint_type", "count", "share"])

    counts = filtered["complaint_type"].value_counts().head(top_n).rename_axis("complaint_type").reset_index(name="count")
    counts["share"] = counts["count"] / counts["count"].sum() * 100
    return counts


def sector_distribution(filtered: pd.DataFrame) -> pd.DataFrame:
    """Return sector-level counts and shares."""
    if filtered.empty:
        return pd.DataFrame(columns=["sector", "count", "share", "sector_display"])
    counts = filtered["sector"].value_counts().rename_axis("sector").reset_index(name="count")
    counts["share"] = counts["count"] / counts["count"].sum() * 100
    counts["sector_display"] = counts["sector"].map(SECTOR_DISPLAY_NAMES)
    return counts


def aggregate_time_series(filtered: pd.DataFrame, granularity: str) -> pd.DataFrame:
    """Aggregate complaints over time at the chosen granularity."""
    if filtered.empty:
        return pd.DataFrame(columns=["period", "count"])
    rule_map = {"Daily": "D", "Monthly": "MS", "Yearly": "YS"}
    aggregated = (
        filtered.set_index("created_at")
        .sort_index()
        .resample(rule_map[granularity])
        .size()
        .reset_index(name="count")
        .rename(columns={"created_at": "period"})
    )
    return aggregated


def detect_anomalies(time_series: pd.DataFrame, granularity: str, threshold: float) -> pd.DataFrame:
    """Detect explainable timeline anomalies with rolling z-scores."""
    if time_series.empty:
        return pd.DataFrame(columns=["period", "count", "rolling_mean", "rolling_std", "zscore", "is_anomaly"])

    window_lookup = {"Daily": 14, "Monthly": 6, "Yearly": 4}
    window = window_lookup[granularity]
    timeline = time_series.copy().sort_values("period")
    rolling_mean = timeline["count"].rolling(window=window, min_periods=2).mean()
    rolling_std = timeline["count"].rolling(window=window, min_periods=2).std().replace(0, np.nan)
    timeline["rolling_mean"] = rolling_mean.fillna(timeline["count"])
    timeline["rolling_std"] = rolling_std.fillna(1.0)
    timeline["zscore"] = ((timeline["count"] - timeline["rolling_mean"]) / timeline["rolling_std"]).fillna(0.0)
    timeline["is_anomaly"] = timeline["zscore"].abs() >= threshold
    return timeline


def build_anomaly_events(filtered: pd.DataFrame, anomaly_frame: pd.DataFrame) -> pd.DataFrame:
    """Build a table of anomaly events using the next true time bucket as the boundary."""
    if filtered.empty or anomaly_frame.empty:
        return pd.DataFrame(columns=["date", "count", "zscore", "dominant_type", "dominant_sector"])

    timeline = anomaly_frame.sort_values("period").reset_index(drop=True).copy()
    timeline["next_period"] = timeline["period"].shift(-1)
    flagged = timeline[timeline["is_anomaly"]].copy()
    if flagged.empty:
        return pd.DataFrame(columns=["date", "count", "zscore", "dominant_type", "dominant_sector"])

    rows = []
    inferred_step = timeline["period"].diff().dropna().median() if len(timeline) > 1 else pd.Timedelta(days=1)
    for _, row in flagged.sort_values("count", ascending=False).head(12).iterrows():
        window_start = pd.Timestamp(row["period"])
        window_end = pd.Timestamp(row["next_period"]) if pd.notna(row["next_period"]) else window_start + inferred_step
        window = filtered[(filtered["created_at"] >= window_start) & (filtered["created_at"] < window_end)]
        dominant_type = window["complaint_type"].mode().iloc[0] if not window.empty else "Unknown"
        dominant_sector = window["sector"].mode().iloc[0] if not window.empty else "Unknown"
        rows.append(
            {
                "date": window_start.strftime("%Y-%m-%d"),
                "count": int(row["count"]),
                "zscore": round(float(row["zscore"]), 2),
                "dominant_type": dominant_type,
                "dominant_sector": SECTOR_DISPLAY_NAMES.get(dominant_sector, dominant_sector),
            }
        )
    return pd.DataFrame(rows)


def region_sector_matrix(filtered: pd.DataFrame) -> pd.DataFrame:
    """Return a region-by-sector matrix for heatmaps and tables."""
    if filtered.empty:
        return pd.DataFrame(columns=["region", "sector", "count"])
    frame = filtered.groupby(["region", "sector"]).size().reset_index(name="count")
    frame["sector_display"] = frame["sector"].map(SECTOR_DISPLAY_NAMES)
    return frame


def generate_insights(filtered: pd.DataFrame, anomaly_frame: pd.DataFrame, domain: str) -> str:
    """Build short, non-technical insight text for the active dataset slice."""
    if filtered.empty:
        return "No complaints match the active filters. Expand the date range or add back sectors or complaint types."

    region_name = domain_region_label(domain)
    top_type = filtered["complaint_type"].value_counts().idxmax()
    top_region = filtered["region"].value_counts().idxmax()
    top_sector = filtered["sector"].value_counts().idxmax()
    lines = [
        f"- Leading complaint type: **{top_type}**",
        f"- Highest complaint {region_name.lower()}: **{top_region}**",
        f"- Highest pressure sector: **{SECTOR_DISPLAY_NAMES.get(top_sector, top_sector)}**",
    ]
    flagged = anomaly_frame[anomaly_frame["is_anomaly"]]
    if not flagged.empty:
        strongest = flagged.sort_values("count", ascending=False).iloc[0]
        lines.append(
            f"- Strongest anomaly: **{pd.Timestamp(strongest['period']).strftime('%b %d, %Y')}** with **{int(strongest['count']):,}** requests"
        )
    else:
        lines.append("- Anomaly radar: no strong spikes above the current threshold")
    return "\n".join(lines)


def transfer_examples(evaluated_kerala: pd.DataFrame, limit: int = 12) -> pd.DataFrame:
    """Return user-facing transfer examples for the Kerala evaluation tab."""
    if evaluated_kerala.empty:
        return pd.DataFrame(
            columns=[
                "region",
                "localized_complaint_type",
                "predicted_localized_label",
                "prediction_confidence",
                "sector",
            ]
        )

    columns = [
        "region",
        "localized_complaint_type",
        "predicted_localized_label",
        "prediction_confidence",
        "sector",
    ]
    if "correct_reference" in evaluated_kerala.columns:
        columns.append("correct_reference")
    sample = evaluated_kerala[columns].sample(min(limit, len(evaluated_kerala)), random_state=42)
    sample["prediction_confidence"] = sample["prediction_confidence"].map(lambda value: round(float(value), 3))
    sample["sector"] = sample["sector"].map(SECTOR_DISPLAY_NAMES)
    return sample.reset_index(drop=True)


def write_filtered_download(filtered: pd.DataFrame, domain: str) -> str:
    """Persist the filtered slice as a temporary CSV and return the path."""
    temp_dir = Path(tempfile.gettempdir())
    path = temp_dir / f"civic_intelligence_{domain.lower()}_filtered.csv"
    filtered.to_csv(path, index=False)
    return str(path)
