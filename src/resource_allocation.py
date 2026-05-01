from __future__ import annotations

import numpy as np
import pandas as pd

from src.domain import SECTOR_DISPLAY_NAMES, SECTOR_ORDER


def safe_normalize(series: pd.Series) -> pd.Series:
    """Normalize a non-negative series into the [0, 1] range safely."""
    filled = series.fillna(0.0).clip(lower=0.0)
    maximum = filled.max()
    if maximum <= 0:
        return pd.Series(0.0, index=filled.index)
    return filled / maximum


def compute_resource_split(filtered: pd.DataFrame, anomaly_frame: pd.DataFrame) -> pd.DataFrame:
    """Compute explainable sector-wise allocation with a minimum fairness floor."""
    base = pd.DataFrame({"sector": SECTOR_ORDER})
    if filtered.empty:
        base["current_load_pct"] = 0.0
        base["recommended_pct"] = round(100.0 / len(base), 2)
        base["resource_score"] = 0.0
        base["sector_display"] = base["sector"].map(SECTOR_DISPLAY_NAMES)
        return base

    total_rows = len(filtered)
    current_share = filtered["sector"].value_counts(normalize=True).reindex(SECTOR_ORDER, fill_value=0.0)

    latest_date = filtered["created_at"].max()
    recent = filtered[filtered["created_at"] >= latest_date - pd.Timedelta(days=30)]
    previous = filtered[
        filtered["created_at"].between(latest_date - pd.Timedelta(days=60), latest_date - pd.Timedelta(days=30))
    ]
    recent_share = recent["sector"].value_counts(normalize=True).reindex(SECTOR_ORDER, fill_value=0.0)
    previous_share = previous["sector"].value_counts(normalize=True).reindex(SECTOR_ORDER, fill_value=0.0)
    growth = ((recent_share - previous_share) / previous_share.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    positive_growth = growth.clip(lower=0.0)

    anomaly_counts = pd.Series(0.0, index=SECTOR_ORDER)
    if not anomaly_frame.empty and anomaly_frame["is_anomaly"].any():
        flagged_periods = anomaly_frame.sort_values("period").reset_index(drop=True).copy()
        flagged_periods["next_period"] = flagged_periods["period"].shift(-1)
        inferred_step = flagged_periods["period"].diff().dropna().median() if len(flagged_periods) > 1 else pd.Timedelta(days=1)
        flagged_periods = flagged_periods[flagged_periods["is_anomaly"]]
        for _, row in flagged_periods.iterrows():
            period_start = row["period"]
            period_end = row["next_period"] if pd.notna(row["next_period"]) else period_start + inferred_step
            slice_frame = filtered[(filtered["created_at"] >= period_start) & (filtered["created_at"] < period_end)]
            sector_counts = slice_frame["sector"].value_counts()
            anomaly_counts = anomaly_counts.add(sector_counts.reindex(SECTOR_ORDER, fill_value=0.0), fill_value=0.0)
    anomaly_pressure = anomaly_counts / max(anomaly_counts.sum(), 1.0)

    closure_median = filtered.groupby("sector")["closure_hours"].median().reindex(SECTOR_ORDER, fill_value=0.0)
    closure_delay = safe_normalize(closure_median.fillna(0.0))

    recent_component = safe_normalize(recent_share)
    growth_component = safe_normalize(positive_growth)
    anomaly_component = safe_normalize(anomaly_pressure)

    score = (
        0.50 * recent_component
        + 0.20 * growth_component
        + 0.20 * anomaly_component
        + 0.10 * closure_delay
    )

    floor = 0.05
    raw = score / max(score.sum(), 1e-9)
    adjusted = floor + (1.0 - floor * len(SECTOR_ORDER)) * raw
    adjusted = adjusted / adjusted.sum()

    base["current_load_pct"] = (current_share.reindex(SECTOR_ORDER).fillna(0.0) * 100).round(2).to_numpy()
    base["recommended_pct"] = (adjusted.reindex(SECTOR_ORDER).fillna(0.0) * 100).round(2).to_numpy()
    base["resource_score"] = score.reindex(SECTOR_ORDER).fillna(0.0).round(4).to_numpy()
    base["recent_share_component"] = recent_component.reindex(SECTOR_ORDER).fillna(0.0).round(4).to_numpy()
    base["growth_component"] = growth_component.reindex(SECTOR_ORDER).fillna(0.0).round(4).to_numpy()
    base["anomaly_component"] = anomaly_component.reindex(SECTOR_ORDER).fillna(0.0).round(4).to_numpy()
    base["closure_component"] = closure_delay.reindex(SECTOR_ORDER).fillna(0.0).round(4).to_numpy()
    base["sector_display"] = base["sector"].map(SECTOR_DISPLAY_NAMES)
    return base


def build_resource_explanation(allocation: pd.DataFrame) -> str:
    """Create a user-friendly explanation of the resource formula."""
    leader = allocation.sort_values("recommended_pct", ascending=False).iloc[0]
    return (
        "Recommended split uses a transparent pressure score: 50% recent complaint share, "
        "20% positive trend growth, 20% anomaly pressure, and 10% closure-delay pressure. "
        "Every sector keeps a minimum 5% floor so no service area is neglected. "
        f"Right now, {leader['sector_display']} receives the highest recommendation at {leader['recommended_pct']:.1f}%."
    )
