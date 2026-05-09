from __future__ import annotations

import numpy as np
import pandas as pd

from src.domain import SECTOR_DISPLAY_NAMES, SECTOR_ORDER

DEFAULT_ALLOCATION_WEIGHTS = {
    "recent": 50.0,
    "growth": 20.0,
    "anomaly": 20.0,
    "closure": 10.0,
}
DEFAULT_FLOOR_PCT = 5.0


def safe_normalize(series: pd.Series) -> pd.Series:
    """Normalize a non-negative series into the [0, 1] range safely."""
    filled = series.fillna(0.0).clip(lower=0.0)
    maximum = filled.max()
    if maximum <= 0:
        return pd.Series(0.0, index=filled.index)
    return filled / maximum


def _normalize_weights(weights: dict[str, float] | None) -> dict[str, float]:
    """Normalize weight percentages into fractions with a safe all-zero fallback."""
    source = DEFAULT_ALLOCATION_WEIGHTS if weights is None else weights
    raw = {
        "recent": max(float(source.get("recent", 0.0)), 0.0),
        "growth": max(float(source.get("growth", 0.0)), 0.0),
        "anomaly": max(float(source.get("anomaly", 0.0)), 0.0),
        "closure": max(float(source.get("closure", 0.0)), 0.0),
    }
    total = sum(raw.values())
    if total <= 0:
        return {key: 0.0 for key in raw}
    return {key: value / total for key, value in raw.items()}


def _apply_floor(raw: pd.Series, floor_pct: float) -> pd.Series:
    """Apply a sector floor to raw allocation fractions and normalize."""
    aligned = raw.reindex(SECTOR_ORDER, fill_value=0.0).clip(lower=0.0)
    total = aligned.sum()
    if total <= 0:
        aligned = pd.Series(1.0 / len(SECTOR_ORDER), index=SECTOR_ORDER)
    else:
        aligned = aligned / total

    max_floor = 1.0 / len(SECTOR_ORDER)
    floor = min(max(float(floor_pct), 0.0) / 100.0, max_floor)
    adjusted = floor + (1.0 - floor * len(SECTOR_ORDER)) * aligned
    return adjusted / max(adjusted.sum(), 1e-9)


def _allocation_components(filtered: pd.DataFrame, anomaly_frame: pd.DataFrame) -> dict[str, pd.Series]:
    """Compute all reusable sector pressure components."""
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
    return {
        "current_share": current_share,
        "recent": recent_component,
        "growth": growth_component,
        "anomaly": anomaly_component,
        "closure": closure_delay,
    }


def compute_resource_split_with_weights(
    filtered: pd.DataFrame,
    anomaly_frame: pd.DataFrame,
    weights: dict[str, float] | None = None,
    floor_pct: float = DEFAULT_FLOOR_PCT,
) -> pd.DataFrame:
    """Compute allocation using caller-provided pressure weights."""
    base = pd.DataFrame({"sector": SECTOR_ORDER})
    if filtered.empty:
        base["current_load_pct"] = 0.0
        base["recommended_pct"] = round(100.0 / len(base), 2)
        base["resource_score"] = 0.0
        base["recent_share_component"] = 0.0
        base["growth_component"] = 0.0
        base["anomaly_component"] = 0.0
        base["closure_component"] = 0.0
        base["sector_display"] = base["sector"].map(SECTOR_DISPLAY_NAMES)
        return base

    components = _allocation_components(filtered, anomaly_frame)
    normalized_weights = _normalize_weights(weights)
    score = (
        normalized_weights["recent"] * components["recent"]
        + normalized_weights["growth"] * components["growth"]
        + normalized_weights["anomaly"] * components["anomaly"]
        + normalized_weights["closure"] * components["closure"]
    )
    adjusted = _apply_floor(score, floor_pct)

    base["current_load_pct"] = (components["current_share"].reindex(SECTOR_ORDER).fillna(0.0) * 100).round(2).to_numpy()
    base["recommended_pct"] = (adjusted.reindex(SECTOR_ORDER).fillna(0.0) * 100).round(2).to_numpy()
    base["resource_score"] = score.reindex(SECTOR_ORDER).fillna(0.0).round(4).to_numpy()
    base["recent_share_component"] = components["recent"].reindex(SECTOR_ORDER).fillna(0.0).round(4).to_numpy()
    base["growth_component"] = components["growth"].reindex(SECTOR_ORDER).fillna(0.0).round(4).to_numpy()
    base["anomaly_component"] = components["anomaly"].reindex(SECTOR_ORDER).fillna(0.0).round(4).to_numpy()
    base["closure_component"] = components["closure"].reindex(SECTOR_ORDER).fillna(0.0).round(4).to_numpy()
    base["sector_display"] = base["sector"].map(SECTOR_DISPLAY_NAMES)
    return base


def compute_resource_split_with_demand_changes(
    filtered: pd.DataFrame,
    anomaly_frame: pd.DataFrame,
    demand_changes: dict[str, float] | None = None,
    floor_pct: float = DEFAULT_FLOOR_PCT,
) -> pd.DataFrame:
    """Compute allocation after applying simple sector-level demand changes."""
    default = compute_resource_split(filtered, anomaly_frame)
    if filtered.empty:
        return compute_resource_split_with_weights(filtered, anomaly_frame, floor_pct=floor_pct)

    changes = demand_changes or {}
    multipliers = pd.Series(
        {
            sector: max(0.0, 1.0 + float(changes.get(sector, 0.0)) / 100.0)
            for sector in SECTOR_ORDER
        },
        dtype=float,
    )
    default_scores = default.set_index("sector")["resource_score"].reindex(SECTOR_ORDER, fill_value=0.0)
    adjusted_scores = default_scores * multipliers
    adjusted = _apply_floor(adjusted_scores, floor_pct)

    simulated = default.copy()
    simulated["recommended_pct"] = (adjusted.reindex(SECTOR_ORDER).fillna(0.0) * 100).round(2).to_numpy()
    simulated["resource_score"] = adjusted_scores.reindex(SECTOR_ORDER).fillna(0.0).round(4).to_numpy()
    return simulated


def compute_resource_split(filtered: pd.DataFrame, anomaly_frame: pd.DataFrame) -> pd.DataFrame:
    """Compute explainable sector-wise allocation with a minimum fairness floor."""
    return compute_resource_split_with_weights(
        filtered,
        anomaly_frame,
        weights=DEFAULT_ALLOCATION_WEIGHTS,
        floor_pct=DEFAULT_FLOOR_PCT,
    )


def build_resource_explanation(allocation: pd.DataFrame) -> str:
    """Create a user-friendly explanation of the resource formula."""
    leader = allocation.sort_values("recommended_pct", ascending=False).iloc[0]
    return (
        "Recommended split uses a transparent pressure score: 50% recent complaint share, "
        "20% positive trend growth, 20% anomaly pressure, and 10% closure-delay pressure. "
        "Every sector keeps a minimum 5% floor so no service area is neglected. "
        f"Right now, {leader['sector_display']} receives the highest recommendation at {leader['recommended_pct']:.1f}%."
    )
