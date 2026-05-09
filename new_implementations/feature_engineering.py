"""
src/feature_engineering.py

Engineered features for Urban Complaint ML.

All features produced by this module are safe to use at complaint-intake
time — no post-closure information is used by the classifier functions.

Fields that are ⚠️ POST-HOC (only for allocation / ops analysis):
    closure_hours, status, confidence_bin, correct_reference

Entry point:
    from src.feature_engineering import add_all_features
    df = add_all_features(df)

Or call individual group functions for targeted use:
    df = add_text_features(df)
    df = add_temporal_features(df)
    df = add_geographic_features(df)
    df = add_operational_features(df)
    df = add_transfer_features(df)   # Kerala layer only
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

# ── constants ────────────────────────────────────────────────────────────────

_URGENCY_RE = re.compile(
    r"\b(?:urgent|emergency|danger|immediate|critical|hazard|severe|risk)\b",
    re.IGNORECASE,
)
_LOCATION_REF_RE = re.compile(
    r"\b(?:near|junction|opposite|beside|adjacent|ward|road\s*no|plot|behind|in\s*front)\b",
    re.IGNORECASE,
)
_NEGATIVE_WORDS: frozenset[str] = frozenset({
    "broken", "damaged", "blocked", "overflowing", "failed",
    "missing", "burst", "leaking", "collapsed", "flooded",
    "pothole", "stagnant", "sewage", "defective", "non-functional",
    "clogged", "cracked", "eroded", "dead", "disconnected",
})
_COASTAL_DISTRICTS: frozenset[str] = frozenset({
    "alappuzha", "ernakulam", "thrissur", "kozhikode",
    "kannur", "kasaragod", "thiruvananthapuram", "kollam", "malappuram",
})
# ordinal 1 = most urban; lower is higher urban pressure
_DISTRICT_URBAN_RANK: dict[str, int] = {
    "ernakulam": 1, "thiruvananthapuram": 2, "kozhikode": 3,
    "thrissur": 4, "kollam": 5, "kannur": 6, "malappuram": 7,
    "palakkad": 8, "alappuzha": 9, "kottayam": 10,
    "idukki": 11, "wayanad": 12, "kasaragod": 13, "pathanamthitta": 14,
}
_MONSOON_MONTHS: frozenset[int] = frozenset({6, 7, 8, 9})
_FESTIVAL_MONTHS: frozenset[int] = frozenset({10, 11})
_MORNING_RUSH_HOURS: frozenset[int] = frozenset({7, 8, 9, 10})


# ── text features ─────────────────────────────────────────────────────────────

def add_text_features(df: pd.DataFrame, desc_col: str = "descriptor") -> pd.DataFrame:
    """
    Features derived from the complaint descriptor text.
    All intake-safe.

    Added columns
    -------------
    text_length           : character count
    word_count            : space-tokenised word count
    avg_word_length       : text_length / word_count
    has_urgency_keyword   : 1 if urgency term present
    has_location_reference: 1 if spatial reference term present
    exclamation_flag      : 1 if '!' appears
    negative_word_count   : count of damage/failure vocabulary words
    descriptor_is_short   : 1 if word_count < 5  (often ambiguous complaints)
    """
    d = df[desc_col].fillna("").astype(str)

    df["text_length"] = d.str.len()
    wc = d.str.split().str.len().clip(lower=1)
    df["word_count"] = wc
    df["avg_word_length"] = (df["text_length"] / wc).round(2)
    df["has_urgency_keyword"] = d.str.contains(_URGENCY_RE).astype(np.int8)
    df["has_location_reference"] = d.str.contains(_LOCATION_REF_RE).astype(np.int8)
    df["exclamation_flag"] = d.str.contains("!").astype(np.int8)
    df["negative_word_count"] = d.apply(
        lambda x: sum(w in _NEGATIVE_WORDS for w in x.lower().split())
    )
    df["descriptor_is_short"] = (df["word_count"] < 5).astype(np.int8)
    return df


# ── temporal features ─────────────────────────────────────────────────────────

def add_temporal_features(df: pd.DataFrame, ts_col: str = "created_at") -> pd.DataFrame:
    """
    Features derived from the complaint creation timestamp.
    All intake-safe.

    Added columns
    -------------
    hour_of_day       : 0–23
    day_of_week       : 0 = Monday, 6 = Sunday
    is_weekend        : 1 if Saturday or Sunday
    month             : 1–12
    is_monsoon        : 1 if June–September (flood/drainage relevance for Kerala)
    is_festival_season: 1 if October–November (Onam window; traffic/public-space spike)
    is_morning_rush   : 1 if 07:00–10:59
    """
    ts = pd.to_datetime(df[ts_col], errors="coerce")

    df["hour_of_day"] = ts.dt.hour
    df["day_of_week"] = ts.dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(np.int8)
    df["month"] = ts.dt.month
    df["is_monsoon"] = df["month"].isin(_MONSOON_MONTHS).astype(np.int8)
    df["is_festival_season"] = df["month"].isin(_FESTIVAL_MONTHS).astype(np.int8)
    df["is_morning_rush"] = df["hour_of_day"].isin(_MORNING_RUSH_HOURS).astype(np.int8)
    return df


# ── geographic features ───────────────────────────────────────────────────────

def add_geographic_features(
    df: pd.DataFrame,
    district_col: str = "city_or_district",
) -> pd.DataFrame:
    """
    Features derived from district name and coordinates.
    All intake-safe.

    Added columns
    -------------
    is_coastal_district      : 1 for coastal Kerala districts (flood/drainage risk)
    district_urbanization_rank: 1 (Ernakulam) → 14 (Pathanamthitta); lower = more urban
    has_coordinates          : 1 if lat + lon both non-null
    """
    d = df[district_col].str.lower().str.strip()

    df["is_coastal_district"] = d.isin(_COASTAL_DISTRICTS).astype(np.int8)
    df["district_urbanization_rank"] = (
        d.map(_DISTRICT_URBAN_RANK).fillna(7).astype(int)
    )
    lat_ok = df["latitude"].notna() if "latitude" in df.columns else pd.Series(False, index=df.index)
    lon_ok = df["longitude"].notna() if "longitude" in df.columns else pd.Series(False, index=df.index)
    df["has_coordinates"] = (lat_ok & lon_ok).astype(np.int8)
    return df


# ── operational features ──────────────────────────────────────────────────────

def add_operational_features(df: pd.DataFrame, channel_col: str = "channel") -> pd.DataFrame:
    """
    Features derived from intake channel.
    All intake-safe.

    Added columns
    -------------
    channel_is_digital: 1 if intake was mobile/online/app/web
    """
    ch = df[channel_col].fillna("").str.lower()
    df["channel_is_digital"] = ch.str.contains(
        r"mobile|online|app|web", regex=True
    ).astype(np.int8)
    return df


# ── transfer / Kerala-specific features ───────────────────────────────────────

def add_transfer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Features specific to the Kerala evaluated transfer set.
    Requires: event_name (from kerala_generator), is_coastal_district and
    is_monsoon (run add_geographic_features + add_temporal_features first).

    Added columns
    -------------
    event_window_active       : 1 if row falls inside a named event window
    coastal_monsoon_interaction: is_coastal × is_monsoon interaction term
    confidence_bin            : low / medium / high bucket of prediction_confidence
                                ⚠️  POST-HOC — do not use in the intake-time classifier
    """
    if "event_name" in df.columns:
        df["event_window_active"] = df["event_name"].notna().astype(np.int8)

    if "is_coastal_district" in df.columns and "is_monsoon" in df.columns:
        df["coastal_monsoon_interaction"] = (
            df["is_coastal_district"] * df["is_monsoon"]
        )

    if "prediction_confidence" in df.columns:
        df["confidence_bin"] = pd.cut(
            df["prediction_confidence"],
            bins=[0.0, 0.5, 0.8, 1.01],
            labels=["low", "medium", "high"],
            right=False,
        )

    return df


# ── composite entry point ─────────────────────────────────────────────────────

def add_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature groups in dependency order.
    Transfer features are skipped if the required columns are absent.
    """
    df = add_text_features(df)
    df = add_temporal_features(df)
    df = add_geographic_features(df)
    df = add_operational_features(df)
    df = add_transfer_features(df)
    return df


# ── feature catalogue ─────────────────────────────────────────────────────────

FEATURE_CATALOGUE: list[dict] = [
    # text
    {"name": "text_length",            "group": "text",      "leakage": False},
    {"name": "word_count",             "group": "text",      "leakage": False},
    {"name": "avg_word_length",        "group": "text",      "leakage": False},
    {"name": "has_urgency_keyword",    "group": "text",      "leakage": False},
    {"name": "has_location_reference", "group": "text",      "leakage": False},
    {"name": "exclamation_flag",       "group": "text",      "leakage": False},
    {"name": "negative_word_count",    "group": "text",      "leakage": False},
    {"name": "descriptor_is_short",    "group": "text",      "leakage": False},
    # temporal
    {"name": "hour_of_day",            "group": "temporal",  "leakage": False},
    {"name": "day_of_week",            "group": "temporal",  "leakage": False},
    {"name": "is_weekend",             "group": "temporal",  "leakage": False},
    {"name": "month",                  "group": "temporal",  "leakage": False},
    {"name": "is_monsoon",             "group": "temporal",  "leakage": False},
    {"name": "is_festival_season",     "group": "temporal",  "leakage": False},
    {"name": "is_morning_rush",        "group": "temporal",  "leakage": False},
    # geographic
    {"name": "is_coastal_district",       "group": "geographic", "leakage": False},
    {"name": "district_urbanization_rank","group": "geographic", "leakage": False},
    {"name": "has_coordinates",           "group": "geographic", "leakage": False},
    # operational
    {"name": "channel_is_digital",     "group": "operational","leakage": False},
    # transfer / Kerala
    {"name": "event_window_active",        "group": "transfer", "leakage": False},
    {"name": "coastal_monsoon_interaction","group": "transfer", "leakage": False},
    {"name": "confidence_bin",             "group": "transfer", "leakage": True},
]
