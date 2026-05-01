from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.domain import (
    CANONICAL_COLUMNS,
    DOMAIN_NYC,
    NYC_SOURCE_COLUMNS,
    map_sector_from_complaint,
    normalize_borough,
)
from src.paths import CACHE_DIR, PROCESSED_DIR, RAW_DIR, ensure_project_dirs

DATASET_SLUG = "new-york-city/ny-311-service-requests"
CACHE_VERSION = "v3"


def resolve_nyc_dataset_file() -> Path:
    """Resolve the NYC 311 CSV from the local raw directory or kagglehub."""
    local_files = sorted(RAW_DIR.glob("*.csv"))
    if local_files:
        return local_files[0]

    import kagglehub

    dataset_root = Path(kagglehub.dataset_download(DATASET_SLUG))
    csv_files = sorted(dataset_root.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError("NYC 311 dataset could not be located after kagglehub download.")
    return csv_files[0]


def canonicalize_text_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Clean nullable text columns consistently."""
    for column in columns:
        frame[column] = frame[column].fillna("Unknown").astype(str).str.strip()
    return frame


def build_text_input(frame: pd.DataFrame) -> pd.Series:
    """Construct classifier input text using intake-time fields only."""
    return (
        frame[["descriptor", "location_type"]]
        .fillna("")
        .agg(" ".join, axis=1)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def normalize_nyc_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """Normalize a raw NYC 311 chunk into the canonical schema."""
    frame = chunk.copy()
    frame.columns = [column.strip() for column in frame.columns]
    frame = frame.rename(
        columns={
            "Unique Key": "request_id",
            "Created Date": "created_at",
            "Closed Date": "closed_at",
            "Complaint Type": "complaint_type",
            "Descriptor": "descriptor",
            "Location Type": "location_type",
            "City": "city_or_district",
            "Borough": "region",
            "Latitude": "latitude",
            "Longitude": "longitude",
            "Status": "status",
            "Open Data Channel Type": "channel",
        }
    )

    frame["created_at"] = pd.to_datetime(frame["created_at"], errors="coerce")
    frame["closed_at"] = pd.to_datetime(frame["closed_at"], errors="coerce")
    frame = frame[frame["created_at"].notna()].copy()

    canonicalize_text_columns(
        frame,
        ["complaint_type", "descriptor", "location_type", "city_or_district", "status", "channel", "region"],
    )
    frame["region"] = frame["region"].map(normalize_borough)
    frame["city_or_district"] = frame["city_or_district"].where(
        frame["city_or_district"].ne("Unknown"),
        frame["region"],
    )
    frame["latitude"] = pd.to_numeric(frame["latitude"], errors="coerce")
    frame["longitude"] = pd.to_numeric(frame["longitude"], errors="coerce")

    closure_hours = (frame["closed_at"] - frame["created_at"]).dt.total_seconds() / 3600
    frame["closure_hours"] = closure_hours.where(closure_hours >= 0)
    frame["sector"] = frame["complaint_type"].map(map_sector_from_complaint)
    frame["source_domain"] = DOMAIN_NYC
    frame["localized_complaint_type"] = frame["complaint_type"]
    frame["text_input"] = build_text_input(frame)
    frame["event_name"] = "normal"
    return frame[CANONICAL_COLUMNS + ["source_domain", "localized_complaint_type", "text_input", "event_name"]]


def load_nyc_canonical_sample(
    sample_size: int = 180_000,
    random_state: int = 42,
    max_scan_rows: int = 1_250_000,
) -> tuple[pd.DataFrame, dict]:
    """Load a representative normalized NYC sample suitable for local training."""
    ensure_project_dirs()
    cache_file = CACHE_DIR / f"nyc_canonical_{sample_size}_{CACHE_VERSION}.pkl"
    meta_file = CACHE_DIR / f"nyc_canonical_{sample_size}_{CACHE_VERSION}.json"
    if cache_file.exists() and meta_file.exists():
        try:
            return pd.read_pickle(cache_file), json.loads(meta_file.read_text(encoding="utf-8"))
        except Exception:
            cache_file.unlink(missing_ok=True)
            meta_file.unlink(missing_ok=True)

    source_file = resolve_nyc_dataset_file()
    rng = np.random.default_rng(random_state)
    sampled = pd.DataFrame()
    rows_scanned = 0
    chunks_processed = 0

    reader = pd.read_csv(
        source_file,
        usecols=lambda column: column in NYC_SOURCE_COLUMNS,
        chunksize=200_000,
        low_memory=False,
    )

    for chunk in reader:
        chunks_processed += 1
        normalized = normalize_nyc_chunk(chunk)
        if normalized.empty:
            continue

        rows_scanned += len(normalized)
        normalized["_sample_key"] = rng.random(len(normalized))
        sampled = pd.concat([sampled, normalized], ignore_index=True)
        if len(sampled) > sample_size * 3:
            sampled = sampled.nsmallest(sample_size, "_sample_key").copy()
        if rows_scanned >= max_scan_rows and len(sampled) >= sample_size:
            break

    sampled = (
        sampled.nsmallest(sample_size, "_sample_key")
        .drop(columns="_sample_key")
        .sort_values("created_at")
        .reset_index(drop=True)
    )

    metadata = {
        "source_file": str(source_file),
        "rows_scanned": int(rows_scanned),
        "sample_rows": int(len(sampled)),
        "chunks_processed": int(chunks_processed),
        "cache_version": CACHE_VERSION,
    }
    sampled.to_pickle(cache_file)
    meta_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return sampled, metadata


def save_processed_dataframe(frame: pd.DataFrame, filename: str) -> Path:
    """Save a processed dataframe in compressed CSV form for runtime use."""
    ensure_project_dirs()
    path = PROCESSED_DIR / filename
    frame.to_csv(path, index=False, compression="gzip")
    return path


def load_processed_dataframe(filename: str, parse_dates: list[str] | None = None) -> pd.DataFrame:
    """Load a processed dataframe from the processed data directory."""
    path = PROCESSED_DIR / filename
    return pd.read_csv(path, parse_dates=parse_dates)
