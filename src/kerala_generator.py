from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd

from src.domain import (
    CANONICAL_COLUMNS,
    DOMAIN_KERALA,
    KERALA_DISTRICT_CENTERS,
    KERALA_EVENT_WINDOWS,
    KERALA_REQUEST_LIBRARY,
    LANDMARKS,
)


def district_weights() -> dict[str, float]:
    """Return relative complaint volumes by district."""
    return {
        "Thiruvananthapuram": 1.15,
        "Ernakulam": 1.20,
        "Thrissur": 1.00,
        "Kozhikode": 1.05,
        "Kannur": 0.90,
        "Kollam": 0.92,
        "Palakkad": 0.95,
        "Malappuram": 1.02,
        "Kottayam": 0.80,
        "Alappuzha": 0.82,
        "Idukki": 0.62,
        "Wayanad": 0.58,
        "Pathanamthitta": 0.60,
        "Kasaragod": 0.66,
    }


def resolve_event_name(date_value: pd.Timestamp) -> str:
    """Return the named synthetic event active for a date, if any."""
    for event_name, (start, end) in KERALA_EVENT_WINDOWS.items():
        if pd.Timestamp(start) <= date_value <= pd.Timestamp(end):
            return event_name
    return "normal"


def request_weight_for_date(template_sector: str, date_value: pd.Timestamp) -> float:
    """Adjust template weights based on seasonality and anomaly windows."""
    month = date_value.month
    weight = 1.0
    if month in {6, 7, 8, 9}:
        if template_sector in {"roads", "drainage_flooding"}:
            weight *= 1.45
    if month in {3, 4, 5} and template_sector == "water_supply":
        weight *= 1.55
    if month in {11, 12} and template_sector == "traffic_signals":
        weight *= 1.20

    event_name = resolve_event_name(date_value)
    if event_name == "flood_like_spike" and template_sector == "drainage_flooding":
        weight *= 3.8
    elif event_name == "landslide_road_washout" and template_sector == "roads":
        weight *= 3.2
    elif event_name == "water_shortage_spike" and template_sector == "water_supply":
        weight *= 3.0
    return weight


def closure_profile(sector: str, rng: np.random.Generator) -> tuple[pd.Timedelta | pd.NaT, str]:
    """Generate a realistic closure delay and status for a sector."""
    base_hours = {
        "roads": 84,
        "drainage_flooding": 60,
        "water_supply": 36,
        "waste_sanitation": 30,
        "street_lighting": 48,
        "traffic_signals": 42,
        "public_safety_other": 54,
    }
    if rng.random() < 0.14:
        return pd.NaT, "Open"

    hours = max(2.0, rng.gamma(shape=2.8, scale=base_hours[sector] / 2.8))
    return pd.to_timedelta(hours, unit="h"), "Closed"


def generate_kerala_dataset(num_rows: int = 50_000, random_state: int = 42) -> pd.DataFrame:
    """Generate a schema-aligned Kerala municipal transfer dataset."""
    rng = np.random.default_rng(random_state)
    districts = list(KERALA_DISTRICT_CENTERS)
    district_probabilities = np.array(list(district_weights().values()), dtype=float)
    district_probabilities = district_probabilities / district_probabilities.sum()

    date_range = pd.date_range("2020-01-01", "2024-12-31 23:00:00", freq="h")
    sampled_dates = pd.Series(rng.choice(date_range, size=num_rows, replace=True)).sort_values().reset_index(drop=True)

    records: list[dict[str, object]] = []
    for index, created_at in enumerate(sampled_dates, start=1):
        district = str(rng.choice(districts, p=district_probabilities))
        weighted_templates = np.array(
            [template.base_weight * request_weight_for_date(template.sector, created_at) for template in KERALA_REQUEST_LIBRARY],
            dtype=float,
        )
        weighted_templates = weighted_templates / weighted_templates.sum()
        template = KERALA_REQUEST_LIBRARY[int(rng.choice(len(KERALA_REQUEST_LIBRARY), p=weighted_templates))]

        location_type = str(rng.choice(template.location_types))
        landmark = str(rng.choice(LANDMARKS))
        descriptor = str(rng.choice(template.descriptor_templates)).format(landmark=landmark, district=district)
        center_lat, center_lon = KERALA_DISTRICT_CENTERS[district]
        latitude = center_lat + float(rng.normal(0, 0.015))
        longitude = center_lon + float(rng.normal(0, 0.018))
        closure_delta, status = closure_profile(template.sector, rng)
        closed_at = created_at + closure_delta if not pd.isna(closure_delta) else pd.NaT
        closure_hours = float(closure_delta.total_seconds() / 3600) if not pd.isna(closure_delta) else np.nan
        channel = str(rng.choice(["Call Centre", "Mobile App", "Ward Portal", "WhatsApp Desk"], p=[0.34, 0.26, 0.20, 0.20]))

        records.append(
            {
                "request_id": f"KER-{index:06d}",
                "created_at": created_at,
                "closed_at": closed_at,
                "complaint_type": template.localized_label,
                "descriptor": descriptor,
                "location_type": location_type,
                "city_or_district": district,
                "region": district,
                "latitude": latitude,
                "longitude": longitude,
                "status": status,
                "channel": channel,
                "closure_hours": closure_hours,
                "sector": template.sector,
                "source_domain": DOMAIN_KERALA,
                "localized_complaint_type": template.localized_label,
                "expected_nyc_label": template.expected_nyc_label,
                "event_name": resolve_event_name(created_at),
            }
        )

    frame = pd.DataFrame.from_records(records)
    return frame[CANONICAL_COLUMNS + ["source_domain", "localized_complaint_type", "expected_nyc_label", "event_name"]]
