from __future__ import annotations

from dataclasses import dataclass

DOMAIN_NYC = "NYC"
DOMAIN_KERALA = "Kerala"

CANONICAL_COLUMNS = [
    "request_id",
    "created_at",
    "closed_at",
    "complaint_type",
    "descriptor",
    "location_type",
    "city_or_district",
    "region",
    "latitude",
    "longitude",
    "status",
    "channel",
    "closure_hours",
    "sector",
]

NYC_SOURCE_COLUMNS = [
    "Unique Key",
    "Created Date",
    "Closed Date",
    "Complaint Type",
    "Descriptor",
    "Location Type",
    "City",
    "Borough",
    "Latitude",
    "Longitude",
    "Status",
    "Open Data Channel Type",
]

SECTOR_ORDER = [
    "roads",
    "drainage_flooding",
    "water_supply",
    "waste_sanitation",
    "street_lighting",
    "traffic_signals",
    "public_safety_other",
]

SECTOR_DISPLAY_NAMES = {
    "roads": "Roads",
    "drainage_flooding": "Drainage & Flooding",
    "water_supply": "Water Supply",
    "waste_sanitation": "Waste & Sanitation",
    "street_lighting": "Street Lighting",
    "traffic_signals": "Traffic & Parking",
    "public_safety_other": "Public Safety & Other",
}

BOROUGH_MAP = {
    "BROOKLYN": "Brooklyn",
    "BRONX": "Bronx",
    "MANHATTAN": "Manhattan",
    "QUEENS": "Queens",
    "STATEN ISLAND": "Staten Island",
    "UNSPECIFIED": "Unspecified",
    "UNKNOWN": "Unknown",
}

SECTOR_KEYWORDS = {
    "roads": ["street", "road", "sidewalk", "pothole", "driveway", "curb"],
    "drainage_flooding": ["sewer", "drain", "flood", "water leak", "catch basin"],
    "water_supply": ["water system", "hydrant", "no water", "hot water", "heat"],
    "waste_sanitation": ["sanitation", "garbage", "collection", "dirty", "rodent", "waste"],
    "street_lighting": ["light", "lamp"],
    "traffic_signals": ["traffic", "signal", "parking", "vehicle", "bus lane"],
}

KERALA_DISTRICT_CENTERS = {
    "Thiruvananthapuram": (8.5241, 76.9366),
    "Kollam": (8.8932, 76.6141),
    "Pathanamthitta": (9.2648, 76.7870),
    "Alappuzha": (9.4981, 76.3388),
    "Kottayam": (9.5916, 76.5222),
    "Idukki": (9.8497, 76.9710),
    "Ernakulam": (9.9816, 76.2999),
    "Thrissur": (10.5276, 76.2144),
    "Palakkad": (10.7867, 76.6548),
    "Malappuram": (11.0730, 76.0740),
    "Kozhikode": (11.2588, 75.7804),
    "Wayanad": (11.6854, 76.1320),
    "Kannur": (11.8745, 75.3704),
    "Kasaragod": (12.4996, 74.9869),
}

KERALA_EVENT_WINDOWS = {
    "flood_like_spike": ("2021-08-01", "2021-08-20"),
    "landslide_road_washout": ("2022-07-10", "2022-07-28"),
    "water_shortage_spike": ("2024-04-01", "2024-05-10"),
}


@dataclass(frozen=True)
class KeralaRequestTemplate:
    """Blueprint for a synthetic Kerala complaint type."""

    localized_label: str
    expected_nyc_label: str
    sector: str
    base_weight: float
    location_types: tuple[str, ...]
    descriptor_templates: tuple[str, ...]


KERALA_REQUEST_LIBRARY = (
    KeralaRequestTemplate(
        localized_label="Road Surface Damage",
        expected_nyc_label="Street Condition",
        sector="roads",
        base_weight=20.0,
        location_types=("Main Road", "Junction", "Bus Route"),
        descriptor_templates=(
            "Large potholes reported on {landmark} near {district}.",
            "Road surface broken after heavy rain at {landmark}, {district}.",
            "Damaged carriageway causing traffic slowdown near {landmark} in {district}.",
        ),
    ),
    KeralaRequestTemplate(
        localized_label="Footpath Damage",
        expected_nyc_label="Sidewalk Condition",
        sector="roads",
        base_weight=10.0,
        location_types=("Footpath", "School Zone", "Market Area"),
        descriptor_templates=(
            "Broken footpath slabs near {landmark} in {district}.",
            "Unsafe walkway reported beside {landmark}, {district}.",
        ),
    ),
    KeralaRequestTemplate(
        localized_label="Illegal Parking",
        expected_nyc_label="Illegal Parking",
        sector="traffic_signals",
        base_weight=8.0,
        location_types=("Hospital Entrance", "Junction", "Market Road"),
        descriptor_templates=(
            "Vehicles parked illegally blocking access at {landmark}, {district}.",
            "Improper roadside parking causing congestion near {landmark} in {district}.",
        ),
    ),
    KeralaRequestTemplate(
        localized_label="Drain Overflow",
        expected_nyc_label="Sewer",
        sector="drainage_flooding",
        base_weight=9.0,
        location_types=("Drainage Canal", "Residential Lane", "Market Area"),
        descriptor_templates=(
            "Drain overflow reported near {landmark} in {district}.",
            "Stormwater drain blocked and overflowing at {landmark}, {district}.",
        ),
    ),
    KeralaRequestTemplate(
        localized_label="Water Supply Interruption",
        expected_nyc_label="Water System",
        sector="water_supply",
        base_weight=9.0,
        location_types=("Residential Area", "Municipal Ward", "Apartment Block"),
        descriptor_templates=(
            "No water supply for two days near {landmark}, {district}.",
            "Low pressure water complaint from {landmark} area in {district}.",
        ),
    ),
    KeralaRequestTemplate(
        localized_label="Waste Collection Delay",
        expected_nyc_label="Missed Collection",
        sector="waste_sanitation",
        base_weight=8.0,
        location_types=("Residential Colony", "Market Area", "Ward Office"),
        descriptor_templates=(
            "Waste collection missed near {landmark} in {district}.",
            "Garbage pickup delayed for several days at {landmark}, {district}.",
        ),
    ),
    KeralaRequestTemplate(
        localized_label="Unsanitary Drain or Dump",
        expected_nyc_label="Unsanitary Condition",
        sector="waste_sanitation",
        base_weight=7.0,
        location_types=("Open Drain", "Market Area", "Roadside"),
        descriptor_templates=(
            "Unsanitary open drain reported near {landmark} in {district}.",
            "Garbage dumping causing foul smell at {landmark}, {district}.",
        ),
    ),
    KeralaRequestTemplate(
        localized_label="Streetlight Failure",
        expected_nyc_label="Street Light Condition",
        sector="street_lighting",
        base_weight=7.0,
        location_types=("Residential Street", "Bus Stop", "Park Road"),
        descriptor_templates=(
            "Streetlights not working near {landmark} in {district}.",
            "Dark stretch reported because lamp posts failed at {landmark}, {district}.",
        ),
    ),
    KeralaRequestTemplate(
        localized_label="Traffic Signal Failure",
        expected_nyc_label="Traffic Signal Condition",
        sector="traffic_signals",
        base_weight=6.0,
        location_types=("Major Junction", "Signalized Intersection", "Town Center"),
        descriptor_templates=(
            "Traffic signal malfunction reported at {landmark}, {district}.",
            "Signal lights out of service near {landmark} in {district}.",
        ),
    ),
    KeralaRequestTemplate(
        localized_label="Blocked Access Road",
        expected_nyc_label="Blocked Driveway",
        sector="roads",
        base_weight=5.0,
        location_types=("Lane Entrance", "School Gate", "Hospital Access"),
        descriptor_templates=(
            "Access road blocked by temporary obstruction at {landmark}, {district}.",
            "Residents cannot enter property because approach road is blocked near {landmark}.",
        ),
    ),
    KeralaRequestTemplate(
        localized_label="Residential Noise Disturbance",
        expected_nyc_label="Noise - Residential",
        sector="public_safety_other",
        base_weight=5.0,
        location_types=("Residential Area", "Festival Ground", "Community Hall"),
        descriptor_templates=(
            "Excessive loudspeaker noise reported near {landmark}, {district}.",
            "Late-night residential noise complaint from {landmark} area in {district}.",
        ),
    ),
    KeralaRequestTemplate(
        localized_label="Damaged Public Vehicle",
        expected_nyc_label="Derelict Vehicles",
        sector="public_safety_other",
        base_weight=4.0,
        location_types=("Roadside", "Bus Depot", "Market Street"),
        descriptor_templates=(
            "Abandoned damaged vehicle reported near {landmark}, {district}.",
            "Derelict vehicle blocking public space at {landmark}, {district}.",
        ),
    ),
)

NYC_TO_LOCALIZED_LABEL = {
    template.expected_nyc_label: template.localized_label for template in KERALA_REQUEST_LIBRARY
}

LANDMARKS = (
    "Town Bus Stand",
    "Collectorate Road",
    "Municipal Office",
    "Market Junction",
    "High School Road",
    "Temple Junction",
    "Hospital Road",
    "Bypass Signal",
    "Canal Side Road",
    "Railway Station Approach",
)


def normalize_borough(value: str) -> str:
    """Normalize borough naming into a clean display string."""
    text = str(value).strip()
    return BOROUGH_MAP.get(text.upper(), text.title() if text else "Unknown")


def map_sector_from_complaint(complaint_type: str) -> str:
    """Map a complaint type to a municipal sector using stable keyword rules."""
    normalized = complaint_type.lower().strip()
    for sector, keywords in SECTOR_KEYWORDS.items():
        if any(keyword in normalized for keyword in keywords):
            return sector
    return "public_safety_other"


def localized_label_for_prediction(predicted_nyc_label: str) -> str:
    """Map a predicted NYC label into a user-facing Kerala label when available."""
    return NYC_TO_LOCALIZED_LABEL.get(predicted_nyc_label, predicted_nyc_label)
