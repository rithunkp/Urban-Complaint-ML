"""
src/benchmark.py

Allocation benchmark: compare three resource-split strategies side-by-side.

    equal_split   — uniform 1/n per sector, 5% floor applied
                    represents what a municipal board does without data
    count_split   — complaint-share only, no trend / anomaly / delay weighting
                    represents what a naive analyst does with a frequency table
    model_split   — the full 4-factor formula (pass in from resource_allocation)

Usage:
    from src.benchmark import compute_benchmark
    bdf = compute_benchmark(df, model_scores)  # → pd.DataFrame
"""

from __future__ import annotations

import pandas as pd

SECTORS: list[str] = [
    "roads",
    "drainage_flooding",
    "water_supply",
    "waste_sanitation",
    "street_lighting",
    "traffic_signals",
    "public_safety_other",
]
FLOOR: float = 0.05


# ── helpers ──────────────────────────────────────────────────────────────────

def _floor_and_normalise(raw: dict[str, float], floor: float = FLOOR) -> dict[str, float]:
    floored = {k: max(v, floor) for k, v in raw.items()}
    total = sum(floored.values())
    return {k: v / total for k, v in floored.items()}


# ── strategies ───────────────────────────────────────────────────────────────

def equal_split(sectors: list[str] = SECTORS) -> dict[str, float]:
    """Uniform allocation — no data used."""
    n = len(sectors)
    return _floor_and_normalise({s: 1.0 / n for s in sectors})


def count_split(df: pd.DataFrame, sector_col: str = "sector") -> dict[str, float]:
    """
    Allocation based purely on complaint-count share.
    No trend, anomaly, or closure-delay signal.
    """
    counts = df[sector_col].value_counts()
    raw = {s: float(counts.get(s, 0)) for s in SECTORS}
    total = sum(raw.values()) or 1.0
    normalised = {k: v / total for k, v in raw.items()}
    return _floor_and_normalise(normalised)


# ── composite ────────────────────────────────────────────────────────────────

def compute_benchmark(
    df: pd.DataFrame,
    model_scores: dict[str, float],
    sector_col: str = "sector",
) -> pd.DataFrame:
    """
    Build a long-style DataFrame comparing three allocation strategies.

    Parameters
    ----------
    df
        Kerala evaluated dataset (kerala_transfer_evaluated.csv.gz).
    model_scores
        Dict {sector: fraction} from resource_allocation.py — the live
        4-factor formula output.
    sector_col
        Column name for the sector field.

    Returns
    -------
    pd.DataFrame with columns:
        sector | Equal Split | Count-Only | Model (4-Factor)
    Values are percentages (0–100), two decimal places.
    """
    eq = equal_split()
    co = count_split(df, sector_col=sector_col)

    rows = []
    for s in SECTORS:
        rows.append({
            "sector": s,
            "Equal Split": round(eq.get(s, 0) * 100, 2),
            "Count-Only": round(co.get(s, 0) * 100, 2),
            "Model (4-Factor)": round(model_scores.get(s, 0) * 100, 2),
        })
    return pd.DataFrame(rows)


def benchmark_delta(benchmark_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add two delta columns showing how much the model deviates from each baseline.
    Useful for the table view in the Benchmark tab.
    """
    df = benchmark_df.copy()
    df["vs Equal (pp)"] = (df["Model (4-Factor)"] - df["Equal Split"]).round(2)
    df["vs Count-Only (pp)"] = (df["Model (4-Factor)"] - df["Count-Only"]).round(2)
    return df
