from __future__ import annotations

import pandas as pd

from src.domain import SECTOR_DISPLAY_NAMES, SECTOR_ORDER


def _empty_split() -> pd.Series:
    """Return a uniform sector split as fractions."""
    return pd.Series(1.0 / len(SECTOR_ORDER), index=SECTOR_ORDER, dtype=float)


def _apply_floor(split: pd.Series, floor: float = 0.05) -> pd.Series:
    """Apply a minimum floor and normalize sector fractions to sum to 1."""
    aligned = split.reindex(SECTOR_ORDER, fill_value=0.0).clip(lower=0.0)
    total = aligned.sum()
    if total <= 0:
        aligned = _empty_split()
    else:
        aligned = aligned / total

    adjusted = floor + (1.0 - floor * len(SECTOR_ORDER)) * aligned
    adjusted_total = adjusted.sum()
    if adjusted_total <= 0:
        return _empty_split()
    return adjusted / adjusted_total


def equal_split() -> pd.Series:
    """Uniform allocation baseline."""
    return _empty_split()


def count_only_split(frame: pd.DataFrame) -> pd.Series:
    """Allocation baseline using only raw complaint share by sector."""
    if frame.empty or "sector" not in frame.columns:
        return _empty_split()
    share = frame["sector"].value_counts(normalize=True).reindex(SECTOR_ORDER, fill_value=0.0)
    return _apply_floor(share)


def model_split_from_allocation(allocation: pd.DataFrame) -> pd.Series:
    """Extract the existing model recommendation from the allocation table."""
    if allocation.empty or not {"sector", "recommended_pct"}.issubset(allocation.columns):
        return _empty_split()
    model = allocation.set_index("sector")["recommended_pct"].reindex(SECTOR_ORDER, fill_value=0.0) / 100.0
    return _apply_floor(model)


def compute_benchmark(frame: pd.DataFrame, allocation: pd.DataFrame) -> pd.DataFrame:
    """Compare equal, count-only, and model-driven allocation strategies."""
    equal = equal_split()
    count = count_only_split(frame)
    model = model_split_from_allocation(allocation)

    benchmark = pd.DataFrame(
        {
            "sector": SECTOR_ORDER,
            "sector_display": [SECTOR_DISPLAY_NAMES.get(sector, sector) for sector in SECTOR_ORDER],
            "Equal Split": (equal * 100).round(2).to_numpy(),
            "Count-Only": (count * 100).round(2).to_numpy(),
            "Model (4-Factor)": (model * 100).round(2).to_numpy(),
        }
    )
    return benchmark


def benchmark_delta(benchmark: pd.DataFrame) -> pd.DataFrame:
    """Add model-vs-baseline percentage-point deltas to a benchmark table."""
    if benchmark.empty:
        return pd.DataFrame(
            columns=[
                "sector_display",
                "Equal Split",
                "Count-Only",
                "Model (4-Factor)",
                "vs Equal (pp)",
                "vs Count-Only (pp)",
            ]
        )

    output = benchmark.copy()
    output["vs Equal (pp)"] = (output["Model (4-Factor)"] - output["Equal Split"]).round(2)
    output["vs Count-Only (pp)"] = (output["Model (4-Factor)"] - output["Count-Only"]).round(2)
    return output[
        [
            "sector_display",
            "Equal Split",
            "Count-Only",
            "Model (4-Factor)",
            "vs Equal (pp)",
            "vs Count-Only (pp)",
        ]
    ]
