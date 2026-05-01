from __future__ import annotations

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
CACHE_DIR = DATA_DIR / "cache"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = DATA_DIR / "artifacts"
REPORTS_DIR = ROOT_DIR / "reports"


def ensure_project_dirs() -> None:
    """Create runtime directories used by the training and app pipelines."""
    for path in [CACHE_DIR, PROCESSED_DIR, ARTIFACTS_DIR, REPORTS_DIR]:
        path.mkdir(parents=True, exist_ok=True)
