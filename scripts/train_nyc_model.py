from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_pipeline import load_nyc_canonical_sample, save_processed_dataframe
from src.modeling import build_training_frame, save_metrics_report, save_model_bundle, train_classifier
from src.paths import ARTIFACTS_DIR, ensure_project_dirs


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for local NYC training."""
    parser = argparse.ArgumentParser(description="Train the NYC complaint classifier and prepare runtime assets.")
    parser.add_argument("--sample-size", type=int, default=180_000, help="Target NYC sample size for training and runtime analytics.")
    parser.add_argument("--min-examples", type=int, default=250, help="Minimum class frequency to keep a complaint type.")
    parser.add_argument("--max-classes", type=int, default=18, help="Maximum number of complaint classes to train.")
    parser.add_argument("--max-rows", type=int, default=120_000, help="Maximum number of rows used for fitting.")
    return parser.parse_args()


def main() -> None:
    """Train on NYC, save the model bundle, and persist runtime data."""
    args = parse_args()
    ensure_project_dirs()

    nyc_frame, metadata = load_nyc_canonical_sample(sample_size=args.sample_size)
    training = build_training_frame(
        nyc_frame,
        min_examples=args.min_examples,
        max_classes=args.max_classes,
        max_rows=args.max_rows,
    )
    bundle = train_classifier(training)
    save_model_bundle(bundle, metadata=metadata)
    save_processed_dataframe(nyc_frame, "nyc_runtime.csv.gz")
    save_metrics_report("nyc_metrics.json", bundle["metrics"])

    manifest = {
        "training_metadata": metadata,
        "training_rows": int(len(training)),
        "processed_runtime_file": "data/processed/nyc_runtime.csv.gz",
        "model_bundle": str(ARTIFACTS_DIR / "complaint_model_bundle.joblib"),
    }
    (ARTIFACTS_DIR / "runtime_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("NYC training complete.")
    print(json.dumps(bundle["metrics"], indent=2))


if __name__ == "__main__":
    main()
