from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.kerala_generator import generate_kerala_dataset
from src.modeling import evaluate_transfer, load_model_bundle, save_metrics_report
from src.data_pipeline import save_processed_dataframe
from src.paths import ensure_project_dirs


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for Kerala synthetic generation and evaluation."""
    parser = argparse.ArgumentParser(description="Generate and evaluate the Kerala transfer dataset.")
    parser.add_argument("--rows", type=int, default=50_000, help="Number of synthetic Kerala complaints to generate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible generation.")
    return parser.parse_args()


def main() -> None:
    """Generate the Kerala dataset, evaluate the NYC model, and save both outputs."""
    args = parse_args()
    ensure_project_dirs()

    kerala_frame = generate_kerala_dataset(num_rows=args.rows, random_state=args.seed)
    save_processed_dataframe(kerala_frame, "kerala_transfer.csv.gz")

    bundle = load_model_bundle()
    evaluated, metrics = evaluate_transfer(bundle, kerala_frame)
    save_processed_dataframe(evaluated, "kerala_transfer_evaluated.csv.gz")
    save_metrics_report("kerala_transfer_metrics.json", metrics)

    print("Kerala transfer set generation complete.")
    print(metrics)


if __name__ == "__main__":
    main()
