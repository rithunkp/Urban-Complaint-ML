from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.domain import localized_label_for_prediction, map_sector_from_complaint
from src.paths import ARTIFACTS_DIR, REPORTS_DIR, ensure_project_dirs

MODEL_BUNDLE_NAME = "complaint_model_bundle.joblib"


def build_training_frame(
    frame: pd.DataFrame,
    min_examples: int = 250,
    max_classes: int = 18,
    max_rows: int = 120_000,
) -> pd.DataFrame:
    """Select stable classes and inference-aligned text for supervised training."""
    training = frame[["complaint_type", "descriptor", "location_type", "text_input"]].copy()
    training["text_input"] = training["text_input"].fillna("").astype(str).str.strip()
    training = training[training["text_input"].str.len() >= 10].copy()

    class_counts = training["complaint_type"].value_counts()
    keep_labels = class_counts[class_counts >= min_examples].head(max_classes).index
    training = training[training["complaint_type"].isin(keep_labels)].copy()
    if len(training) > max_rows:
        training = training.sample(max_rows, random_state=42)
    return training.reset_index(drop=True)


def compute_top_k_accuracy(probabilities: np.ndarray, true_labels: np.ndarray, classes: np.ndarray, k: int = 3) -> float:
    """Compute top-k accuracy for labels covered by the trained class set."""
    top_k = np.argsort(probabilities, axis=1)[:, -k:]
    class_to_index = {label: index for index, label in enumerate(classes)}
    covered_indices = [class_to_index[label] for label in true_labels if label in class_to_index]
    covered_rows = [row for label, row in zip(true_labels, top_k, strict=False) if label in class_to_index]
    if not covered_indices:
        return 0.0
    true_index = np.array(covered_indices)
    top_k = np.array(covered_rows)
    hits = [(truth in row) for truth, row in zip(true_index, top_k, strict=False)]
    return float(np.mean(hits))


def train_classifier(training: pd.DataFrame) -> dict[str, Any]:
    """Train the production classifier and return a serializable bundle."""
    x_train, x_test, y_train, y_test = train_test_split(
        training["text_input"],
        training["complaint_type"],
        test_size=0.2,
        random_state=42,
        stratify=training["complaint_type"],
    )

    pipeline = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    stop_words="english",
                    max_features=7000,
                    ngram_range=(1, 2),
                    min_df=3,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=700,
                    class_weight="balanced",
                ),
            ),
        ]
    )
    pipeline.fit(x_train, y_train)

    predictions = pipeline.predict(x_test)
    probabilities = pipeline.predict_proba(x_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "macro_precision": float(precision_score(y_test, predictions, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_test, predictions, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
        "top_3_accuracy": compute_top_k_accuracy(probabilities, y_test.to_numpy(), pipeline.classes_, k=3),
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "class_count": int(training["complaint_type"].nunique()),
        "classes": pipeline.classes_.tolist(),
        "classification_report": classification_report(y_test, predictions, output_dict=True, zero_division=0),
    }
    return {"pipeline": pipeline, "metrics": metrics}


def save_model_bundle(bundle: dict[str, Any], metadata: dict[str, Any] | None = None) -> Path:
    """Persist the model bundle and optional metadata for runtime use."""
    ensure_project_dirs()
    payload = {
        "pipeline": bundle["pipeline"],
        "metrics": bundle["metrics"],
        "metadata": metadata or {},
    }
    model_path = ARTIFACTS_DIR / MODEL_BUNDLE_NAME
    joblib.dump(payload, model_path)
    return model_path


def load_model_bundle(path: Path | None = None) -> dict[str, Any]:
    """Load the trained model bundle."""
    model_path = path or ARTIFACTS_DIR / MODEL_BUNDLE_NAME
    return joblib.load(model_path)


def save_metrics_report(filename: str, metrics: dict[str, Any]) -> Path:
    """Write a JSON metrics report into the reports directory."""
    ensure_project_dirs()
    report_path = REPORTS_DIR / filename
    report_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return report_path


def predict_request(bundle: dict[str, Any], descriptor: str, location_type: str = "") -> dict[str, Any]:
    """Run a single inference and produce user-facing labels and probabilities."""
    text_input = " ".join(part.strip() for part in [descriptor, location_type] if part and part.strip()).strip()
    pipeline: Pipeline = bundle["pipeline"]
    probabilities = pipeline.predict_proba([text_input])[0]
    class_names = pipeline.classes_
    top_indices = np.argsort(probabilities)[::-1][:5]

    top_frame = pd.DataFrame(
        {
            "predicted_nyc_label": [class_names[index] for index in top_indices],
            "confidence": [float(probabilities[index]) for index in top_indices],
        }
    )
    top_frame["localized_label"] = top_frame["predicted_nyc_label"].map(localized_label_for_prediction)
    top_frame["sector"] = top_frame["predicted_nyc_label"].map(map_sector_from_complaint)

    winner = top_frame.iloc[0]
    return {
        "predicted_nyc_label": winner["predicted_nyc_label"],
        "localized_label": winner["localized_label"],
        "sector": winner["sector"],
        "confidence": float(winner["confidence"]),
        "top_predictions": top_frame.sort_values("confidence", ascending=True).reset_index(drop=True),
    }


def evaluate_transfer(bundle: dict[str, Any], kerala_frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Evaluate the NYC-trained classifier on the Kerala transfer set."""
    working = kerala_frame.copy()
    pipeline: Pipeline = bundle["pipeline"]
    working["text_input"] = (
        working[["descriptor", "location_type"]]
        .fillna("")
        .agg(" ".join, axis=1)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    probabilities = pipeline.predict_proba(working["text_input"])
    predictions = pipeline.classes_[np.argmax(probabilities, axis=1)]
    working["predicted_nyc_label"] = predictions
    working["predicted_localized_label"] = working["predicted_nyc_label"].map(localized_label_for_prediction)
    working["predicted_sector"] = working["predicted_nyc_label"].map(map_sector_from_complaint)
    working["prediction_confidence"] = probabilities.max(axis=1)

    metrics = {"rows": int(len(working))}
    if "expected_nyc_label" in working.columns:
        covered_mask = working["expected_nyc_label"].isin(pipeline.classes_)
        covered = working[covered_mask].copy()
        metrics["reference_label_coverage"] = float(covered_mask.mean())
        metrics["covered_reference_rows"] = int(covered_mask.sum())
        metrics["uncovered_reference_rows"] = int((~covered_mask).sum())

        working["reference_label_covered"] = covered_mask
        metrics.update(
            {
                "accuracy_against_reference": float(accuracy_score(covered["expected_nyc_label"], covered["predicted_nyc_label"]))
                if not covered.empty
                else 0.0,
                "macro_f1_against_reference": float(
                    f1_score(covered["expected_nyc_label"], covered["predicted_nyc_label"], average="macro", zero_division=0)
                )
                if not covered.empty
                else 0.0,
                "top_3_accuracy_against_reference": compute_top_k_accuracy(
                    probabilities[covered_mask.to_numpy()],
                    covered["expected_nyc_label"].to_numpy(),
                    pipeline.classes_,
                    k=3,
                )
                if not covered.empty
                else 0.0,
            }
        )
        working["correct_reference"] = working["expected_nyc_label"] == working["predicted_nyc_label"]
    return working, metrics
