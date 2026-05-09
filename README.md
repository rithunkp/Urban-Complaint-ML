---
title: UrbanComplaintML
colorFrom: blue
colorTo: green
sdk: gradio
app_file: app.py
pinned: false
---

# Urban Complaint ML

Kerala-focused complaint analytics and resource planning on top of an NYC 311-trained complaint classification pipeline.

The app is designed for municipal operations review. It combines:

- complaint trend analysis
- anomaly detection
- explainable sector-level resource split recommendations
- an interactive ML prediction demo
- a filterable complaint explorer

## Highlights

- Kerala-first dashboard language and district-level analysis
- NYC 311 used as the reference training backbone
- explainable, rule-based resource allocation
- Gradio UI for local demos and hosted app packaging

## Repository Layout

```text
Urban-Complaint-ML/
|-- app.py
|-- requirements.txt
|-- scripts/
|   |-- train_nyc_model.py
|   `-- generate_kerala_transfer_set.py
|-- src/
|   |-- analytics.py
|   |-- data_pipeline.py
|   |-- domain.py
|   |-- kerala_generator.py
|   |-- modeling.py
|   |-- paths.py
|   |-- resource_allocation.py
|   `-- ui_components.py
|-- data/
|   |-- raw/
|   |-- cache/
|   |-- processed/
|   `-- artifacts/
`-- reports/
```

## How It Works

### Training Backbone

- The supervised classifier is trained on NYC 311 service request data.
- The model uses intake-time fields only, so inference aligns with real complaint intake conditions.
- The classifier output is mapped into user-facing municipal sector views.

### Kerala Transfer Layer

- A Kerala-oriented evaluation dataset is generated to mirror the production schema.
- The Kerala view covers all 14 districts.
- Roads are intentionally the dominant complaint family, while water, drainage, waste, lighting, and traffic remain represented.

### Resource Split Logic

The resource recommendation engine is fully explainable and currently weights:

- 50% recent complaint share
- 20% positive trend growth
- 20% anomaly pressure
- 10% closure-delay pressure

Each sector receives a minimum 5% floor before final renormalization.

## Quick Start

### 1. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

macOS or Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Generate runtime assets

The app expects prebuilt runtime files before launch.

```bash
python scripts/train_nyc_model.py
python scripts/generate_kerala_transfer_set.py
```

This produces the core files used by the UI:

- `data/processed/nyc_runtime.csv.gz`
- `data/processed/kerala_transfer.csv.gz`
- `data/processed/kerala_transfer_evaluated.csv.gz`
- `data/artifacts/complaint_model_bundle.joblib`
- `reports/nyc_metrics.json`
- `reports/kerala_transfer_metrics.json`

### 4. Launch the app

```bash
python app.py
```

If assets are missing, the app will stop at startup and show the required file paths.

## App Sections

### Overview

- KPI cards
- complaint distribution
- top complaint categories
- timeline with anomaly markers
- geospatial overview when coordinates are available
- transfer-readiness summary

### Resource Splitup

- current sector load
- recommended allocation split
- district-by-sector pressure matrix
- explanation for the recommendation

### Diagnostics

- NYC vs Kerala evaluation metrics table
- Kerala transfer confusion matrix
- prediction confidence distribution
- district-level transfer accuracy

### Benchmark

- equal split vs count-only vs 4-factor model comparison
- sector-level benchmark chart
- percentage-point delta table

### Simulation

- simple sector-level complaint demand changes from -50% to +100%
- adjustable minimum guaranteed allocation per sector
- default-vs-simulated allocation comparison with plain-language explanation

### Prediction Demo

- free-text complaint description input
- predicted complaint type
- predicted sector
- top-confidence chart

### Data Explorer

- anomaly table
- filtered complaint rows
- downloadable CSV export

## Data Notes

- Raw NYC data may live in `data/raw/`.
- Cached intermediate files may live in `data/cache/`.
- Generated runtime assets belong in `data/processed/`, `data/artifacts/`, and `reports/`.
- Large binary and compressed artifacts are intentionally excluded from normal Git workflows by default.

## Deployment Guidance

### Local-first workflow

This repository is production-ready for local execution after runtime assets are generated.

### Hosted deployment

Hosted deployments need a deliberate artifact strategy because the app depends on generated `.csv.gz` and `.joblib` files at startup.

Recommended options:

- build the assets during image creation in a containerized deploy flow
- load the artifacts from managed object storage
- publish the artifacts with a large-file backend that your host fully supports

### Important rule

Do not retrain the model inside the request-serving app process. Training and artifact generation should remain offline or build-time steps.

## Data Source Helper

If the NYC source CSV is not already available locally, the training flow can use `kagglehub`:

```python
import kagglehub

path = kagglehub.dataset_download("new-york-city/ny-311-service-requests")
```

## Results

- NYC backbone performance reaches **0.989 accuracy**, **0.992 macro F1**, and **1.000 top-3 accuracy** on the retained class set.
- Kerala transfer evaluation covers **50,000** generated rows with **100.0% reference-label coverage**, **0.889 accuracy**, **0.709 macro F1**, and **0.961 top-3 accuracy** against the synthetic reference labels.
- The decision layer combines complaint load, growth, anomaly pressure, and closure delay into an explainable allocation formula.
- The app includes diagnostics, allocation benchmarks, and a scenario simulator for testing how sector complaint increases or decreases shift recommended allocation.

## Status

This repo is suitable for:

- local demos
- analytics review
- model behavior inspection
- packaging into a more controlled production deployment

Before a public hosted rollout, make sure artifact storage, startup behavior, and deployment infrastructure are finalized together.
