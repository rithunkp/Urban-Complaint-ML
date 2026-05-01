---
title: UrbanComplaintML
emoji: 🏙️
colorFrom: blue
colorTo: green
sdk: gradio
app_file: app.py
pinned: false
---

# Kerala Urban Complaint Analysis for Smart City Resource Optimization

This project delivers a Kerala-first civic operations product with a Hugging Face interface that:

- trains its complaint intelligence layer on the NYC 311 dataset,
- uses a Kerala municipal operations dataset with roads as the dominant category,
- evaluates the trained model on the Kerala dataset,
- recommends a fair municipal resource splitup so no sector is neglected,
- explains the whole flow in a simple analytics UI.

## What lives in the repo

```text
CityML/
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
|   |   `-- 311-service-requests-from-2010-to-present.csv
|   |-- cache/
|   |-- processed/
|   `-- artifacts/
`-- reports/
```

## Core idea

### Model backbone
- Real data: NYC 311 service requests loaded from `data/raw` or via `kagglehub`.
- The model is trained only on intake-time fields aligned with real inference usage.
- NYC is the reference training source, not the public-facing product identity.

### Kerala operations domain
- Kerala data is schema-aligned to the NYC training pipeline.
- It covers all 14 Kerala districts.
- Roads are the dominant complaint family, but other municipal sectors are also represented.
- It includes seasonal and named anomaly periods:
  - flood-like spike,
  - landslide / road washout spike,
  - water shortage spike.

### Resource splitup
Municipal resource allocation is rule-based and explainable:

- 50% recent complaint share
- 20% positive trend growth
- 20% anomaly pressure
- 10% closure-delay pressure

Every sector gets a minimum 5% floor before the final distribution is renormalized to 100%.

## Local setup

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

## Training and asset generation

The Gradio app expects saved artifacts. Build them locally first.

### Train the NYC model and save runtime assets

```bash
python scripts/train_nyc_model.py
```

This produces:

- `data/processed/nyc_runtime.csv.gz`
- `data/artifacts/complaint_model_bundle.joblib`
- `reports/nyc_metrics.json`

### Generate and evaluate the Kerala transfer dataset

```bash
python scripts/generate_kerala_transfer_set.py
```

This produces:

- `data/processed/kerala_transfer.csv.gz`
- `data/processed/kerala_transfer_evaluated.csv.gz`
- `reports/kerala_transfer_metrics.json`

## Launch the app

```bash
python app.py
```

The app will refuse to fully boot if the processed runtime assets are missing, and it will tell you which script to run.

### Recommended deployment flow

1. Create a new Hugging Face Space.
2. Choose **Gradio** as the SDK.
3. Push this repository after generating the saved assets locally.
4. Make sure the repo contains:
   - `app.py`
   - `requirements.txt`
   - `data/processed/*.csv.gz`
   - `data/artifacts/*.joblib`
   - `reports/*.json`

### Important runtime rule

The Space should not retrain the model. Training is local/offline. The Space only loads saved artifacts.

## Kaggle dataset loading

If the raw CSV is not already present in `data/raw`, the training script can use:

```python
import kagglehub
path = kagglehub.dataset_download("new-york-city/ny-311-service-requests")
```

## UI overview

The Gradio app contains these tabs:

- `Overview`
  - KPI cards
  - model backbone and Kerala readiness summary
  - complaint distribution
  - top complaint types
  - timeline with anomaly markers
  - geospatial view when coordinates are available

- `Resource Splitup`
  - current sector load
  - recommended allocation pie chart
  - district or borough vs sector matrix
  - fairness explanation

- `ML Demo`
  - free-text complaint prediction
  - predicted complaint type
  - predicted sector
  - top confidence chart

- `Data Explorer`
  - anomaly table
  - filtered rows
  - downloadable CSV
