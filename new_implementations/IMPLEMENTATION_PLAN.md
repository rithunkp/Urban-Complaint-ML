# Urban Complaint ML — Demo Enhancement Plan

## Current State

| Layer | Status |
|---|---|
| NYC backbone (TF-IDF + LogReg) | ✅ 0.989 acc, 0.992 F1 |
| Kerala transfer evaluation | ✅ 50k rows, 70.3% coverage, 0.355 transfer acc |
| 4-factor allocation (50/20/20/10) | ✅ live in app |
| Scenario simulator | ✅ exists per README |
| 4 Gradio tabs: Overview, Resource Splitup, Prediction Demo, Data Explorer | ✅ |

---

## Phase 1 — Now (this sprint) ✅ implemented

These are fully self-contained additions. Zero refactoring of existing code.

### 1.1 `src/benchmark.py` — Allocation benchmark utility

Three-way comparison of allocation strategies:
- **Equal Split** — uniform 1/n per sector, floor applied. What a municipal board does without data.
- **Count-Only** — complaint share only, no trend/anomaly/delay. What a naive analyst does.
- **Model (4-Factor)** — the full existing formula.

Produces a `pd.DataFrame` ready to pass to the benchmark chart.

**Integration:** new "Benchmark" tab in `app.py` using `gr.Plot`. One `gr.Button` to compute, one chart output.

### 1.2 `src/plots.py` — Diagnostic plot functions

Four new Plotly figures, all returning `go.Figure` for drop-in Gradio use:

| Function | What it shows |
|---|---|
| `allocation_benchmark_chart` | Grouped bar: 3 strategies × 7 sectors |
| `transfer_confusion_matrix` | Heatmap: expected vs predicted NYC label, row-normalized |
| `confidence_distribution` | Overlapping histogram: confidence on correct vs incorrect |
| `per_district_accuracy` | Horizontal bar: transfer accuracy per district |
| `sector_load_vs_recommended` | Scatter: load share vs recommended, diagonal = equal weighting |

**Integration:** benchmark chart goes in the new Benchmark tab. The remaining 4 go in a new "Diagnostics" tab or distributed into the existing Overview and Resource Splitup tabs.

### 1.3 `src/feature_engineering.py` — Engineered features

10 intake-safe features across 4 groups:

| Group | Features |
|---|---|
| Text | `text_length`, `word_count`, `avg_word_length`, `has_urgency_keyword`, `has_location_reference`, `exclamation_flag`, `negative_word_count`, `descriptor_is_short` |
| Temporal | `hour_of_day`, `day_of_week`, `is_weekend`, `month`, `is_monsoon`, `is_festival_season`, `is_morning_rush` |
| Geographic | `is_coastal_district`, `district_urbanization_rank`, `has_coordinates` |
| Operational | `channel_is_digital` |
| Transfer/Kerala | `event_window_active`, `coastal_monsoon_interaction`, `confidence_bin` |

**Integration:** call `add_all_features(df)` after loading in `data_pipeline.py`. Run an ablation to measure delta. None of these cause leakage for the intake-time classifier.

---

## Phase 2 — Next sprint

These require a retrain or modest refactoring.

### 2.1 Plug feature engineering into training pipeline

- Add `feature_engineering.add_all_features()` to the preprocessing step in `scripts/train_nyc_model.py`
- Add numeric features alongside TF-IDF using `ColumnTransformer`
- Retrain, compare metrics vs baseline

### 2.2 Feature ablation table

- Train 5 variants: baseline + each feature group added
- Record accuracy delta and macro F1 delta per variant
- Output as `reports/ablation_results.json` and a table in the Diagnostics tab

### 2.3 Confidence calibration plot (reliability diagram)

- Group predictions into 10 confidence bins
- Plot mean confidence vs fraction correct
- A well-calibrated model's line follows the diagonal
- Useful for the viva: shows LogReg is actually well-calibrated

### 2.4 Event-window drilldown

- Filter Kerala evaluated set by `event_name`
- Show how accuracy and sector distribution shift during flood/shortage events
- Directly demonstrates why the anomaly pressure weight matters

---

## Phase 3 — Polish / Demo hardening

### 3.1 Narrative scenario panel

Extend the existing scenario simulator with a text explanation that auto-generates when weights change: *"Increasing anomaly pressure to 35% primarily shifts allocation toward drainage_flooding and roads. Under current conditions, Ernakulam and Alappuzha are the highest-pressure coastal districts."*

### 3.2 Export buttons

Add `gr.DownloadButton` for each plot (PNG) and the benchmark table (CSV). Useful for slides.

### 3.3 README and submission materials

- Record a 60-second screen capture of the benchmark + weight slider interaction
- Add performance badge table to README
- Add architecture diagram

---

## What NOT to add right now

- Live municipal data ingestion — out of scope for demo
- Production deployment config — separate concern
- Additional classifiers (XGBoost, BERT) — too disruptive to the existing pipeline before demo
- A new training run — only after Phase 2 ablation confirms the features help

---

## File map after Phase 1

```
src/
├── analytics.py          (existing)
├── benchmark.py          ← NEW
├── data_pipeline.py      (existing)
├── domain.py             (existing)
├── feature_engineering.py ← NEW
├── kerala_generator.py   (existing)
├── modeling.py           (existing)
├── paths.py              (existing)
├── plots.py              ← NEW
├── resource_allocation.py (existing)
└── ui_components.py      (existing, add new tab here or in app.py)
```
