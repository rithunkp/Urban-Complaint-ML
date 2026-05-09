# Implement Benchmarking and Diagnostic Plots Now

## Summary
Add the demo-ready benchmarking and metrics plot features from `new_implementations/`, but skip simulation for this session. The implementation should be small, local, and avoid retraining or dependency changes.

## Key Changes
- Add `src/benchmark.py` from the useful parts of `new_implementations/benchmark.py`.
- Add diagnostic Plotly helpers, preferably into `src/ui_components.py` to keep UI plotting centralized.
- Wire new outputs into `app.py` as two new tabs:
  - **Diagnostics**
  - **Benchmark**
- Do not integrate `feature_engineering.py`.
- Do not add the resource split simulation now.

## Implementation Details
- Diagnostics tab:
  - Render Kerala transfer confusion matrix from `expected_nyc_label` vs `predicted_nyc_label`.
  - Render confidence distribution using `prediction_confidence` and `correct_reference`.
  - Render district accuracy using `city_or_district`.
  - Render a metrics comparison table/card for NYC vs Kerala using loaded JSON reports.
- Benchmark tab:
  - Use the current filtered dataset, not the full dataset blindly.
  - Compare:
    - Equal Split
    - Count-Only Split
    - Current Model 4-Factor Split
  - Show a grouped sector bar chart.
  - Show a table with model deltas versus each baseline.
- Keep all functions defensive:
  - Empty filtered data returns placeholder plots/tables.
  - Missing Kerala prediction columns returns placeholder diagnostics.
  - Percentages should total approximately 100.

## Test Plan
- Run app import check:
  - `uv run --isolated --with-requirements requirements.txt python -c "import app; app.build_app()"`
- Run lightweight helper checks:
  - Load `kerala_transfer_evaluated.csv.gz`.
  - Build all new diagnostic figures.
  - Build benchmark dataframe from a filtered Kerala slice.
- Launch smoke test if time/rate limits allow:
  - `uv run --isolated --with-requirements requirements.txt python app.py`
- Confirm:
  - Existing Overview, Resource Splitup, Prediction Demo, and Data Explorer still load.
  - Diagnostics plots render without exceptions.
  - Benchmark chart/table render without exceptions.
  - No simulation UI is added in this pass.

## Deferred
- Resource split simulation with sliders.
- Feature-engineering/retraining.
- Ablation benchmarking.
- Plot export/download buttons.
