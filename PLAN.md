# Add Metrics Diagnostics, Benchmarks, and Resource Split Simulation

## Summary
Build a focused app enhancement using the useful pieces from `new_implementations/` while ignoring the noisier feature-engineering/retraining ideas for now. The implementation should add three demo-ready capabilities to the existing Gradio app:

- Metrics/detail plots for NYC and Kerala evaluation results.
- Allocation benchmark comparing current 4-factor split against simpler baselines.
- A basic resource split simulation so users can change formula weights and see how recommendations shift.

The app remains a runtime dashboard only. No retraining pipeline changes are required.

## Key Changes
- Add two new production modules based on `new_implementations/benchmark.py` and `new_implementations/plots.py`:
  - `src/benchmark.py`
  - `src/diagnostic_plots.py` or fold plot helpers into `src/ui_components.py`
- Do not integrate `new_implementations/feature_engineering.py` in this pass. It is more useful for a later retraining/ablation sprint and would expand scope.
- Do not copy `APP_PATCH.py` directly. Use it only as guidance, because it references placeholder variables like `kerala_df` and `model_scores`.

## Implementation Changes
- Metrics/details plots:
  - Add a new **Diagnostics** tab in `app.py`.
  - Show Kerala transfer confusion matrix using `expected_nyc_label` vs `predicted_nyc_label`.
  - Show confidence distribution split by `correct_reference`.
  - Show district-level transfer accuracy using `city_or_district`.
  - Show a compact metrics table/cards for:
    - NYC accuracy, macro F1, top-3 accuracy, train rows, test rows, class count.
    - Kerala rows, coverage, accuracy, macro F1, top-3 accuracy.
- Benchmark:
  - Add a new **Benchmark** tab.
  - Compute three allocation strategies for the currently filtered data:
    - Equal Split: uniform sector allocation.
    - Count-Only: raw complaint share by sector.
    - Model 4-Factor: existing `compute_resource_split`.
  - Display:
    - grouped bar chart by sector,
    - benchmark table,
    - delta columns: model minus equal split, model minus count-only.
- Basic simulation:
  - Add a **Simulation** tab or a subsection inside **Resource Splitup**.
  - Add sliders for formula weights:
    - recent complaint share,
    - positive trend growth,
    - anomaly pressure,
    - closure-delay pressure.
  - Add a numeric floor slider, default `5%`.
  - Normalize weights automatically if they do not sum to 100%.
  - Recompute simulated allocation from the same filtered dataset and anomaly frame.
  - Show:
    - simulated pie/bar chart,
    - comparison table: current load, default recommended split, simulated split, delta,
    - short explanation naming the sector with the biggest increase and biggest decrease.
- Refactor allocation logic:
  - Keep existing `compute_resource_split(filtered, anomalies)` behavior unchanged.
  - Add a new helper such as `compute_resource_split_with_weights(filtered, anomalies, weights, floor)` used by both default and simulation paths.
  - Existing dashboard outputs should remain visually and numerically stable with default weights `50/20/20/10` and floor `5%`.

## Test Plan
- Run the app smoke path:
  - `uv run --isolated --with-requirements requirements.txt python app.py`
  - Confirm existing tabs still load.
- Validate diagnostics:
  - Confusion matrix renders for Kerala view.
  - Confidence distribution renders with correct/incorrect groups.
  - District accuracy renders for all 14 Kerala districts.
  - Empty-filter state returns placeholder plots, not crashes.
- Validate benchmark:
  - Equal Split totals approximately 100%.
  - Count-Only totals approximately 100%.
  - Model 4-Factor totals approximately 100%.
  - Delta table values equal model percentage minus baseline percentage.
- Validate simulation:
  - Default slider values reproduce the existing recommended split.
  - Increasing anomaly weight visibly changes simulated allocation when anomalies exist.
  - Setting all weights to zero falls back safely to equal/floor-normalized allocation.
  - Floor slider prevents any sector from dropping below the selected minimum after normalization.

## Assumptions
- The intended folder is `new_implementations/`; no top-level `implementations/` folder exists in the current repo.
- The priority is demo-visible functionality, not retraining or feature ablation.
- Current regenerated Kerala metrics remain the source of truth:
  - coverage `100.0%`,
  - accuracy `0.889`,
  - macro F1 `0.709`,
  - top-3 accuracy `0.961`.
- Keep dependencies unchanged; use existing `pandas`, `plotly`, `gradio`, and `scikit-learn`.
