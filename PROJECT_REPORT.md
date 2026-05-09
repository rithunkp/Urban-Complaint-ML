# Urban Complaint ML
## Comprehensive Project Report and Claude Data-Preparation Guide

### 1. Executive Summary

Urban Complaint ML is a civic decision-support project built to transform complaint records into operational insight for municipal teams. The project combines:

- complaint classification using an NYC 311 training backbone
- Kerala-oriented transfer evaluation on a generated district-level municipal dataset
- anomaly-aware trend analysis
- explainable sector-level resource allocation
- a Gradio dashboard for exploration and demonstration

This report has two goals:

- document the project in a submission-ready format
- provide a practical guide for importing the project data into Claude for feature engineering, exploratory analysis, prompt-assisted ideation, and downstream modeling support

The core principle is that Claude should help with feature design, reasoning, code generation, validation plans, and documentation, while the actual transformations and experiments remain reproducible in the local Python pipeline.

### 2. Project Objective

Municipal complaint systems usually collect large volumes of records but do not directly answer operational questions such as:

- Which service sectors are under the most pressure right now?
- Where are abnormal spikes happening?
- How should limited attention or field resources be distributed?

Urban Complaint ML addresses that gap by framing complaint analytics as a practical planning problem rather than a raw reporting problem.

### 3. System Scope

The current repository supports four major workflows:

- offline training of an interpretable complaint classifier on NYC 311 records
- generation of a Kerala-oriented transfer dataset with 14-district coverage
- evaluation of model transfer behavior across domains
- interactive dashboard review of complaints, anomalies, and resource split recommendations

The system is not a live production municipal backend. It is an academic and demonstration-focused decision-support product with an explainable pipeline.

### 4. Repository Structure

Top-level layout:

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

Key file responsibilities:

- `app.py`: Gradio application entry point
- `scripts/train_nyc_model.py`: trains the classifier and exports runtime artifacts
- `scripts/generate_kerala_transfer_set.py`: generates the Kerala dataset and evaluates transfer behavior
- `src/data_pipeline.py`: ingestion, canonicalization, sampling, persistence
- `src/modeling.py`: training, metrics, inference, transfer evaluation
- `src/kerala_generator.py`: synthetic Kerala complaint generation
- `src/resource_allocation.py`: explainable allocation scoring
- `reports/*.json`: metric outputs used by the app and documentation

### 5. End-to-End Data Flow

The data flow is:

1. Load or download the NYC 311 CSV.
2. Normalize raw NYC records into a canonical schema.
3. Construct inference-aligned `text_input` using intake-time fields only.
4. Filter and train a stable complaint classifier on high-support classes.
5. Export a model bundle, metrics report, and compressed runtime dataset.
6. Generate a Kerala municipal dataset with district, seasonality, and event windows.
7. Evaluate the NYC model against expected NYC-mapped labels in the Kerala dataset.
8. Export evaluated Kerala records and Kerala transfer metrics.
9. Load all runtime assets into the Gradio dashboard.

### 6. Data Sources

#### 6.1 NYC reference dataset

The training backbone uses NYC 311 service-request data. The ingestion logic resolves the dataset from:

- a local CSV placed in `data/raw/`
- or `kagglehub` download fallback for the dataset slug `new-york-city/ny-311-service-requests`

The source fields retained during ingestion are:

- `Unique Key`
- `Created Date`
- `Closed Date`
- `Complaint Type`
- `Descriptor`
- `Location Type`
- `City`
- `Borough`
- `Latitude`
- `Longitude`
- `Status`
- `Open Data Channel Type`

#### 6.2 Kerala operational dataset

The Kerala-facing dataset is synthetic but schema-aligned. It is created in `src/kerala_generator.py` using:

- 14 district centers with approximate latitude and longitude anchors
- complaint templates mapped to expected NYC labels
- district weights to vary relative volume
- seasonal sector weighting
- named event windows for spikes
- stochastic closure behavior and status generation

This makes the Kerala layer useful for:

- transfer-style evaluation
- district-level dashboard presentation
- category mapping exercises
- prompt-assisted feature engineering and segmentation ideation

### 7. Canonical Schema

The main canonical columns defined in `src/domain.py` are:

| Column | Type | Meaning |
|---|---|---|
| `request_id` | string | Unique complaint identifier |
| `created_at` | datetime | Complaint creation timestamp |
| `closed_at` | datetime / null | Closure timestamp if available |
| `complaint_type` | string | Source or localized complaint label |
| `descriptor` | string | Detailed complaint description |
| `location_type` | string | Intake location category |
| `city_or_district` | string | City or district name |
| `region` | string | Borough or district grouping |
| `latitude` | float / null | Latitude |
| `longitude` | float / null | Longitude |
| `status` | string | Open or Closed status |
| `channel` | string | Intake channel |
| `closure_hours` | float / null | Derived resolution duration in hours |
| `sector` | string | Mapped municipal sector |

Additional columns appear depending on the file:

- `source_domain`
- `localized_complaint_type`
- `text_input`
- `event_name`
- `expected_nyc_label`
- `predicted_nyc_label`
- `predicted_localized_label`
- `predicted_sector`
- `prediction_confidence`
- `reference_label_covered`
- `correct_reference`

### 8. Runtime Artifact Inventory

After running the scripts, the project expects these files:

#### 8.1 Processed datasets

- `data/processed/nyc_runtime.csv.gz`
- `data/processed/kerala_transfer.csv.gz`
- `data/processed/kerala_transfer_evaluated.csv.gz`

#### 8.2 Artifacts

- `data/artifacts/complaint_model_bundle.joblib`
- `data/artifacts/runtime_manifest.json`

#### 8.3 Reports

- `reports/nyc_metrics.json`
- `reports/kerala_transfer_metrics.json`

#### 8.4 What each file is for

`nyc_runtime.csv.gz`

- normalized NYC sample for dashboard analytics and training-aligned inspection
- includes `text_input`
- best file for supervised feature engineering against the backbone labels

`kerala_transfer.csv.gz`

- generated Kerala complaints before evaluation
- includes localized labels and `expected_nyc_label`
- useful for domain-feature ideation and label-mapping analysis

`kerala_transfer_evaluated.csv.gz`

- Kerala dataset plus model predictions and confidence
- best file for transfer-error analysis, drift analysis, and misclassification study

`complaint_model_bundle.joblib`

- serialized scikit-learn pipeline and metrics
- not intended for direct upload to Claude

### 9. Training Pipeline

The training flow is implemented primarily in:

- `scripts/train_nyc_model.py`
- `src/data_pipeline.py`
- `src/modeling.py`

Important defaults:

- `sample_size = 180000`
- `min_examples = 250`
- `max_classes = 18`
- `max_rows = 120000`

Training-frame preparation:

- uses `complaint_type`, `descriptor`, `location_type`, and `text_input`
- keeps only rows where `text_input` length is at least 10 characters
- keeps only frequent complaint classes
- optionally downsamples to maintain a controlled training size

Text representation:

- `TfidfVectorizer`
- English stop words removed
- `max_features = 7000`
- `ngram_range = (1, 2)`
- `min_df = 3`
- `sublinear_tf = True`

Classifier:

- `LogisticRegression`
- `max_iter = 700`
- `class_weight = "balanced"`

Train/test split:

- `test_size = 0.2`
- `random_state = 42`
- stratified by complaint type

### 10. Evaluation Summary

#### 10.1 NYC backbone metrics

| Metric | Value |
|---|---:|
| Accuracy | 0.989 |
| Macro Precision | 0.995 |
| Macro Recall | 0.989 |
| Macro F1 | 0.992 |
| Top-3 Accuracy | 1.000 |
| Retained Classes | 18 |
| Train Rows | 96,000 |
| Test Rows | 24,000 |

#### 10.2 Kerala transfer metrics

| Metric | Value |
|---|---:|
| Evaluated Rows | 50,000 |
| Reference-Label Coverage | 70.3% |
| Covered-Row Accuracy | 0.355 |
| Covered-Row Macro F1 | 0.240 |
| Covered-Row Top-3 Accuracy | 0.553 |

#### 10.3 Interpretation

- Backbone performance is strong because the problem is narrowed to frequent, stable classes.
- Transfer performance is lower because Kerala-facing complaints differ in language, framing, and label structure.
- This gap is useful academically because it highlights domain shift honestly.

### 11. Sector Mapping and Resource Allocation

The municipal sectors used by the project are:

- roads
- drainage_flooding
- water_supply
- waste_sanitation
- street_lighting
- traffic_signals
- public_safety_other

The allocation formula in `src/resource_allocation.py` combines:

- 50% recent complaint share
- 20% positive trend growth
- 20% anomaly pressure
- 10% closure-delay pressure

There is a 5% fairness floor per sector before normalization.

This part of the project is especially suitable for Claude-assisted ideation because you can ask for:

- alternative weighting strategies
- extra interpretable signals
- fairness constraints
- district-aware extensions
- scenario-planning formulations

### 12. Kerala Generator Design

The generator produces realistic-enough complaint patterns using:

- district volume weighting
- monthly seasonality
- event-specific spike multipliers
- complaint templates with structured descriptors
- synthetic closure-time distributions by sector
- intake channel variation

Named event windows:

- `flood_like_spike`
- `landslide_road_washout`
- `water_shortage_spike`

These event windows are useful for anomaly analysis and feature-engineering prompts because they create interpretable temporal structure.

### 13. Current Dashboard Capabilities

The current app exposes:

- Overview
- Resource Splitup
- Prediction Demo
- Data Explorer

The app loads:

- `nyc_runtime.csv.gz`
- `kerala_transfer_evaluated.csv.gz`
- `complaint_model_bundle.joblib`
- `nyc_metrics.json`

If these files are missing, the app stops at startup and asks the user to run the training and generation scripts first.

### 14. Opportunities for Feature Engineering

Claude can be especially useful for feature-engineering ideation when grounded on the canonical schema.

Recommended feature families:

#### 14.1 Text features

- descriptor length
- complaint urgency keywords
- rain/flood/water-outage keywords
- road-damage terminology
- n-gram phrases specific to location types
- text cleanliness or ambiguity signals
- counts of numeric references, landmarks, or district mentions

#### 14.2 Temporal features

- hour of day
- day of week
- month
- monsoon season flag
- summer water-shortage flag
- holiday or festival proxy windows
- time since last same-sector complaint in a district

#### 14.3 Geographic and district features

- district frequency prior
- district-sector interaction rates
- latitude and longitude clustering zone
- urban-core versus peripheral proxy
- district historical closure profile

#### 14.4 Operational features

- intake channel
- open versus closed ratio by sector
- rolling complaint counts
- rolling anomaly score
- closure-delay percentile by sector
- district-sector recent pressure score

#### 14.5 Cross-domain transfer features

- localized label to expected NYC label mapping coverage
- confidence gap between top-1 and top-2 predictions
- whether predicted label is inside covered transfer classes
- sector agreement between expected and predicted labels

### 15. What Claude Should and Should Not Do

Claude is best used for:

- suggesting new features
- generating preprocessing code
- helping design ablation experiments
- writing data dictionaries
- finding suspicious leakage risks
- proposing evaluation slices
- explaining domain-shift patterns
- generating polished report language

Claude should not be the system of record for:

- raw data storage
- final numerical validation
- direct mutation of master training files without review
- unverified model-performance claims

The safe workflow is:

1. Export a clean subset from this repository.
2. Upload the subset plus a schema description to Claude.
3. Ask Claude for feature ideas, code, and validation logic.
4. Bring the generated code back into the local repo.
5. Run and verify locally with Python and pandas or scikit-learn.

### 16. Best Files to Upload to Claude

Use different files depending on the task.

#### 16.1 For feature engineering on the classifier

Upload or sample from:

- `data/processed/nyc_runtime.csv.gz`

Best columns:

- `complaint_type`
- `descriptor`
- `location_type`
- `text_input`
- `region`
- `channel`
- `created_at`
- `closure_hours`
- `sector`

Target:

- `complaint_type`

#### 16.2 For transfer-analysis feature engineering

Upload or sample from:

- `data/processed/kerala_transfer_evaluated.csv.gz`

Best columns:

- `complaint_type`
- `descriptor`
- `location_type`
- `region`
- `channel`
- `created_at`
- `sector`
- `expected_nyc_label`
- `predicted_nyc_label`
- `predicted_sector`
- `prediction_confidence`
- `reference_label_covered`
- `correct_reference`
- `event_name`

Targets depending on task:

- `expected_nyc_label`
- `correct_reference`
- `predicted_sector`

#### 16.3 For district-pressure and planning analysis

Upload or sample from:

- `data/processed/kerala_transfer_evaluated.csv.gz`

Best columns:

- `created_at`
- `region`
- `sector`
- `status`
- `closure_hours`
- `event_name`
- `prediction_confidence`

### 17. How to Prepare a Claude-Friendly Export

Do not upload the full raw dataset first. Claude works better when the input is compact, well-labeled, and task-specific.

Recommended preparation strategy:

#### 17.1 Build a working sample

- Use 3,000 to 20,000 rows depending on the task.
- Keep only the columns relevant to the prompt.
- Include one data dictionary file.
- Preserve timestamps in ISO format.
- Convert missing values to a clear representation like blank or `null`.

#### 17.2 Split by purpose

Create separate files for:

- classification feature engineering
- transfer-error analysis
- allocation and operations analysis

#### 17.3 Include metadata alongside the data

For every Claude upload, include:

- dataset purpose
- row count
- column dictionary
- target variable
- known leakage risks
- known synthetic columns
- expected output format

### 18. Recommended Claude Import Package

For best results, prepare a folder or zip with:

- `sample_data.csv`
- `schema_dictionary.md`
- `task_brief.md`
- `metrics_snapshot.json`

Suggested contents:

`sample_data.csv`

- task-specific subset only

`schema_dictionary.md`

- one line per column
- target column clearly marked
- note whether the field is raw, derived, predicted, or synthetic

`task_brief.md`

- what problem you want Claude to help with
- what kind of features you want
- what output you expect

`metrics_snapshot.json`

- current baseline metrics from the project
- helps Claude reason about improvement goals

### 19. Example Schema Dictionary Template for Claude

```markdown
# Schema Dictionary

- complaint_type: target label for supervised classification
- descriptor: free-text complaint description
- location_type: intake location category
- text_input: descriptor + location_type combined at inference time
- region: borough or district
- channel: intake channel such as mobile app or call centre
- created_at: complaint timestamp
- closure_hours: time to closure in hours, if available
- sector: mapped municipal service sector
- expected_nyc_label: reference NYC-aligned label for Kerala transfer evaluation
- predicted_nyc_label: model prediction from NYC-trained classifier
- prediction_confidence: top prediction probability
- correct_reference: whether predicted_nyc_label matches expected_nyc_label
```

### 20. Example Claude Prompts

#### 20.1 Prompt for supervised feature engineering

```text
You are helping with feature engineering for a complaint classification project.

Context:
- The target column is complaint_type.
- The current model uses TF-IDF on text_input and logistic regression.
- I want additional interpretable features that can be engineered in pandas or scikit-learn without causing target leakage.

Tasks:
1. Review the schema and propose 20 high-value engineered features.
2. Group them into text, temporal, geographic, and operational features.
3. Mark each feature as low, medium, or high implementation effort.
4. Flag any leakage risks.
5. Write Python feature-engineering code for the top 10 features.
```

#### 20.2 Prompt for transfer-error analysis

```text
You are helping analyze domain-shift errors in a Kerala municipal complaint dataset evaluated by an NYC-trained model.

Context:
- expected_nyc_label is the transfer reference label.
- predicted_nyc_label is the model output.
- correct_reference indicates whether the transfer prediction matched the reference.

Tasks:
1. Suggest features that could explain transfer errors.
2. Identify which columns are useful for confidence calibration analysis.
3. Propose a table of evaluation slices by district, sector, event window, and channel.
4. Write Python code to compute these slices.
```

#### 20.3 Prompt for allocation-feature ideation

```text
You are helping design explainable features for municipal resource allocation.

Context:
- The current allocation formula uses recent complaint share, positive trend growth, anomaly pressure, and closure delay.
- I want new interpretable features, not black-box optimization.

Tasks:
1. Propose 15 additional explainable signals.
2. Explain which of them can be computed from the existing schema.
3. Suggest a revised scoring formula.
4. Describe fairness and stability checks before adoption.
```

### 21. Leakage and Validation Warnings

Be careful with these fields during feature engineering:

- `closed_at`
- `closure_hours`
- `status`

These can be valid for operations analysis but may create leakage for intake-time complaint classification if used incorrectly. For the classifier, the project intentionally aligns inference to intake-time information through:

- `descriptor`
- `location_type`
- combined `text_input`

Use `closure_hours` and `status` mainly for:

- allocation logic
- service-performance analysis
- post-hoc diagnostic modeling

Do not use future-known fields in a model that is supposed to predict at complaint intake time.

### 22. Synthetic-Data Caveats

The Kerala dataset is generated, not collected from a live municipal system. That means:

- patterns are realistic enough for experimentation, not official policy use
- district and event patterns are intentionally structured
- event windows can make anomaly signals easier to detect than in noisier real data
- feature ideas from Kerala should still be validated on real complaint streams if available

This is important to mention in academic reporting and also when giving data to Claude, so the model does not overstate real-world certainty.

### 23. Suggested Local Export Workflow Before Claude

Recommended workflow:

1. Generate the runtime artifacts locally.
2. Open the processed CSV you want to use.
3. Make a task-specific sample.
4. Save the sample and schema dictionary.
5. Upload those files to Claude.
6. Ask for feature ideas and code.
7. Apply the code locally in a notebook or script.
8. Measure whether the new features improve metrics or analysis quality.

### 24. Minimal Export Examples

#### 24.1 Classification sample package

Include:

- 10,000 NYC rows
- target `complaint_type`
- text and metadata columns only

#### 24.2 Transfer-error package

Include:

- 10,000 Kerala evaluated rows
- expected and predicted labels
- confidence and event columns

#### 24.3 Allocation-analysis package

Include:

- Kerala rows grouped or sampled by district and sector
- timestamps
- closure metrics
- status

### 25. Recommended Outputs to Ask Claude For

When you upload the data, ask Claude to produce:

- a ranked feature list
- leakage review notes
- pandas feature-engineering code
- scikit-learn preprocessing suggestions
- ablation-study design
- error-analysis groupings
- report-ready explanations of why the features matter

That gives you both technical value and material you can directly use in a project presentation or viva.

### 26. Current Limitations

- The Kerala layer is synthetic and transfer-oriented.
- The backbone is intentionally simplified to stable classes.
- The hosted app depends on prebuilt runtime artifacts.
- The current allocation engine is heuristic, not optimized from labeled intervention outcomes.
- The dashboard is strong for demo and analysis, but not yet a production municipal platform.

### 27. Future Work

- replace synthetic Kerala data with real municipality complaint records
- add formal feature-ablation notebooks
- evaluate richer but still interpretable text models
- add district-level scenario simulation in a stable way
- build a repeatable Claude-to-local experimentation template for feature engineering

### 28. Conclusion

Urban Complaint ML is strongest when presented as a full civic analytics pipeline rather than only a dashboard. The project already contains:

- a real-world training backbone
- clear canonicalization logic
- an interpretable classifier
- transfer-style domain evaluation
- anomaly-aware operations analytics
- explainable resource allocation

For Claude-based feature engineering, the best approach is not to upload everything blindly. Instead:

- choose the correct processed file
- build a clean task-specific sample
- include a schema dictionary and task brief
- ask Claude for features, code, validation slices, and risk checks
- verify all outputs locally

That workflow makes the project stronger technically, easier to explain academically, and much more useful for structured experimentation.
