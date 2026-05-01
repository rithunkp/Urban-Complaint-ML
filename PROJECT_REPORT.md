# Urban Complaint ML
## Project Report

### Objective

Urban Complaint ML is a civic decision-support project that turns complaint records into operational insight. The system is designed to help municipal teams answer three practical questions:

- What kind of service pressure is building right now?
- Where are abnormal complaint spikes happening?
- How should limited municipal attention be distributed across sectors?

The project is presented through a Gradio dashboard, but its core value is the technical pipeline behind the UI: supervised complaint classification, transfer-style evaluation for Kerala-facing use, anomaly detection, and explainable resource allocation.

### Problem Statement

Municipal complaint data often exists in large volumes but is difficult to convert into clear action. Raw records tell us what was reported, but not how to prioritize sector response across roads, drainage, water, sanitation, lighting, traffic, and other public-service areas.

This project addresses that gap by combining:

- complaint classification
- anomaly-aware trend analysis
- district-level exploration
- explainable sector-level allocation recommendations

### Data and Processing

#### NYC reference backbone

The main supervised model is trained on NYC 311 service requests. The pipeline uses intake-time text fields aligned to realistic inference conditions and builds a stable class set from frequent complaint categories.

Key processing decisions:

- normalize complaint records into a canonical schema
- preserve intake-time text fields for inference alignment
- keep only classes with enough support for stable training
- cap the class count to maintain clarity and control

#### Kerala operational layer

The Kerala-facing view is built as a transfer-style operational layer. A Kerala-oriented dataset is generated with district coverage and complaint templates aligned to expected municipal service categories.

This allows the project to:

- show district-level analysis in a Kerala framing
- evaluate how a backbone trained on NYC transfers to a different complaint context
- support dashboard exploration and scenario testing

### Model Design

The classification pipeline uses:

- **TF-IDF vectorization**
- **Logistic Regression**
- **balanced class weighting**

This model family was chosen because it offers a strong tradeoff between performance, interpretability, and implementation simplicity. For an academic and civic-operations setting, a transparent baseline is preferable to a harder-to-explain black-box model.

### Evaluation Summary

#### NYC backbone metrics

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

#### Kerala transfer metrics

| Metric | Value |
|---|---:|
| Evaluated Rows | 50,000 |
| Reference-Label Coverage | 70.3% |
| Covered-Row Accuracy | 0.355 |
| Covered-Row Macro F1 | 0.240 |
| Covered-Row Top-3 Accuracy | 0.553 |

### Visual Evidence

#### Backbone class distribution evidence

The backbone model retains 18 high-support complaint classes rather than attempting to learn from a long tail of sparse categories. This improves stability and makes the evaluation easier to defend academically.

Examples of retained categories include:

- Illegal Parking
- Noise - Residential
- Street Condition
- Water System
- Street Light Condition

#### Operational evidence

The dashboard exposes:

- KPI cards for complaint volume, category count, regional coverage, and median closure time
- anomaly markers over time using rolling z-scores
- region-versus-sector pressure heatmaps
- resource allocation charts with direct explanation of the formula
- a scenario simulator that shows how allocation shifts under changed sector pressure

### Anomaly Detection Method

The anomaly layer uses a rolling statistical approach:

- rolling mean
- rolling standard deviation
- z-score thresholding

This method was selected because it is easy to explain, deterministic, and appropriate for a classroom project where interpretability matters as much as behavior.

### Resource Allocation Logic

The allocation engine is rule-based and explainable. The recommendation score combines:

- 50% recent complaint share
- 20% positive trend growth
- 20% anomaly pressure
- 10% closure-delay pressure

Each sector receives a minimum fairness floor of 5% before the final recommendation is normalized. This prevents the system from starving smaller but still essential municipal services.

### Scenario Simulator

The project includes a scenario simulator inside the resource-allocation workflow. It allows the user to:

- choose a sector
- apply a complaint-demand increase or decrease
- recompute the recommended allocation locally
- compare baseline and scenario recommendations

This feature is valuable because it moves the project from passive analytics into planning-oriented decision support.

### Key Findings

- A lightweight interpretable text model can perform strongly on the NYC backbone task.
- Transfer performance is meaningfully lower in Kerala-facing evaluation, which honestly demonstrates domain-shift challenges.
- Complaint analytics become more useful when paired with anomaly detection and allocation logic rather than prediction alone.
- The project is strongest as a civic decision-support system, not only as a machine-learning classifier.

### Limitations

- Kerala-facing evaluation is transfer-oriented and depends on generated operational data rather than a full real-world labeled Kerala complaint corpus.
- Some Kerala complaint categories do not map cleanly into the retained NYC class set.
- The allocation engine is heuristic and explainable, but not learned from optimization targets.
- Hosted deployment still requires deliberate runtime-artifact packaging.

### Future Work

- Replace generated Kerala evaluation data with real municipality complaint records.
- Add richer local explainability for individual model predictions.
- Extend scenario planning to district-level intervention comparisons.
- Formalize production deployment around build-time or managed artifact storage.

### Conclusion

Urban Complaint ML demonstrates a complete project arc:

- real-world data backbone
- supervised text modeling
- transfer-style evaluation
- anomaly analytics
- explainable resource planning
- a presentation-ready decision-support interface
