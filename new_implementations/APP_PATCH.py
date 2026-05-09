# APP_PATCH.md
# How to wire the new modules into app.py
#
# This is not a full app.py replacement — it shows exactly what to add/change.
# Search for the markers below and insert the corresponding blocks.

# ─────────────────────────────────────────────────────────────────────────────
# 1. IMPORTS  (add after existing src imports)
# ─────────────────────────────────────────────────────────────────────────────

from src.benchmark import compute_benchmark, benchmark_delta
from src.plots import (
    allocation_benchmark_chart,
    transfer_confusion_matrix,
    confidence_distribution,
    per_district_accuracy,
    sector_load_vs_recommended,
)

# ─────────────────────────────────────────────────────────────────────────────
# 2. BENCHMARK TAB  (add as a new tab inside gr.Blocks / gr.TabbedInterface)
# ─────────────────────────────────────────────────────────────────────────────

with gr.Tab("Benchmark"):
    gr.Markdown("""
    ## Allocation Strategy Comparison
    Compare the model's 4-factor recommendation against two baselines:
    - **Equal Split** — uniform allocation, no data used
    - **Count-Only** — raw complaint frequency only, no trend / anomaly / delay signal
    """)

    with gr.Row():
        benchmark_btn = gr.Button("Compute Benchmark", variant="primary")

    benchmark_chart = gr.Plot(label="Strategy Comparison")
    benchmark_table = gr.Dataframe(
        label="Allocation Table with Deltas (pp = percentage points vs model)",
        wrap=True,
    )

    def run_benchmark():
        # kerala_df and model_scores come from the existing app state / loaders
        # Replace these with your actual variable names
        bdf = compute_benchmark(kerala_df, model_scores)
        bdf_delta = benchmark_delta(bdf)
        fig = allocation_benchmark_chart(bdf)
        return fig, bdf_delta

    benchmark_btn.click(run_benchmark, outputs=[benchmark_chart, benchmark_table])

# ─────────────────────────────────────────────────────────────────────────────
# 3. DIAGNOSTICS TAB  (add as another new tab)
# ─────────────────────────────────────────────────────────────────────────────

with gr.Tab("Diagnostics"):
    gr.Markdown("""
    ## Model Diagnostics
    Four views into model behaviour on the Kerala transfer set.
    """)

    with gr.Tabs():
        with gr.Tab("Transfer Confusion Matrix"):
            gr.Markdown(
                "Row-normalised confusion between expected and predicted NYC labels. "
                "Bright diagonal = good transfer. Off-diagonal clusters = domain-shift errors."
            )
            cm_plot = gr.Plot()

        with gr.Tab("Confidence Distribution"):
            gr.Markdown(
                "Prediction confidence split by correct vs incorrect. "
                "A well-calibrated model concentrates correct predictions at high confidence."
            )
            conf_plot = gr.Plot()

        with gr.Tab("District Accuracy"):
            gr.Markdown(
                "Per-district transfer accuracy. "
                "Coastal districts during monsoon season tend to perform differently from inland districts."
            )
            district_plot = gr.Plot()

        with gr.Tab("Sector Load vs Recommended"):
            gr.Markdown(
                "Each point is a sector. "
                "Points above the dashed diagonal are upweighted by trend / anomaly / closure-delay signals beyond their raw complaint share."
            )
            scatter_plot = gr.Plot()

    diagnostics_btn = gr.Button("Load Diagnostic Plots", variant="primary")

    def load_diagnostics():
        # kerala_df, model_scores, sector_load — from existing app loaders
        fig_cm     = transfer_confusion_matrix(kerala_df)
        fig_conf   = confidence_distribution(kerala_df)
        fig_dist   = per_district_accuracy(kerala_df)
        fig_scatter = sector_load_vs_recommended(sector_load, model_scores)
        return fig_cm, fig_conf, fig_dist, fig_scatter

    diagnostics_btn.click(
        load_diagnostics,
        outputs=[cm_plot, conf_plot, district_plot, scatter_plot],
    )

# ─────────────────────────────────────────────────────────────────────────────
# 4. FEATURE ENGINEERING  (call in data_pipeline.py or train_nyc_model.py)
#    Add after df is loaded and canonicalized, before training.
#    Only add_text_features, add_temporal_features, add_geographic_features,
#    add_operational_features are intake-safe for the classifier.
#    add_transfer_features is for the Kerala evaluated set only.
# ─────────────────────────────────────────────────────────────────────────────

# In scripts/train_nyc_model.py, after df = pipeline.load():
from src.feature_engineering import (
    add_text_features,
    add_temporal_features,
    add_geographic_features,
    add_operational_features,
)
df = add_text_features(df)
df = add_temporal_features(df)
df = add_geographic_features(df)
df = add_operational_features(df)

# Then extend the ColumnTransformer to include numeric features alongside TF-IDF:
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

NUMERIC_FEATURES = [
    "text_length", "word_count", "avg_word_length",
    "has_urgency_keyword", "has_location_reference",
    "exclamation_flag", "negative_word_count", "descriptor_is_short",
    "hour_of_day", "day_of_week", "is_weekend", "month",
    "is_monsoon", "is_festival_season", "is_morning_rush",
    "is_coastal_district", "district_urbanization_rank",
    "has_coordinates", "channel_is_digital",
]

# Replace the existing pipeline's TfidfVectorizer step with a ColumnTransformer:
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import scipy.sparse as sp

preprocessor = ColumnTransformer(
    transformers=[
        ("tfidf", TfidfVectorizer(
            max_features=7000,
            ngram_range=(1, 2),
            min_df=3,
            sublinear_tf=True,
            stop_words="english",
        ), "text_input"),
        ("num", StandardScaler(), NUMERIC_FEATURES),
    ],
    remainder="drop",
)

model_pipeline = Pipeline([
    ("features", preprocessor),
    ("clf", LogisticRegression(
        max_iter=700,
        class_weight="balanced",
        random_state=42,
    )),
])
# Then fit: model_pipeline.fit(df[["text_input"] + NUMERIC_FEATURES], df["complaint_type"])

# ─────────────────────────────────────────────────────────────────────────────
# 5. SECTOR LOAD HELPER  (needed by sector_load_vs_recommended)
#    Add alongside or inside resource_allocation.py
# ─────────────────────────────────────────────────────────────────────────────

def compute_sector_load(df, sector_col="sector"):
    """Raw complaint share per sector — no smoothing, no floors."""
    counts = df[sector_col].value_counts(normalize=True)
    from src.benchmark import SECTORS
    return {s: float(counts.get(s, 0)) for s in SECTORS}
